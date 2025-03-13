import pathlib
from typing import Optional

import dask
import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem

from fegrow import RMol
from fegrow.receptor import ForceField


class Settings(BaseModel):
    """A class to configure the runtime settings of a high throughput scoring."""

    num_confs: int = Field(
        50,
        description="The number of initial conformers which should be generated using RDKit.",
    )
    conf_rms: float = Field(
        0.2,
        description="The rms cutoff in angstrom for which two conformers are considered the same. Used while generating the conformations.",
    )
    ligand_force_field: ForceField = Field(
        "openff",
        description="The force field model to use for the small molecule during the restrained optimisation.",
    )
    use_ani: bool = Field(
        True,
        description="If we should attempt to use ANI2x to model the internal energies of the ligand in an ML/MM optimisation.",
    )
    sigma_scale_factor: float = Field(
        0.8,
        description="The amount the sigma of the force field should be scaled by to compensate for the rigid recptor.",
    )
    relative_permittivity: float = Field(
        4,
        description="The relative permittivity which should be used to scale the charge interactions of the system to mimic a condensed phase environment.",
    )
    water_model: Optional[str] = Field(
        None,
        description="The name of the water force field model from openmmforcefields which should be used, eg tip3p.xml",
    )
    platform_name: str = Field(
        "CPU",
        description="The name of the OpenMM platform which should be used during the geometry optimisation",
    )
    gnina_path: str = Field(
        ...,
        description="The path to the gnina executable which should be used to score the ligands.",
    )
    energy_filter: float = Field(
        2,
        description="The relative energy cutoff in kcal/mol used to select the top conformers for scoring.",
    )


@dask.delayed
def score_ligand(
    core_ligand: Chem.Mol,
    target_smiles: str,
    receptor: pathlib.Path,
    settings: Settings,
) -> dict:
    """
    Score the ligand given by the target smiles using the core ligand to restrain the geometry.

    Note:
        We assume the core does not need to be altered and is a substructure of the target ligand
    """
    # create the target ligand with Hs
    candidate_mol = Chem.MolFromSmiles(target_smiles)
    candidate_mol = Chem.AddHs(candidate_mol)
    rmol = RMol(candidate_mol)
    # set up the core as the template
    rmol._save_template(core_ligand)

    # conformer gen
    rmol.generate_conformers(
        num_conf=settings.num_confs, minimum_conf_rms=settings.conf_rms
    )
    # remove missing
    rmol.remove_clashing_confs(protein=receptor.as_posix())

    # optimise
    rmol.optimise_in_receptor(
        receptor_file=receptor,
        ligand_force_field=settings.ligand_force_field,
        use_ani=settings.use_ani,
        sigma_scale_factor=settings.sigma_scale_factor,
        relative_permittivity=settings.relative_permittivity,
        water_model=settings.water_model,
        platform_name=settings.platform_name,
    )

    if rmol.GetNumConformers() == 0:
        # set a pentalty
        cnnaffinity = 0
        cnnaffinityIC50 = 0
    else:
        # score only the lowest energy conformer
        rmol.sort_conformers(energy_range=settings.energy_filter)  # kcal/mol
        # purge all but the lowest energy conformers
        rmol = Rmol(rmol, confId=0)
        affinities = rmol.gnina(receptor_file=receptor.as_posix())
        cnnaffinity = -affinities.CNNaffinity.values[0]
        cnnaffinityIC50 = affinities["CNNaffinity->IC50s"].values[0]

    data = {
        "cnnaffinity": cnnaffinity,
        "cnnaffinityIC50": cnnaffinityIC50,
        "molecule": rmol,
    }

    return data


def load_target_ligands(ligand_file: pathlib.Path) -> list[str]:
    """
    Load a set of ligands from any RDKit supported file format.

    Note:
        For CSV we assume that the smiles have the column name "Smiles"
    """
    if ligand_file.stem.lower == "csv":
        target_molecules = pd.read_csv(ligand_file)
        return list(target_molecules.Smiles.values)

    if ligand_file.stem.lower() in ["sdf", "mol"]:
        ligands = list(Chem.SDMolSupplier(ligand_file, removeHs=False))
    elif ligand_file.stem.lower() == "smi":
        ligands = list(Chem.SmilesMolSupplier(ligand_file, remoeHs=False))
    else:
        raise RuntimeError(f"Can extract smiles from input file {ligand_file}")

    smiles = [Chem.MolToSmiles(mol) for mol in ligands]
    return smiles
