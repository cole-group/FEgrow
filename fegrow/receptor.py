import logging
import tempfile
from copy import deepcopy
from typing import List, Tuple, Union, Optional

import numpy
import parmed
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from tqdm import tqdm
from typing_extensions import Literal, get_args

from .mlp import _MLFF_NAME_TO_CLASS, AVAILABLE_ML_FORCE_FIELDS

import warnings

import subprocess

import shutil

# fix for new openmm versions
try:
    from openmm import Platform, app, openmm, unit, OpenMMException
except (ImportError, ModuleNotFoundError):
    from simtk import unit
    from simtk.openmm import app, openmm

from openff.toolkit.topology import Molecule as OFFMolecule

logger = logging.getLogger(__name__)


class NoPostMinimisationConformersError(Exception):
    """Raise if no conformers survive minimisation (due to e.g. simulation blowing up)"""


def chimera_path_check():
    # check if chimera is in the path, if not, raise an error
    if not shutil.which("chimera"):
        raise EnvironmentError(
            "Chimera is not in the PATH. Please install Chimera and ensure it is accessible from the command line."
        )


def chimera_protonate(input_file: str, output_file: str, verbose: bool = False):
    """
    Use Chimera to protonate the receptor.

    :param input_file: The name of the pdb file which contains the receptor.
    :param output_file: The name of the pdb file the fixed receptor should be wrote to.
    :param pH:The ph the pronation state should be fixed for.
    :param verbose: If True, print the Chimera output.
    """
    chimera_path_check()

    cmds = [
        "open {}".format(input_file),
        "addh hbond true",
        "write format pdb 0 {}".format(output_file),
        "close all",
    ]

    subprocess.run(
        ["chimera", "--nogui", input_file],
        input="\n".join(cmds).encode(),
        check=True,
    )


def fix_receptor(
    input_file: str,
    output_file: str,
    pH: float = 7.0,
    prefer_chimera_protonation: bool = False,
):
    """
    Use PDBFixer to correct the input and add hydrogens with the given pH.

    :param input_file: The name of the pdb file which contains the receptor.
    :param output_file: The name of the pdb file the fixed receptor should be wrote to.
    :param pH: The pH the protonation state should be fixed for.
    :param prefer_chimera_protonation: If True, use Chimera to protonate the receptor instead of PDBFixer.
    """
    fixer = PDBFixer(filename=input_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if not prefer_chimera_protonation:
        warnings.warn(
            "Using PDBFixer for protonation can lead to less accurate results than using Chimera. Please install chimera",
            UserWarning,
        )
        fixer.addMissingHydrogens(pH)
        app.PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file, "w"))

    if prefer_chimera_protonation:
        # write out a temporary file for chimera to read in
        with tempfile.NamedTemporaryFile(suffix=".pdb") as temp_pdb:
            app.PDBFile.writeFile(
                fixer.topology, fixer.positions, open(temp_pdb.name, "w")
            )
            # use chimera to protonate the file
            chimera_protonate(temp_pdb.name, output_file)


def _scale_system(
    system: openmm.System, sigma_scale_factor: float, relative_permittivity: float
):
    """
    Scale the sigma and charges of the openMM system in place.
    """
    if relative_permittivity != 1:
        charge_scale_factor = 1 / numpy.sqrt(relative_permittivity)
    else:
        charge_scale_factor = 1
    forces = {
        system.getForce(i).__class__.__name__: system.getForce(i)
        for i in range(system.getNumForces())
    }
    # assuming all nonbonded interactions are via the standard force
    nonbonded_force = forces["NonbondedForce"]
    # scale all particle parameters
    for i in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        nonbonded_force.setParticleParameters(
            i, charge * charge_scale_factor, sigma * sigma_scale_factor, epsilon
        )


ForceField = Literal["openff", "gaff"]


def optimise_in_receptor(
    ligand: Chem.Mol,
    receptor_file: Union[str, app.PDBFile],
    ligand_force_field: ForceField,
    ligand_intramolecular_mlp: Optional[AVAILABLE_ML_FORCE_FIELDS] = None,
    sigma_scale_factor: float = 0.8,
    relative_permittivity: float = 4,
    water_model: str = "tip3p.xml",
    platform_name: str = "CPU",
    ligand_indices_to_freeze: Optional[list[int]] = None,
) -> Tuple[Chem.Mol, List[float]]:
    """
    For each of the input molecule conformers optimise the system using the chosen force field with the receptor held fixed.

    Args:
        ligand:
            The ligand with starting conformers already filtered for clashes with the receptor.
        receptor_file:
            The pdb file of the fixed and pronated receptor.
        ligand_force_field:
            The base ligand force field that should be used.
        ligand_intramolecular_mlp:
            The machine learning force field that should be used for the ligand intramolecular interactions.
            If set to None, the molecular mechanics ligand_force_field is used for all ligand interactions.
            Representations of available MLPs can be imported from the `fegrow.mlp` module.
        sigma_scale_factor:
            The factor by which all sigma values should be scaled
        relative_permittivity:
            The relativity permittivity which should be used to scale all charges 1/sqrt(permittivity)
        water_model:
            If set to None, the water model is ignored. Acceptable can be found in the
            openmmforcefields package.
        platform_name:
            The OpenMM platform name, 'cuda' if available, with the 'cpu' used by default.
            See the OpenMM documentation of Platform.
        ligand_indices_to_freeze:
            The ligand indices to be frozen (relative to the ligand)

    Returns:
        A copy of the input molecule with the optimised positions.
    """

    ligand_force_fields = {
        "openff": "openff_unconstrained-2.0.0.offxml",
        "gaff": "gaff-2.11",
    }

    platform = Platform.getPlatformByName(platform_name.upper())

    # assume the receptor has already been fixed and hydrogens have been added.
    if type(receptor_file) is str:
        receptor_file = app.PDBFile(receptor_file)
    # receptor forcefield
    receptor_ff = "amber14/protein.ff14SB.xml"

    # load the molecule into openff
    openff_mol = OFFMolecule.from_rdkit(ligand, allow_undefined_stereo=True)

    forcefields = [receptor_ff]
    if water_model:
        forcefields.append(water_model)

    # now we need to make our parameterised system, make a system generator
    system_generator = SystemGenerator(
        forcefields=forcefields,
        small_molecule_forcefield=ligand_force_fields[ligand_force_field],
        cache=None,
        molecules=openff_mol,
    )
    # now make a combined receptor and ligand topology
    parmed_receptor = parmed.openmm.load_topology(
        receptor_file.topology, xyz=receptor_file.positions
    )
    parmed_ligand = parmed.openmm.load_topology(
        openff_mol.to_topology().to_openmm(), xyz=openff_mol.conformers[0]
    )
    complex_structure = parmed_receptor + parmed_ligand
    # build the complex system
    system = system_generator.create_system(complex_structure.topology)

    # work out the index of the atoms in the ligand assuming they are the last residue?
    # is this always true if we add the ligand to the receptor?
    ligand_res = list(complex_structure.topology.residues())[-1]
    ligand_idx = [atom.index for atom in ligand_res.atoms()]
    # now set all atom mass to 0 if not in the ligand
    for i in range(system.getNumParticles()):
        if i not in ligand_idx:
            system.setParticleMass(i, 0)

    if ligand_indices_to_freeze is not None:
        logger.info("Freezing ligand indices")
        for idx in ligand_indices_to_freeze:
            system.setParticleMass(ligand_idx[idx], 0)

    # Check if we want to add an intramolecular MLP
    if ligand_intramolecular_mlp is not None:
        logger.info(
            f"Using ligand intramolecular MLP: {ligand_intramolecular_mlp} "
            "for ligand intramolecular interactions and "
            f"{ligand_force_field} for intermolecular interactions."
        )

        if ligand_intramolecular_mlp not in get_args(AVAILABLE_ML_FORCE_FIELDS):
            raise ValueError(
                f"ligand_intramolecular_mlp must be one of {AVAILABLE_ML_FORCE_FIELDS}"
            )

        # get the MLP class
        ligand_intramolecular_mlp_class = _MLFF_NAME_TO_CLASS[ligand_intramolecular_mlp]

        if not ligand_intramolecular_mlp_class.is_compatible_with_molecule(openff_mol):
            raise ValueError(
                f"The ligand {ligand} is not compatible with the MLP {ligand_intramolecular_mlp}"
            )

        # create the potential
        potential = ligand_intramolecular_mlp_class.get_potential()
        # add the potential to the system
        complex_system = potential.createMixedSystem(
            complex_structure.topology, system, ligand_idx
        )

    else:  # Use the MM FF for all ligand interactions
        logger.info(
            f"Using {ligand_force_field} force field for ligand intra and "
            "inter-molecular interactions."
        )
        complex_system = system

    # scale the charges and sigma values
    _scale_system(
        system=complex_system,
        sigma_scale_factor=sigma_scale_factor,
        relative_permittivity=relative_permittivity,
    )

    # propagate the System with Langevin dynamics note integrator note used.
    time_step = 1 * unit.femtoseconds  # simulation timestep
    temperature = 300 * unit.kelvin  # simulation temperature
    friction = 1 / unit.picosecond  # collision rate
    integrator_min = openmm.LangevinIntegrator(temperature, friction, time_step)

    # set up an openmm simulation
    simulation = app.Simulation(
        complex_structure.topology, complex_system, integrator_min, platform=platform
    )

    # save the receptor coords as they should be consistent
    receptor_coords = unit.Quantity(
        parmed_receptor.coordinates.tolist(), unit=unit.angstrom
    )

    # loop over the conformers and energy minimise and store the final positions
    final_mol = deepcopy(ligand)
    final_mol.RemoveAllConformers()
    energies = []
    for i, conformer in enumerate(
        tqdm(openff_mol.conformers, desc="Optimising conformer: ", ncols=80)
    ):
        # make the ligand coords
        lig_vec = unit.Quantity([c.m.tolist() for c in conformer], unit=unit.angstrom)

        complex_coords = receptor_coords + lig_vec
        # set the initial positions
        simulation.context.setPositions(complex_coords)

        # minimize the energy
        try:
            simulation.minimizeEnergy()
        except OpenMMException as E:
            logger.warning(f"Conformer (index: {i}) minimisation failed due to: {E}")
            continue

        # write out the final coords
        min_state = simulation.context.getState(getPositions=True, getEnergy=True)
        positions = min_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        final_conformer = Chem.Conformer()
        for j, coord in enumerate(positions[ligand_idx[0] :]):
            atom_position = Point3D(*coord)
            final_conformer.SetAtomPosition(j, atom_position)

        # ignore minimised conformers that have very long bonds
        # this is a temporary fix to ANI generated
        has_long_bonds = False
        for bond in final_mol.GetBonds():
            atom_from = final_conformer.GetAtomPosition(bond.GetBeginAtomIdx())
            atom_to = final_conformer.GetAtomPosition(bond.GetEndAtomIdx())
            if atom_from.Distance(atom_to) > 3:
                has_long_bonds = True
                break
        if has_long_bonds:
            continue

        energies.append(
            min_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        )
        final_mol.AddConformer(final_conformer, assignId=True)

    if final_mol.GetNumConformers() == 0:
        raise NoPostMinimisationConformersError()

    return final_mol, energies


def sort_conformers(
    ligand: Chem.Mol, energies: List[float], energy_range: float = 5
) -> Tuple[Chem.Mol, List[float]]:
    """
    For the given molecule and the conformer energies order the energies and only keep any conformers with in the energy
    range of the lowest energy conformer.

    Note:
        This sorting is done on a copy of the molecule.

    Args:
        ligand:
            A molecule instance whose optimised conformers should be sorted.
        energies:
            The list of energies in the same order as the conformers.
        energy_range:
            The energy range (kcal/mol), above the minimum, for which conformers should be kept.
    """
    copy_mol = deepcopy(ligand)
    copy_mol.RemoveAllConformers()
    energies = numpy.array(energies)
    # normalise
    energies -= energies.min()
    # order by lowest energy
    energy_and_conformers = []
    for i, conformer in enumerate(ligand.GetConformers()):
        energy_and_conformers.append((energies[i], conformer))
    energy_and_conformers.sort(key=lambda x: x[0])
    final_energies = []
    for energy, conformer in energy_and_conformers:
        if energy <= energy_range:
            copy_mol.AddConformer(conformer, assignId=True)
            final_energies.append(energy)
    return copy_mol, final_energies
