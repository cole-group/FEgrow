import logging
from copy import deepcopy
from typing import List, Optional

import numpy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.rdMolAlign import AlignMol

logger = logging.getLogger(__name__)


class WrongCoreForMolecule(Exception):
    pass


def conformer_exists(
    mol: Chem.Mol, potential_conformer: Chem.Conformer, nonh_atom_indices, rms_limit: float = 0.5
) -> bool:
    potential_nonh_pos = potential_conformer.GetPositions()[nonh_atom_indices]
    for conformer in mol.GetConformers():
        conformer_nonh_pos = conformer.GetPositions()[nonh_atom_indices]

        dsts = numpy.sqrt(numpy.sum(numpy.square(potential_nonh_pos - conformer_nonh_pos), axis=1, keepdims=True))
        rms = numpy.sqrt(numpy.mean(numpy.square(dsts)))
        if rms < rms_limit:
            return True

    return False


def generate_conformers(
    rmol: Chem.rdchem.Mol,
    num_conf: int,
    minimum_conf_rms:float=0.5,
    flexible:Optional[List[int]]=[],
    scaffold_heavy_atoms=True,
    use_ties_mcs:bool=False,
) -> List[Chem.rdchem.Mol]:
    """
    flexible:
            The list of atomic indices on the @core_ligand that should not be constrained during the conformer generation
    :param scaffold_heavy_atoms: use only the heavy atoms in the scaffold
        and in the molecule to generate the match.
    """
    scaffold_mol = deepcopy(rmol.template)

    if scaffold_heavy_atoms:
        scaffold_mol = Chem.RemoveHs(scaffold_mol)

    scaffold_conformer = scaffold_mol.GetConformer(0)

    # fixme - check if the conformer has H, it helps with conformer generation
    rmol = deepcopy(rmol)

    # map scaffold atoms to the new molecules
    match = rmol.GetSubstructMatch(scaffold_mol)
    if match and not use_ties_mcs:
        # extract the scaffold coordinates
        coordinates_map = {}
        manmap = []
        for core_index, matched_index in enumerate(match):
            if matched_index in flexible:
                continue

            scaffold_atom = scaffold_mol.GetAtomWithIdx(core_index)

            # ignore the R atom being matched
            if scaffold_atom.GetAtomicNum() == 0:
                continue

            # verify that the appropriate atom types were matched
            rmol_atom = rmol.GetAtomWithIdx(matched_index)
            if (scaffold_atom.GetAtomicNum() != rmol_atom.GetAtomicNum()):
                raise ValueError(f"Scaffold {core_index}:{scaffold_atom.GetSymbol()} "
                                 f"does not match {matched_index}:{rmol_atom.GetSymbol()}. "
                                 f"RDKit .GetSubstructMatch appears to have failed. Please report.  ")

            core_atom_coordinate = scaffold_conformer.GetAtomPosition(core_index)
            coordinates_map[matched_index] = core_atom_coordinate
            manmap.append((matched_index, core_index))
    else:
        try:
            from ties.topology_superimposer import superimpose_topologies, Atom, get_starting_configurations
        except ModuleNotFoundError as NoTies:
            raise WrongCoreForMolecule("Molecule doesn't match the core. "
                                       "This can be caused by the order of SMILES, for example. "
                                       "You can install the python package 'ties' to use MCS instead. ", match) from NoTies

        def to_ties_atoms(rdkit_mol):
            ties_atoms = {}
            for rdkit_atom in rdkit_mol.GetAtoms():
                ties_atom = Atom(name=rdkit_atom.GetSymbol() + str(rdkit_atom.GetIdx()), atom_type=rdkit_atom.GetSymbol())
                ties_atom.id = rdkit_atom.GetIdx()
                ties_atoms[rdkit_atom.GetIdx()] = ties_atom

            for bond in rdkit_mol.GetBonds():
                ties_atoms[bond.GetBeginAtomIdx()].bind_to(ties_atoms[bond.GetEndAtomIdx()], str(bond.GetBondType()))
            return list(ties_atoms.values())

        rmol_ties = to_ties_atoms(rmol)
        scaffold_ties = to_ties_atoms(scaffold_mol)
        mapping = superimpose_topologies(scaffold_ties, rmol_ties, ignore_coords=True, ignore_charges_completely=True)

        coordinates_map = {}
        manmap = []
        for core_index, matched_index in sorted(mapping.matched_pairs, key = lambda p: p[0].id):
            if matched_index.id in flexible:
                continue

            core_atom_coordinate = scaffold_conformer.GetAtomPosition(core_index.id)
            coordinates_map[matched_index.id] = core_atom_coordinate
            manmap.append((matched_index.id, core_index.id))
        #
        print("Used the TIES (Bieniek et al) package to get the mapping")

    # use a reproducible random seed
    randomseed = 194715

    # Generate conformers with constrained embed
    mol_for_conformer_generation = deepcopy(rmol)
    mol_nonh_indices = [a.GetIdx() for a in rmol.GetAtoms() if a.GetSymbol() != 'H']
    dup_count = 0
    for core_index in range(num_conf):
        temp_mol = ConstrainedEmbedR2(
            mol_for_conformer_generation,
            scaffold_mol,
            coordinates_map,
            manmap,
            randomseed=randomseed + core_index,
        )

        assert temp_mol.GetNumConformers() == 1
        if conformer_exists(rmol, temp_mol.GetConformer(), mol_nonh_indices, rms_limit=minimum_conf_rms):
            dup_count += 1
        else:
            rmol.AddConformer(temp_mol.GetConformer(), assignId=True)

    if dup_count:
        logger.info(f"Removed {dup_count} duplicated conformations, leaving {rmol.GetNumConformers()} in total. ")

    print(f"Generated {rmol.GetNumConformers()} conformers. ")
    return rmol


def ConstrainedEmbedR2(
    mol,
    core,
    coordinates_map,
    indices_map,
    randomseed=0,
    getForceField=UFFGetMoleculeForceField,
):
    ci = EmbedMolecule(
        mol,
        coordMap=coordinates_map,
        randomSeed=randomseed,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        enforceChirality=True,
        useSmallRingTorsions=True,
    )
    if ci < 0:
        raise ValueError("Could not embed molecule.", mol, coordinates_map)

    # rotate the embedded conformation onto the core:
    rms = AlignMol(mol, core, atomMap=indices_map)

    ff = getForceField(mol, confId=0)
    conf = core.GetConformer()
    for matched_index, core_index in indices_map:
        coord = conf.GetAtomPosition(core_index)
        coord_index = ff.AddExtraPoint(coord.x, coord.y, coord.z, fixed=True) - 1
        ff.AddDistanceConstraint(coord_index, matched_index, 0, 0, 100.0 * 100)

    ff.Initialize()
    n = 4
    more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
    while more and n:
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        n -= 1

    # realign
    rms = AlignMol(mol, core, atomMap=indices_map)
    mol.SetProp("EmbedRMS", str(rms))
    return mol
