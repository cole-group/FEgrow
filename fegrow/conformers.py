import logging
from copy import deepcopy
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.rdMolAlign import AlignMol

logger = logging.getLogger(__name__)


class WrongCoreForMolecule(Exception):
    pass


def duplicate_conformers(
    m: Chem.rdchem.Mol, new_conf_idx: int, rms_limit: float = 0.5
) -> bool:
    for conformer in m.GetConformers():
        if conformer.GetId() == new_conf_idx:
            continue

        rms = AllChem.GetConformerRMS(m, new_conf_idx, conformer.GetId(), prealigned=True)
        if rms < rms_limit:
            return True

    return False


def generate_conformers(
    rmol: Chem.rdchem.Mol,
    num_conf: int,
    minimum_conf_rms: Optional[float] = None,
    flexible: Optional[List[int]] = [],
    use_ties_mcs: bool = False,
) -> List[Chem.rdchem.Mol]:
    """
    flexible:
            The list of atomic indices on the @core_ligand that should not be constrained during the conformer generation
    """
    scaffold_mol = deepcopy(rmol.template)
    coreConf = scaffold_mol.GetConformer(0)

    # fixme - check if the conformer has H, it helps with conformer generation
    rmol = deepcopy(rmol)

    # map scaffold atoms to the new molecules
    match = rmol.GetSubstructMatch(scaffold_mol)
    print('match', match)
    if match and not use_ties_mcs:
        # remember the scaffold coordinates
        coordMap = {}
        manmap = []
        for coreI, matchedMolI in enumerate(match):
            if matchedMolI in flexible:
                continue

            corePtI = coreConf.GetAtomPosition(coreI)
            coordMap[matchedMolI] = corePtI
            manmap.append((matchedMolI, coreI))
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

        coordMap = {}
        manmap = []
        for coreI, matchedMolI in sorted(mapping.matched_pairs, key = lambda p: p[0].id):
            if matchedMolI.id in flexible:
                continue

            corePtI = coreConf.GetAtomPosition(coreI.id)
            coordMap[matchedMolI.id] = corePtI
            manmap.append((matchedMolI.id, coreI.id))
        #
        print("Used the TIES (Bieniek et al) package to get the mapping")

    # use a reproducible random seed
    randomseed = 194715

    # Generate conformers with constrained embed
    dup_count = 0
    for coreI in range(num_conf):
        # temp_mol = AllChem.ConstrainedEmbed(deepcopy(mol), template_mol, useTethers=False, randomseed=random.randint(1, 9e5))
        temp_mol = ConstrainedEmbedR2(
            deepcopy(rmol),
            scaffold_mol,
            coordMap,
            manmap,
            randomseed=randomseed + coreI,
        )

        conf_idx = rmol.AddConformer(temp_mol.GetConformer(-1), assignId=True)
        if minimum_conf_rms is not None:
            if duplicate_conformers(rmol, conf_idx, rms_limit=minimum_conf_rms):
                dup_count += 1
                rmol.RemoveConformer(conf_idx)

    if dup_count:
        logger.debug(
            f"Removed {dup_count} duplicated conformations, leaving {rmol.GetNumConformers()} in total. "
        )

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
