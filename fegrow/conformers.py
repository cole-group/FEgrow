import logging
from copy import deepcopy
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem


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
            match,
            manmap,
            flexible,
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


from rdkit import DataStructs
from rdkit import ForceField
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Geometry import rdGeometry
from rdkit.Chem.EnumerateStereoisomers import (
    StereoEnumerationOptions,
    EnumerateStereoisomers,
)


def ConstrainedEmbedR2(
    mol,
    core,
    coordMap,
    match,
    manmap,
    flexible,
    useTethers=True,
    coreConfId=-1,
    randomseed=2342,
    getForceField=UFFGetMoleculeForceField,
    **kwargs,
):
    ci = EmbedMolecule(
        mol,
        coordMap=coordMap,
        randomSeed=randomseed,
        **kwargs,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        enforceChirality=True,
        useSmallRingTorsions=True,
    )
    if ci < 0:
        raise ValueError("Could not embed molecule.", mol, coordMap)

    # rms = AlignMol(mol, core, atomMap=manmap)
    # mol.SetProp('EmbedRMS', str(rms))
    # return mol

    if not useTethers:
        print("not using tethers")
        # clean up the conformation
        ff = getForceField(mol, confId=0)
        for i, idxI in enumerate(match):
            if i in flexible:
                continue

            for j in range(i + 1, len(match)):
                if j in flexible:
                    continue

                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.0 * 999999)
        ff.Initialize()
        n = 4
        more = ff.Minimize()
        while more and n:
            more = ff.Minimize()
            n -= 1
        # rotate the embedded conformation onto the core:
        rms = AlignMol(mol, core, atomMap=manmap)
    else:
        # rotate the embedded conformation onto the core:
        rms = AlignMol(mol, core, atomMap=manmap)
        ff = getForceField(mol, confId=0)
        conf = core.GetConformer()
        for matchedMolI, coreI in manmap:
            # for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(coreI)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, matchedMolI, 0, 0, 100.0 * 100)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AlignMol(mol, core, atomMap=manmap)
    mol.SetProp("EmbedRMS", str(rms))
    return mol


def ConstrainedEmbedR(
    mol,
    core,
    useTethers=True,
    coreConfId=-1,
    randomseed=2342,
    getForceField=UFFGetMoleculeForceField,
    **kwargs,
):
    """generates an embedding of a molecule where part of the molecule
    is constrained to have particular coordinates

    Arguments
      - mol: the molecule to embed
      - core: the molecule to use as a source of constraints
      - useTethers: (optional) if True, the final conformation will be
          optimized subject to a series of extra forces that pull the
          matching atoms to the positions of the core atoms. Otherwise
          simple distance constraints based on the core atoms will be
          used in the optimization.
      - coreConfId: (optional) id of the core conformation to use
      - randomSeed: (optional) seed for the random number generator


    An example, start by generating a template with a 3D structure:

    >>> from rdkit.Chem import AllChem
    >>> template = AllChem.MolFromSmiles("c1nn(Cc2ccccc2)cc1")
    >>> AllChem.EmbedMolecule(template)
    0
    >>> AllChem.UFFOptimizeMolecule(template)
    0

    Here's a molecule:

    >>> mol = AllChem.MolFromSmiles("c1nn(Cc2ccccc2)cc1-c3ccccc3")

    Now do the constrained embedding
    >>> newmol=AllChem.ConstrainedEmbed(mol, template)

    Demonstrate that the positions are the same:

    >>> newp=newmol.GetConformer().GetAtomPosition(0)
    >>> molp=mol.GetConformer().GetAtomPosition(0)
    >>> list(newp-molp)==[0.0,0.0,0.0]
    True
    >>> newp=newmol.GetConformer().GetAtomPosition(1)
    >>> molp=mol.GetConformer().GetAtomPosition(1)
    >>> list(newp-molp)==[0.0,0.0,0.0]
    True

    """

    ci = EmbedMolecule(mol, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    if ci < 0:
        raise ValueError("Could not embed molecule.")

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        ff = getForceField(mol, confId=0)
        for i, idxI in enumerate(match):
            for j in range(i + 1, len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.0 * 999999)
        ff.Initialize()
        n = 4
        more = ff.Minimize()
        while more and n:
            more = ff.Minimize()
            n -= 1
        # rotate the embedded conformation onto the core:
        rms = AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        rms = AlignMol(mol, core, atomMap=algMap)
        ff = getForceField(mol, confId=0)
        conf = core.GetConformer()
        for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.0)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AlignMol(mol, core, atomMap=algMap)
    mol.SetProp("EmbedRMS", str(rms))
    return mol
