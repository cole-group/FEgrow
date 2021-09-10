from copy import deepcopy
from typing import Optional, List
import random

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS


def duplicate_conformers(m: Chem.rdchem.Mol, new_conf_idx: int, rms_limit: float = 0.5) -> bool:
    rmslist = []
    for conf_idx in range(m.GetNumConformers()):
        if conf_idx == new_conf_idx:
            continue
        rms = AllChem.GetConformerRMS(m, new_conf_idx, conf_idx, prealigned=True)
        rmslist.append(rms)
    # True if it is too similar to any already generated conformers
    return any(rms < rms_limit for rms in rmslist)


def generate_conformers(mol: Chem.rdchem.Mol,
                        num_conf: int,
                        minimum_conf_rms: Optional[float] = None,
                       ) -> List[Chem.rdchem.Mol]:
	
    ref_mol =deepcopy(mol.template)
    # Add Hs so that conf gen is improved
    mol = deepcopy(mol)
    # mol.RemoveAllConformers()
    # mol = Chem.AddHs(mol)

    core = ref_mol
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(0)
    # for i, idxI in enumerate(match):
    #   # print((i, idxI), end=', ')
    #   corePtI = coreConf.GetAtomPosition(i)
    #   coordMap[idxI] = corePtI
    # for i in range(48 + 1):
    #   corePtI = coreConf.GetAtomPosition(i)
    #   coordMap[i] = corePtI
    # for i, idxI in [(0, 0), (1, 1), (2, 3), (3, 2), (4, 6), (5, 5), (6, 4), (7, 7), (8, 8), (9, 9), 
    # (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), 
    # (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31), 
    # (32, 32), (33, 34), (34, 33), (35, 36), (36, 35), (37, 37), (38, 38), (39, 39), (40, 40), (41, 41), (42, 42), 
    # (43, 43), (44, 44), (45, 45), (46, 46), (47, 47), (48, 48)]:
    # 16S , 18 N, 17 19 O, 39 40 H
    manmap = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), 
    (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), 
    # (16, 16), # S
    #(17, 17), # O
    (18, 18), # N
    # (19, 19), # O
    (20, 20), 
    (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31), 
    (32, 32), (33, 33), (34, 34), (35, 35), (36, 36), (37, 37), (38, 38), 
    #(39, 39), (40, 40), 
    (41, 41), (42, 42), 
    (43, 43), (44, 44), (45, 45), (46, 46), (47, 47), (48, 48)]

    for i, idxI in manmap:
        corePtI = coreConf.GetAtomPosition(i)
        # print((i, idxI), corePtI)
        coordMap[idxI] = corePtI

    # Generate conformers with constrained embed
    dup_count = 0
    for i in range(num_conf):
    	# TODO - consider tethers as a feature
        #temp_mol = AllChem.ConstrainedEmbed(deepcopy(mol), ref_mol, useTethers=False, randomseed=random.randint(1, 9e5))
        
        temp_mol = ConstrainedEmbedR2(deepcopy(mol), ref_mol, coordMap, match, manmap, useTethers=True, randomseed=random.randint(1, 9e5))
        conf_idx = mol.AddConformer(temp_mol.GetConformer(-1), assignId=True)
        if minimum_conf_rms is not None:
            if duplicate_conformers(mol, conf_idx, rms_limit=minimum_conf_rms):
                dup_count += 1
                mol.RemoveConformer(conf_idx)
    if dup_count:
        print(f'removed {dup_count} duplicated conformations')
    return mol

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
from rdkit.RDLogger import logger
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions, EnumerateStereoisomers
def ConstrainedEmbedR2(mol, core, coordMap, match, manmap, useTethers=True, coreConfId=-1, randomseed=2342,
                     getForceField=UFFGetMoleculeForceField, **kwargs):
  """ generates an embedding of a molecule where part of the molecule
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
    raise ValueError('Could not embed molecule.')

  # algMap = [(j, i) for i, j in enumerate(match)]
  algMap = manmap

  if not useTethers:
    # clean up the conformation
    ff = getForceField(mol, confId=0)
    for i, idxI in enumerate(match):
      for j in range(i + 1, len(match)):
        idxJ = match[j]
        d = coordMap[idxI].Distance(coordMap[idxJ])
        ff.AddDistanceConstraint(idxI, idxJ, d, d, 100. * 999999)
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
      ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
    ff.Initialize()
    n = 4
    more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
    while more and n:
      more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
      n -= 1
    # realign
    rms = AlignMol(mol, core, atomMap=algMap)
  mol.SetProp('EmbedRMS', str(rms))
  return mol

def ConstrainedEmbedR(mol, core, useTethers=True, coreConfId=-1, randomseed=2342,
                     getForceField=UFFGetMoleculeForceField, **kwargs):
  """ generates an embedding of a molecule where part of the molecule
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
    raise ValueError('Could not embed molecule.')

  algMap = [(j, i) for i, j in enumerate(match)]

  if not useTethers:
    # clean up the conformation
    ff = getForceField(mol, confId=0)
    for i, idxI in enumerate(match):
      for j in range(i + 1, len(match)):
        idxJ = match[j]
        d = coordMap[idxI].Distance(coordMap[idxJ])
        ff.AddDistanceConstraint(idxI, idxJ, d, d, 100. * 999999)
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
      ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
    ff.Initialize()
    n = 4
    more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
    while more and n:
      more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
      n -= 1
    # realign
    rms = AlignMol(mol, core, atomMap=algMap)
  mol.SetProp('EmbedRMS', str(rms))
  return mol