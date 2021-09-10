import random
from copy import deepcopy
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS


def duplicate_conformers(
    m: Chem.rdchem.Mol, new_conf_idx: int, rms_limit: float = 0.5
) -> bool:
    rmslist = []
    for conf_idx in range(m.GetNumConformers()):
        if conf_idx == new_conf_idx:
            continue
        rms = AllChem.GetConformerRMS(m, new_conf_idx, conf_idx, prealigned=True)
        rmslist.append(rms)
    # True if it is too similar to any already generated conformers
    return any(rms < rms_limit for rms in rmslist)


def generate_conformers(
    mol: Chem.rdchem.Mol,
    num_conf: int,
    minimum_conf_rms: Optional[float] = None,
) -> List[Chem.rdchem.Mol]:

    ref_mol = mol.template
    # Add Hs so that conf gen is improved
    mol = deepcopy(mol)
    mol.RemoveAllConformers()
    # mol = Chem.AddHs(mol)

    # Generate conformers with constrained embed
    dup_count = 0
    for i in range(num_conf):
        # TODO - consider tethers as a feature
        temp_mol = AllChem.ConstrainedEmbed(
            deepcopy(mol), ref_mol, randomseed=random.randint(1, 9e5)
        )
        conf_idx = mol.AddConformer(temp_mol.GetConformer(0), assignId=True)
        if minimum_conf_rms is not None:
            if duplicate_conformers(mol, conf_idx, rms_limit=minimum_conf_rms):
                dup_count += 1
                mol.RemoveConformer(conf_idx)
    if dup_count:
        print(f"removed {dup_count} duplicated conformations")
    return mol
