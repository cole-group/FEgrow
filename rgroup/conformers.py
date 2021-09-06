from copy import deepcopy
from typing import Optional, List
import random

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS


def duplicate_conformers(m: Chem.rdchem.Mol, new_conf_idx: int, rms_limit: float = 0.5) -> bool:
    rmslist = []
    for i in range(m.GetNumConformers()):
        if i == new_conf_idx:
            continue
        rms = AllChem.GetConformerRMS(m, new_conf_idx, i, prealigned=True)
        rmslist.append(rms)
    return any(i < rms_limit for i in rmslist)


def generate_conformers(mol: Chem.rdchem.Mol,
                        ref_mol: Chem.rdchem.Mol,
                        num_conf: int,
                        minimum_conf_rms: Optional[float] = None,
                       ) -> List[Chem.rdchem.Mol]:
	
    # Add Hs so that conf gen is improved
    mol = deepcopy(mol)
    mol.RemoveAllConformers()
    # mol = Chem.AddHs(mol)

    # Generate conformers with constrained embed
    dup_count = 0
    for i in range(num_conf):
    	# TODO - consider tethers as a feature
        temp_mol = AllChem.ConstrainedEmbed(mol, ref_mol, randomseed=random.randint(1, 9e5))
        print(list(temp_mol.GetConformers()))
        conf_idx = mol.AddConformer(temp_mol.GetConformer(0), assignId=True)
        if minimum_conf_rms is not None and False:
            if duplicate_conformers(mol, conf_idx, rms_limit=minimum_conf_rms):
                dup_count += 1
                mol.RemoveConformer(conf_idx)
    if dup_count:
        print(f'removed {dup_count} duplicated conformations')
    return mol