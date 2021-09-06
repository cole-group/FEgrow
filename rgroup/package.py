import copy

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdMolAlign import AlignMol
import py3Dmol


def replace_atom(mol: Chem.Mol, target_idx: int, new_atom: int) -> Chem.Mol:
    edit_mol = Chem.RWMol(mol)
    for atom in edit_mol.GetAtoms():
        if atom.GetIdx() == target_idx:
            atom.SetAtomicNum(new_atom)
    return Chem.Mol(edit_mol)


def rep2D(mol, idx=True):
    numbered = copy.deepcopy(mol)
    numbered.RemoveAllConformers()
    if idx:
        for atom in numbered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    AllChem.Compute2DCoords(numbered)
    return numbered

def draw3D(mol):
    viewer = py3Dmol.view(width=300, height=300, viewergrid=(1,1))
    viewer.addModel(Chem.MolToMolBlock(mol), 'mol')
    viewer.setStyle({"stick":{}})
    viewer.zoomTo()
    return viewer


def __getAttachmentVector(R_group):
    """ for a fragment to add, search for the position of 
    the attachment point (R) and extract the atom and the connected atom 
    (currently only single bond supported)
    rgroup: fragment passed as rdkit molecule
    return: tuple (ratom, ratom_neighbour)
    """
    for atom in R_group.GetAtoms():
        if not atom.GetAtomicNum() == 0:
            continue 
        
        neighbours = atom.GetNeighbors()
        if len(neighbours) > 1:
            raise Exception("The linking R atom in the R group has two or more attachment points. "
                            "NOT IMPLEMENTED. ")
        
        return atom, neighbours[0]
    
    raise Exception('No R atom in the R group. ')


def merge_R_group(mol, R_group, replaceIndex):
    """function originally copied from
    https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py"""
    
    # the linking R atom on the R group
    rgroup_R_atom, R_atom_neighbour = __getAttachmentVector(R_group)
    print(f'Rgroup atom index {rgroup_R_atom} neighbouring {R_atom_neighbour}')
    
    # atom to be replaced in the molecule
    replace_atom = mol.GetAtomWithIdx(replaceIndex)
    assert len(replace_atom.GetNeighbors())==1, 'The atom being replaced on the molecule has more neighbour atoms than 1. Not supported.'
    replace_atom_neighbour = replace_atom.GetNeighbors()[0]
    
    # align the Rgroup
    AlignMol(R_group, mol, atomMap=(
        (R_atom_neighbour.GetIdx(),replace_atom.GetIdx()),
        (rgroup_R_atom.GetIdx(), replace_atom_neighbour.GetIdx())
                                    )
            )
    
    # merge the two molecules
    combined = Chem.CombineMols(mol, R_group)
    emol = Chem.EditableMol(combined)

    # connect
    bond_order = rgroup_R_atom.GetBonds()[0].GetBondType()
    emol.AddBond(replace_atom_neighbour.GetIdx(),
                 R_atom_neighbour.GetIdx() + mol.GetNumAtoms(),
                 order=bond_order)
    # -1 accounts for the removed linking atom on the template
    emol.RemoveAtom(rgroup_R_atom.GetIdx() + mol.GetNumAtoms())
    # remove the linking atom on the template
    emol.RemoveAtom(replace_atom.GetIdx())
    
    merged = emol.GetMol()
    Chem.SanitizeMol(merged)

    # prepare separately the template
    etemp = Chem.EditableMol(mol)
    etemp.RemoveAtom(replace_atom.GetIdx())
    template = etemp.GetMol()

    with_template = Mol(merged)
    with_template.save_template(template)

    return with_template


class Mol(rdkit.Chem.rdchem.Mol):

    def save_template(self, mol):
        self.template = mol
