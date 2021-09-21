import copy
from typing import Optional, List, Union
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from MDAnalysis.analysis.distances import distance_array
from prody.proteins.functions import showProtein, view3D
import py3Dmol
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdMolAlign import AlignMol
import mols2grid
import pandas as pd


from .conformers import generate_conformers
from .toxicity import tox_props


def replace_atom(mol: Chem.Mol, target_idx: int, new_atom: int) -> Chem.Mol:
    edit_mol = Chem.RWMol(mol)
    for atom in edit_mol.GetAtoms():
        if atom.GetIdx() == target_idx:
            atom.SetAtomicNum(new_atom)
    return Chem.Mol(edit_mol)


def rep2D(mol, idx=-1, rdkit_mol=False, **kwargs):
    numbered = copy.deepcopy(mol)
    numbered.RemoveAllConformers()
    if idx:
        for atom in numbered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    AllChem.Compute2DCoords(numbered)

    if rdkit_mol:
        return numbered
    else:
        return Draw.MolToImage(numbered, **kwargs)


def rep3D(mol):
    viewer = py3Dmol.view(width=300, height=300, viewergrid=(1, 1))
    for i in range(mol.GetNumConformers()):
        mb = Chem.MolToMolBlock(mol, confId=i)
        viewer.addModel(mb, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer


def __getAttachmentVector(R_group):
    """for a fragment to add, search for the position of
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
            raise Exception(
                "The linking R atom in the R group has two or more attachment points. "
                "NOT IMPLEMENTED. "
            )

        return atom, neighbours[0]

    raise Exception("No R atom in the R group. ")


def merge_R_group(mol, R_group, replaceIndex):
    """function originally copied from
    https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py"""

    # the linking R atom on the R group
    rgroup_R_atom, R_atom_neighbour = __getAttachmentVector(R_group)
    print(f"Rgroup atom index {rgroup_R_atom} neighbouring {R_atom_neighbour}")

    # atom to be replaced in the molecule
    replace_atom = mol.GetAtomWithIdx(replaceIndex)
    assert (
        len(replace_atom.GetNeighbors()) == 1
    ), "The atom being replaced on the molecule has more neighbour atoms than 1. Not supported."
    replace_atom_neighbour = replace_atom.GetNeighbors()[0]

    # align the Rgroup
    AlignMol(
        R_group,
        mol,
        atomMap=(
            (R_atom_neighbour.GetIdx(), replace_atom.GetIdx()),
            (rgroup_R_atom.GetIdx(), replace_atom_neighbour.GetIdx()),
        ),
    )

    # merge the two molecules
    combined = Chem.CombineMols(mol, R_group)
    emol = Chem.EditableMol(combined)

    # connect
    bond_order = rgroup_R_atom.GetBonds()[0].GetBondType()
    emol.AddBond(
        replace_atom_neighbour.GetIdx(),
        R_atom_neighbour.GetIdx() + mol.GetNumAtoms(),
        order=bond_order,
    )
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

    with_template = Rmol(merged)
    with_template.save_template(template)

    return with_template


class Rmol(rdkit.Chem.rdchem.Mol):
    def save_template(self, mol):
        self.template = Rmol(copy.deepcopy(mol))

    def toxicity(self):
        return tox_props(self)

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        cons = generate_conformers(self, num_conf, minimum_conf_rms, **kwargs)
        self.RemoveAllConformers()
        [self.AddConformer(con, assignId=True) for con in cons.GetConformers()]

    def rep2D(self, **kwargs):
        return rep2D(self, **kwargs)

    def rep3D(self, view=None, prody=None, template=False, confIds: Optional[List[int]] = None):
        if prody is not None:
            view = view3D(prody)

        if view is None:
            view = py3Dmol.view(width=400, height=400, viewergrid=(1, 1))

        for conf in self.GetConformers():
            # ignore the confIds that we're not asked for
            if confIds is not None and conf.GetId() not in confIds:
                continue
            mb = Chem.MolToMolBlock(self, confId=conf.GetId())
            view.addModel(mb, "lig")

            # use reverse indexing to reference the just added conformer
            # http://3dmol.csb.pitt.edu/doc/types.html#AtomSelectionSpec
            # cmap = plt.get_cmap("tab20c")
            # hex = to_hex(cmap.colors[i]).split('#')[-1]
            view.setStyle({'model': -1}, {'stick': {}})

        if template:
            mb = Chem.MolToMolBlock(self.template)
            view.addModel(mb, "template")
            # show as sticks
            view.setStyle({'model': -1}, {'stick': {'color': '0xAF10AB'}})

        # zoom to the last added model
        view.zoomTo({'model': -1})
        return view

    def removeConfsClashingWithProdyProt(self, prot, min_dst_allowed=1):
        prot_coords = prot.getCoords()

        counter = 0
        for conf in list(self.GetConformers())[::-1]:
            confid = conf.GetId()

            min_dst = np.min(distance_array(conf.GetPositions(), prot_coords))

            if min_dst < min_dst_allowed:
                self.RemoveConformer(confid)
                print(f"Clash with the protein. Removing conformer id: {confid}")

    def to_file(self, file_name: str):
        """
        Write the molecule and all conformers to file.

        Note:
            The file type is worked out from the name extension by splitting on `.`.
        """
        file_type = file_name.split(".")[-1]
        write_functions = {
            "mol": Chem.MolToMolBlock,
            "sdf": Chem.MolToMolBlock,
            "pdb": Chem.MolToPDBBlock,
            "xyz": Chem.MolToXYZBlock,
        }

        func = write_functions.get(file_type, None)
        if func is None:
            raise RuntimeError(
                f"The file type {file_type} is not support please chose from {write_functions.keys()}"
            )

        with open(file_name, "w") as output:
            for conformer in self.GetConformers():
                output.write(func(self, confId=conformer.GetId()))


class RGroupGrid(mols2grid.MolGrid):
    """
    A wrapper around the mols to grid class to load and process the r group folders locally.
    """

    def __init__(self):
        dataframe = self._load_molecules()

        super(RGroupGrid, self).__init__(dataframe, mol_col="Molecules", use_coords=False)

    def _load_molecules(self) -> pd.DataFrame:
        """
        Load the local r groups into rdkit molecules
        """
        molecules = []
        groups = []
        names = []
        molfiles = []
        root_path = "data/rgroups/molecules"
        for group in os.listdir(root_path):
            group_path = os.path.join(root_path, group)
            if os.path.isdir(group_path):
                # load all of the molecules in the folder
                for f in os.listdir(group_path):
                    molfile = os.path.join(group_path, f)
                    r_mol = Chem.MolFromMolFile(molfile, removeHs=False)
                    groups.append(group)
                    names.append(r_mol.GetProp("_Name"))
                    molfiles.append(molfile)

                    # highlight the attachment atom
                    for atom in r_mol.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            setattr(r_mol, "__sssAtoms", [atom.GetIdx()])
                    molecules.append(r_mol)

        return pd.DataFrame({"Molecules": molecules, "Functional Group": groups, "Name": names, "Mol File": molfiles})

    def _ipython_display_(self):
        from IPython.display import display
        subset = ["img", "Functional Group", "Name", "mols2grid-id"]
        return display(self.display(subset=subset))


def build_molecules(core_ligand: Rmol,
                    attachment_points: List[int],
                    r_groups: Union[RGroupGrid, List[Chem.Mol]],
                    ) ->List[Rmol]:
    """
    For the given core molecule and list of attachment points and r groups enumerate the possible molecules and return a list of them.

    Args:
        core_ligand:
            The core scaffold molecule to attach the r groups to.
        attachment_points:
            The list of atom index in the core ligand that the r groups should be attached to.
        r_groups:
            The list of rdkit molecules which should be considered r groups or the RGroup Grid with highlighted molecules.
    """
    # get a list of rdkit molecules
    if isinstance(r_groups, RGroupGrid):
        selection = mols2grid.selection
        # now get a list of the molecules
        r_mols = [r_groups.dataframe.iloc[i]["Molecules"] for i in selection.keys()]
    else:
        r_mols = r_groups

    combined_mols = []
    # loop over the attachment points and r_groups
    for atom_idx in attachment_points:
        for r_mol in r_mols:
            core_mol = Rmol(copy.deepcopy(core_ligand))
            combined_mols.append(merge_R_group(mol=core_mol, R_group=r_mol, replaceIndex=atom_idx))

    return combined_mols
