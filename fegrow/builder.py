import copy
import itertools
import logging
from typing import List, Optional, Union

import networkx
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol


logger = logging.getLogger(__name__)


def build_molecules_with_rdkit(
    scaffold: Chem.Mol,
    r_group: Chem.Mol,
    attachment_point: Optional[int] = None,
    keep_components: Optional[int] = None,
):
    """
    For the given core molecule and list of attachment points
     and r groups enumerate the possible molecules and
     return a list of them.

    :param scaffold: The core scaffold molecule to attach the r groups to, or a list of them.
    :param r_group: The list of rdkit molecules which should be considered
      r groups or the RGroup Grid with highlighted molecules.
    :param attachment_point: The list of atom index in the core ligand
      that the r groups should be attached to. If it is empty, connecting points are sought out and matched.
    """

    if keep_components is None:
        keep_components = []

    # make a deep copy of r_groups/linkers to ensure we don't modify the library
    scaffold = copy.deepcopy(scaffold)
    r_group = copy.deepcopy(r_group)

    # get attachment points for each template
    if not attachment_point:
        # attempt to generate the attachment points by picking the joining molecule
        # case: a list of templates previously joined with linkers requires iterating over them
        attachment_point = get_attachment_vector(scaffold)[0].GetIdx()

    if not attachment_point:
        raise Exception("Could not find attachement points. Either the atom index has to be specified,"
                        "or an atom needs to be marked rdkit.atom.SetAtomicNum(0). ")

    # for atom_idx, scaffold_ligand, keep_submolecule_cue in itertools.zip_longest(
    #     attachment_points, template, keep_components, fillvalue=None
    # ):
    #     for r_mol in r_groups:
    merged_mol, scaffold_no_attachement = merge_R_group(
        scaffold=scaffold,
        RGroup=r_group,
        replace_index=attachment_point,
        keep_cue_idx=keep_components,
    )

    merged_mol.SetProp("attachement_point", str(attachment_point))

    return merged_mol, scaffold, scaffold_no_attachement


def split(molecule, splitting_atom, keep_neighbour_idx=None):
    """
    Return the smaller part of the molecule or one that that contains the prespecified atom.

    :param molecule: RDKit Molecule
    :param splitting_atom: RDKit Atom, the growing vector used to divide the molecule into submolecules.
    :param splitting_atom: The index of the neighbouring atom on the side of the molecule that should be kept
        as the scaffold.
    :return:
    """
    G = networkx.from_numpy_array(Chem.GetAdjacencyMatrix(molecule, useBO=False))
    G.remove_node(splitting_atom.GetIdx())

    connected_components = list(networkx.connected_components(G))
    if len(connected_components) == 1:
        raise ValueError(
            f"The molecule is not divided into two separate components "
            f"with the Atom ID={splitting_atom.GetIdx()}, so we cannot decide which component to remove. "
        )

    if keep_neighbour_idx:
        # select the user specifid component
        component_to_keep = [c for c in connected_components if keep_neighbour_idx in c][0]
    else:
        # keep the largest component
        largest_component_size = max(map(len, connected_components))
        component_to_keep = [c for c in connected_components if len(c) == largest_component_size][0]

    atom_ids_for_removal = {item for sublist in connected_components for item in sublist} - component_to_keep

    # remove the unwanted component
    edit_scaffold = Chem.EditableMol(molecule)
    for idx in sorted(list(atom_ids_for_removal), reverse=True):
        edit_scaffold.RemoveAtom(idx)
    scaffold = edit_scaffold.GetMol()

    kept_atoms = [
        a for a in molecule.GetAtoms() if a.GetIdx() not in atom_ids_for_removal
    ]
    scaffold_elements = [a for a in scaffold.GetAtoms()]

    # removing atoms changes the IDs of the atoms that remain
    if [a.GetAtomicNum() for a in kept_atoms] != [
        a.GetAtomicNum() for a in scaffold_elements
    ]:
        raise Exception(
            "The assumption that the modified molecule will keep the atoms in the same order is false. "
            "Please get in touch with the FEgrow maintainers. "
        )
    idx_map = dict(
        zip([a.GetIdx() for a in kept_atoms], [a.GetIdx() for a in scaffold_elements])
    )
    return scaffold, idx_map


def merge_R_group(scaffold, RGroup, replace_index, keep_cue_idx=None):
    """function originally copied from
    https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py
    """

    # the linking R atom on the R group
    # fixme: attempt to do the same on the template if replace index is not provided
    rgroup_R_atom, R_atom_neighbour = get_attachment_vector(RGroup)

    # atom to be replaced in the scaffold
    atom_to_replace = scaffold.GetAtomWithIdx(replace_index)
    if len(atom_to_replace.GetNeighbors()) == 1:
        hook = atom_to_replace.GetNeighbors()[0]
    elif len(atom_to_replace.GetNeighbors()) != 1:
        scaffold, idx_map = split(scaffold, atom_to_replace, keep_cue_idx)
        replace_index = idx_map[replace_index]
        atom_to_replace = scaffold.GetAtomWithIdx(replace_index)
        hook = atom_to_replace.GetNeighbors()[0]

    if RGroup.GetNumConformers() == 0:
        logger.warning("The R-Group lacks initial coordinates. Defaulting to Chem.rdDistGeom.EmbedMolecule.")
        Chem.rdDistGeom.EmbedMolecule(RGroup)

    # align the Rgroup
    AlignMol(
        RGroup,
        scaffold,
        atomMap=(
            (R_atom_neighbour.GetIdx(), atom_to_replace.GetIdx()),
            (rgroup_R_atom.GetIdx(), hook.GetIdx()),
        ),
    )

    # merge
    combined = Chem.CombineMols(scaffold, RGroup)
    emol = Chem.EditableMol(combined)

    # connect
    bond_order = rgroup_R_atom.GetBonds()[0].GetBondType()
    emol.AddBond(
        hook.GetIdx(),
        R_atom_neighbour.GetIdx() + scaffold.GetNumAtoms(),
        order=bond_order,
    )
    # -1 accounts for the removed linking atom on the template
    emol.RemoveAtom(rgroup_R_atom.GetIdx() + scaffold.GetNumAtoms())
    # remove the linking atom on the template
    emol.RemoveAtom(atom_to_replace.GetIdx())

    merged = emol.GetMol()
    Chem.SanitizeMol(merged)

    # use only the best/first conformer
    for c in list(merged.GetConformers())[1:]:
        merged.RemoveConformer(c.GetId())

    # bookkeeping about scaffolding
    edit_scaffold = Chem.EditableMol(scaffold)
    edit_scaffold.RemoveAtom(atom_to_replace.GetIdx())
    scaffold_no_attachement = edit_scaffold.GetMol()

    if is_linker(RGroup):
        # update the linker so that there is an attachment point left for the future
        # atom in the linker with a label=1 was used for the merging
        # rename label=2 to 0 to turn it into a simple R-group
        for atom in merged.GetAtoms():
            if atom.GetAtomMapNum() == 2:
                atom.SetAtomMapNum(0)

    return merged, scaffold_no_attachement


def get_attachment_vector(R_group):
    """In the R-group or a linker, search for the position of the attachment point (R atom)
    and extract the atom (currently only single bond supported). In case of the linker,
    the R1 atom is selected.
    rgroup: fragment passed as rdkit molecule
    return: tuple (ratom, ratom_neighbour)
    """

    # find the R groups in the molecule
    ratoms = [atom for atom in R_group.GetAtoms() if atom.GetAtomicNum() == 0]
    if not len(ratoms):
        raise Exception(
            "The R-group does not have R-atoms (Atoms with index == 0, visualised with a '*' character)"
        )

    # if it is a linker, it will have more than 1 R group, pick the one with index 1
    if len(ratoms) == 1:
        atom = ratoms[0]
    elif is_linker(R_group):
        # find the attachable point
        ratoms = [atom for atom in ratoms if atom.GetAtomMapNum() == 1]
        atom = ratoms[0]
    else:
        raise Exception(
            "Either missing R-atoms, or more than two R-atoms. "
            '"Atom.GetAtomicNum" should be 0 for the R-atoms, and in the case of the linker,  '
            '"Atom.GetAtomMapNum" has to specify the order (1,2) '
        )

    neighbours = atom.GetNeighbors()
    if len(neighbours) > 1:
        raise NotImplementedError(
            "The linking R atom (*) has two or more attachment points (meaning bonds). "
            "Please make sure that only one bond is present for the linking 'atom'. "
        )

    return atom, neighbours[0]


def is_linker(rmol):
    """
    Check if the molecule is a linker by checking if it has 2 R-group points
    """
    if len([atom for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in (1, 2)]) == 2:
        return True

    return False
