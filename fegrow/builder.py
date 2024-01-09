import copy
import itertools
import logging
from typing import List, Optional, Union

import networkx
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol


logger = logging.getLogger(__name__)


def build_molecules_with_rdkit(
    templates: Chem.Mol,
    r_groups: Union[Chem.Mol, List[Chem.Mol], int],
    attachment_points: Optional[List[int]] = None,
    keep_components: Optional[List[int]] = None,
):
    """
    For the given core molecule and list of attachment points
     and r groups enumerate the possible molecules and
     return a list of them.

    :param template: The core scaffold molecule to attach the r groups to, or a list of them.
    :param r_group: The list of rdkit molecules which should be considered
      r groups or the RGroup Grid with highlighted molecules.
    :param attachment_points: The list of atom index in the core ligand
      that the r groups should be attached to. If it is empty, connecting points are sought out and matched.
    """

    if keep_components is None:
        keep_components = []

    # fixme - special case after changing the API: remove in the future
    # This is a temporary warning about the change in the interface.
    # This change is because there are situations where the attachment_points do not need to be passed to the function.
    if (
        isinstance(r_groups, list)
        and len(r_groups) > 0
        and isinstance(r_groups[0], int)
    ):
        print(
            'Warning: second argument is detected to be an integer. It is now "r_groups" '
            "whereas attachement_points are provided as the 3rd argument. "
        )
        raise Exception(
            "Please note that after adding the linker to FEgrow (version 1.1), "
            'the "build_molecules" function interface has changed to'
            ' "build_molecules(core_ligand, r_groups, attachment_points)". '
        )

    # ensure template and r_group are lists
    if not issubclass(templates.__class__, List):
        templates = [templates]
    if not issubclass(r_groups.__class__, List):
        r_groups = [r_groups]

    # make a deep copy of r_groups/linkers to ensure we don't modify the library
    templates = [copy.deepcopy(mol) for mol in templates]
    r_groups = [copy.deepcopy(mol) for mol in r_groups]

    # get attachment points for each template
    if not attachment_points:
        # attempt to generate the attachment points by picking the joining molecule
        # case: a list of templates previously joined with linkers requires iterating over them
        attachment_points = [
            get_attachment_vector(lig)[0].GetIdx() for lig in templates
        ]
    elif str(attachment_points).isdigit():
        attachment_points = [attachment_points]

    if not attachment_points:
        raise Exception("Could not find attachement points. ")

    if len(attachment_points) != len(templates):
        raise Exception(
            f"There must be one attachment point for each template. "
            f"Provided attachement points = {len(attachment_points)} "
            f"with templates number: {len(templates)}"
        )

    combined_mols = []
    id_counter = 0
    for atom_idx, scaffold_ligand, keep_submolecule_cue in itertools.zip_longest(
        attachment_points, templates, keep_components, fillvalue=None
    ):
        for r_mol in r_groups:
            scaffold_mol = copy.deepcopy(scaffold_ligand)
            merged_mol, scaffold_no_attachement = merge_R_group(
                scaffold=scaffold_mol,
                RGroup=r_mol,
                replace_index=atom_idx,
                keep_cue_idx=keep_submolecule_cue,
            )
            # assign the identifying index to the molecule
            merged_mol.id = id_counter
            combined_mols.append((merged_mol, scaffold_ligand, scaffold_no_attachement))
            id_counter += 1

    assert len(combined_mols) == 1

    return combined_mols[0]


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
            "The linking R atom in the R group has two or more attachment points. "
        )

    return atom, neighbours[0]


def is_linker(rmol):
    """
    Check if the molecule is a linker by checking if it has 2 R-group points
    """
    if len([atom for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in (1, 2)]) == 2:
        return True

    return False
