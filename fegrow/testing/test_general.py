import pathlib
import pytest

import pandas

import fegrow
from rdkit import Chem


def test_adding_ethanol_1mol(RGroups, sars_core_scaffold):
    # use a hydrogen bond N-H
    attachment_index = 7
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmol = fegrow.build_molecule(sars_core_scaffold, ethanol_rgroup, attachment_index)

    assert isinstance(rmol, Chem.Mol), "Did not generate a molecule"


def test_growing_keep_larger_component(RGroups):
    """
    When a growing vector is an internal atom that divides the molecule,
    the largest component becomes the scaffold.
    """
    scaffold = Chem.MolFromSmiles("O=c1c(-c2cccc(Cl)c2)cccn1-c1cccnc1")

    # use C on the chlorinated benzene
    attachment_index = 3
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmol = fegrow.build_molecule(scaffold, ethanol_rgroup, attachment_index)

    assert Chem.MolToSmiles(Chem.RemoveHs(rmol)) == "O=c1c(CCO)cccn1-c1cccnc1"


def test_growing_keep_larger_component_implicit_index(RGroups):
    """
    When a growing vector is an internal atom that divides the molecule,
    the largest component becomes the scaffold.
    """
    scaffold = Chem.MolFromSmiles("O=c1c(-c2cccc(Cl)c2)cccn1-c1cccnc1")
    # use C on the chlorinated benzene
    scaffold.GetAtomWithIdx(3).SetAtomicNum(0)

    methanol = Chem.MolFromSmiles("*CO")
    rmol = fegrow.build_molecule(scaffold, methanol)

    assert Chem.MolToSmiles(Chem.RemoveHs(rmol)) == "O=c1c(CO)cccn1-c1cccnc1"


def test_growing_keep_cue_component(RGroups):
    """
    When a growing vector is an atom that divides the molecule,
    the user can specify which side to keep.

    Keep the smaller chlorinated benzene ring for growing ethanol
    """
    scaffold = Chem.MolFromSmiles("O=c1c(-c2cccc(Cl)c2)cccn1-c1cccnc1")
    Chem.AllChem.Compute2DCoords(scaffold)

    # use C on the chlorinated benzene
    attachment_index = 2
    keep_smaller_ring = 3
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmol = fegrow.build_molecule(
        scaffold, ethanol_rgroup, attachment_index, keep=keep_smaller_ring
    )

    assert Chem.MolToSmiles(Chem.RemoveHs(rmol)) == "OCCc1cccc(Cl)c1"


def test_replace_methyl(RGroups, sars_core_scaffold):
    """ """
    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    mol = Chem.MolFromSmiles(
        "[H]c1nc(N([H])C(=O)C([H])([H])[H])c([H])c([H])c1[H]", params=params
    )
    Chem.AllChem.Compute2DCoords(mol)

    scaffold = fegrow.RMol(mol)

    # replace the methyl group
    attachment_index = 8
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmol = fegrow.build_molecule(scaffold, ethanol_rgroup, attachment_index)

    assert (
        Chem.MolToSmiles(rmol)
        == "[H]OC([H])([H])C([H])([H])C(=O)N([H])c1nc([H])c([H])c([H])c1[H]"
    )


def test_extend_mol_with_smiles_marked(sars_core_scaffold):
    methyl = "*OC([H])([H])[H]"
    stitched = fegrow.build_molecule(sars_core_scaffold, methyl, scaffold_point=7)


def test_extend_mol_with_smiles_explicit(sars_core_scaffold):
    methyl = "[H]OC([H])([H])[H]"
    stitched = fegrow.build_molecule(
        sars_core_scaffold, methyl, scaffold_point=7, rgroup_point=5
    )


def test_extend_mol_with_smiles_early_marking(sars_core_scaffold):
    methyl_smiles = "[H]OC([H])([H])[H]"
    params = Chem.SmilesParserParams()
    params.removeHs = False
    methyl = Chem.MolFromSmiles(methyl_smiles, params=params)
    methyl.GetAtomWithIdx(5).SetAtomicNum(0)

    stitched = fegrow.build_molecule(sars_core_scaffold, methyl, scaffold_point=7)


def test_extend_mol_with_smiles_all_marked(sars_core_scaffold):
    methyl_smiles = "[H]OC([H])([H])[H]"
    params = Chem.SmilesParserParams()
    params.removeHs = False
    methyl = Chem.MolFromSmiles(methyl_smiles, params=params)
    methyl.GetAtomWithIdx(5).SetAtomicNum(0)

    sars_core_scaffold.GetAtomWithIdx(7).SetAtomicNum(0)

    stitched = fegrow.build_molecule(sars_core_scaffold, methyl)


def test_replace_methyl_keep_h(RGroups):
    """ """
    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    mol = Chem.MolFromSmiles(
        "[H]c1nc(N([H])C(=O)C([H])([H])[H])c([H])c([H])c1[H]", params=params
    )
    Chem.AllChem.Compute2DCoords(mol)

    scaffold = fegrow.RMol(mol)

    # replace the methyl group
    attachment_index = 8
    keep_only_h = 10
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmol = fegrow.build_molecule(
        scaffold, ethanol_rgroup, attachment_index, keep=keep_only_h
    )

    assert Chem.MolToSmiles(Chem.RemoveHs(rmol)) == "CCO"


def test_adding_ethanol_number_of_atoms(RGroups, sars_scaffold_sdf):
    # Check if merging ethanol with a molecule yields the right number of atoms.
    template_atoms_num = sars_scaffold_sdf.GetNumAtoms()

    # get a group
    ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    ethanol_atoms_num = ethanol.GetNumAtoms()

    # merge
    rmol = fegrow.build_molecule(sars_scaffold_sdf, ethanol, 40)

    assert (template_atoms_num + ethanol_atoms_num - 2) == rmol.GetNumAtoms()


def test_added_ethanol_conformer_generation(RGroups, sars_scaffold_sdf):
    # Check if conformers are generated correctly.

    # get r-group
    ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]

    rmol = fegrow.build_molecule(sars_scaffold_sdf, ethanol, 40)

    rmol.generate_conformers(num_conf=20, minimum_conf_rms=0.1)

    assert rmol.GetNumConformers() > 2


def test_add_smiles_linker(sars_scaffold_chunk_sdf):
    """
    :return:
    """
    # Check if conformers are generated correctly.
    attachment_index = 7

    linker_rccr = Chem.AddHs(Chem.MolFromSmiles("*CC*"))
    scaffold_with_linker = fegrow.build_molecule(
        sars_scaffold_chunk_sdf, linker_rccr, attachment_index
    )


def test_add_linker_rgroup_first():
    """
    :return:
    """
    # we can also link the linker and the R-group first
    # before merging them into the main structure
    linker_rccr = Chem.MolFromSmiles("*CC*")
    methyl = Chem.MolFromSmiles("CO*")
    prep = fegrow.build_molecule(linker_rccr, methyl)
    assert Chem.MolToSmiles(prep) == "*CCOC"


def test_linker_c_first():
    """
    Mark the C as the first connecting point in the linker

    :return:
    """
    linker_rccr = Chem.MolFromSmiles("[*:0]CO[*:1]")

    carbon = Chem.MolFromSmiles("C*")
    cco = fegrow.build_molecule(linker_rccr, carbon)
    assert Chem.MolToSmiles(cco) == "CCO[*:1]"


def test_linker_o_first():
    """
    Mark the O as the first connecting point in the linker

    :return:
    """
    linker_rccr = Chem.MolFromSmiles("[*:1]CO[*:0]")

    carbon = Chem.MolFromSmiles("C*")
    cco = fegrow.build_molecule(linker_rccr, carbon)
    assert Chem.MolToSmiles(cco) == "COC[*:1]"


def test_add_a_linker_check_star(RLinkers, sars_scaffold_sdf):
    """
    1. load the core
    2. load the linker
    3. add the linker to the core
    4. check if there is a danling R/* atom
    linker = R1 C R2, *1 C *2, Core-C-*1,

    :return:
    """
    # Check if conformers are generated correctly.
    attachment_index = 40
    # Select a linker
    linker = RLinkers[RLinkers.Name == "R1NC(R2)=O"].Mol.values[0]
    template_with_linker = fegrow.build_molecule(
        sars_scaffold_sdf, linker, attachment_index
    )

    for atom in template_with_linker.GetAtoms():
        if atom.GetAtomicNum() == 0:
            assert len(atom.GetBonds()) == 1
