import pathlib
import pytest

import fegrow
from fegrow import RGroupGrid, RLinkerGrid
from rdkit import Chem


# instantiate the libraries
RGroups = RGroupGrid._load_molecules()
RLinkers = RLinkerGrid._load_molecules()

root = pathlib.Path(__file__).parent


@pytest.fixture
def sars_core_scaffold():
    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    scaffold = Chem.MolFromSmiles('[H]c1c([H])c([H])n([H])c(=O)c1[H]', params=params)
    Chem.AllChem.Compute2DCoords(scaffold)
    return scaffold


def test_adding_ethanol_1mol(sars_core_scaffold):
    # use a hydrogen bond N-H
    attachment_index = [7]
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmols = fegrow.build_molecules(sars_core_scaffold, [ethanol_rgroup], attachment_index)

    assert len(rmols) == 1, "Did not generate 1 molecule"


def test_growing_keep_larger_component(sars_core_scaffold):
    """
     When a growing vector is a random internal atom, the molecule is divided using that atom,
     and the largest component becomes the scaffold.
    """
    scaffold = Chem.MolFromSmiles('O=c1c(-c2cccc(Cl)c2)cccn1-c1cccnc1')
    Chem.AddHs(scaffold)
    Chem.AllChem.Compute2DCoords(scaffold)

    # use C on the chlorinated benzene
    attachment_index = [3]
    ethanol_rgroup = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    rmol = fegrow.build_molecules(scaffold, [ethanol_rgroup], attachment_index).pop()

    assert Chem.MolToSmiles(Chem.RemoveHs(rmol)) == 'O=c1c(CCO)cccn1-c1cccnc1'


def test_adding_ethanol_number_of_atoms():
    # Check if merging ethanol with a molecule yields the right number of atoms.
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    template_atoms_num = template_mol.GetNumAtoms()
    attachment_index = [40]

    # get a group
    ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    ethanol_atoms_num = ethanol.GetNumAtoms()

    # merge
    rmols = fegrow.build_molecules(template_mol, [ethanol], attachment_index)

    assert (template_atoms_num + ethanol_atoms_num - 2) == rmols[0].GetNumAtoms()


def test_growing_plural_groups():
    # Check if adding two groups to a templates creates two molecules.
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    attachment_index = [40]

    # get r-group
    ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    cyclopropane = RGroups[RGroups.Name == "*C1CC1"].Mol.values[0]

    rmols = fegrow.build_molecules(
        template_mol, [ethanol, cyclopropane], attachment_index
    )

    assert len(rmols) == 2


def test_added_ethanol_conformer_generation():
    # Check if conformers are generated correctly.
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    attachment_index = [40]

    # get r-group
    ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]

    rmols = fegrow.build_molecules(template_mol, [ethanol], attachment_index)

    rmols.generate_conformers(num_conf=20, minimum_conf_rms=0.1)

    assert rmols[0].GetNumConformers() > 2


def test_add_a_linker_check_star():
    """
    1. load the core
    2. load the linker
    3. add the linker to the core
    4. check if there is a danling R/* atom
    linker = R1 C R2, *1 C *2, Core-C-*1,

    :return:
    """
    # Check if conformers are generated correctly.
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    attachment_index = [40]
    # Select a linker
    linker = RLinkers[RLinkers.Name == 'R1NC(R2)=O'].Mol.values[0]
    template_with_linker = fegrow.build_molecules(
        template_mol, [linker], attachment_index
    )[0]
    for atom in template_with_linker.GetAtoms():
        if atom.GetAtomicNum() == 0:
            assert len(atom.GetBonds()) == 1


def test_two_linkers_two_rgroups():
    # Check combinatorial: ie 2 rgroups and 2 linkers create 4 molecles that contain both

    # get two R-groups
    R_group_ethanol = RGroups[RGroups.Name == '*CCO'].Mol.values[0]
    R_group_cyclopropane = RGroups[RGroups.Name == '*C1CC1'].Mol.values[0]

    # get two linkers
    linker1 = RLinkers[RLinkers.Name == 'R1CR2'].Mol.values[0]
    linker2 = RLinkers[RLinkers.Name == 'R1OR2'].Mol.values[0]

    built_molecules = fegrow.build_molecules([linker1, linker2], [R_group_ethanol, R_group_cyclopropane])

    assert len(built_molecules) == 4
