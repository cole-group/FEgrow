import pathlib

from rdkit import Chem
import fegrow
from fegrow import RGroups, RLinkers

root = pathlib.Path(__file__).parent


def test_adding_ethanol_1mol():
    # Check if adding one group to a molecule creates just one molecule.
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    attachment_index = [40]

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups["Name"] == "*CCO"]["Mol"].values[0]

    # merge
    rmols = fegrow.build_molecules(template_mol, [ethanol], attachment_index)

    assert len(rmols) == 1, "Did not generate 1 molecule"


def test_adding_ethanol_number_of_atoms():
    # Check if merging ethanol with a molecule yields the right number of atoms.
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    template_atoms_num = template_mol.GetNumAtoms()
    attachment_index = [40]

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups["Name"] == "*CCO"]["Mol"].values[0]
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

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups["Name"] == "*CCO"]["Mol"].values[0]
    cyclopropane = groups.loc[groups["Name"] == "*C1CC1"]["Mol"].values[0]

    # merge
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

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups["Name"] == "*CCO"]["Mol"].values[0]

    # merge
    rmols = fegrow.build_molecules(template_mol, [ethanol], attachment_index)

    # generate conformers
    rmols.generate_conformers(num_conf=20, minimum_conf_rms=0.1)

    # there should be multiple conformers
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
    df = RLinkers.dataframe
    # Select a linker
    linker = df.loc[df["mols2grid-id"] == 842]["Mol"].values[0]
    template_with_linker = fegrow.build_molecules(
        template_mol, [linker], attachment_index
    )[0]
    for atom in template_with_linker.GetAtoms():
        if atom.GetAtomicNum() == 0:
            assert len(atom.GetBonds()) == 1


def test_two_linkers_two_rgroups():
    # Check combinatorial: ie 2 rgroups and 2 linkers create 4 molecles that contain both

    # get two R-groups
    groups = RGroups.dataframe
    R_group_ethanol = groups.loc[groups['Name'] == '*CCO']['Mol'].values[0]
    R_group_cyclopropane = groups.loc[groups['Name'] == '*C1CC1']['Mol'].values[0]

    # get two linkers
    df = RLinkers.dataframe
    linker1 = df.loc[df['Name'] == 'R1CR2']['Mol'].values[0]
    linker2 = df.loc[df['Name'] == 'R1CR2']['Mol'].values[0]

    built_molecules = fegrow.build_molecules([linker1, linker2], [R_group_ethanol, R_group_cyclopropane])

    assert len(built_molecules) == 4
