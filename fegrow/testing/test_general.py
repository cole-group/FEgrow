import pathlib

from rdkit import Chem
import fegrow
from fegrow import RGroups

root = pathlib.Path(__file__).parent


def test_adding_ethanol_1mol():
    # Check if adding one group to a molecule creates just one molecule.
    template_mol = Chem.SDMolSupplier(str(root / 'data' / 'sarscov2_coreh.sdf'), removeHs=False)[0]
    attachment_index = [40]

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups['Name'] == '*CCO']['Mol'].values[0]

    # merge
    rmols = fegrow.build_molecules(template_mol,
                                   attachment_index,
                                   [ethanol])

    assert len(rmols) == 1, 'Did not generate 1 molecule'


def test_adding_ethanol_number_of_atoms():
    # Check if merging ethanol with a molecule yields the right number of atoms.
    template_mol = Chem.SDMolSupplier(str(root / 'data' / 'sarscov2_coreh.sdf'), removeHs=False)[0]
    template_atoms_num = template_mol.GetNumAtoms()
    attachment_index = [40]

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups['Name'] == '*CCO']['Mol'].values[0]
    ethanol_atoms_num = ethanol.GetNumAtoms()

    # merge
    rmols = fegrow.build_molecules(template_mol,
                                   attachment_index,
                                   [ethanol])

    assert (template_atoms_num + ethanol_atoms_num - 2) == rmols[0].GetNumAtoms()


def test_growing_plural_groups():
    # Check if adding two groups to a templates creates two molecules.
    template_mol = Chem.SDMolSupplier(str(root / 'data' / 'sarscov2_coreh.sdf'), removeHs=False)[0]
    attachment_index = [40]

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups['Name'] == '*CCO']['Mol'].values[0]
    cyclopropane = groups.loc[groups['Name'] == '*C1CC1']['Mol'].values[0]

    # merge
    rmols = fegrow.build_molecules(template_mol,
                                   attachment_index,
                                   [ethanol, cyclopropane])

    assert len(rmols) == 2


def test_added_ethanol_conformer_generation():
    # Check if conformers are generated correctly.
    template_mol = Chem.SDMolSupplier(str(root / 'data' / 'sarscov2_coreh.sdf'), removeHs=False)[0]
    attachment_index = [40]

    # get a group
    groups = RGroups.dataframe
    ethanol = groups.loc[groups['Name'] == '*CCO']['Mol'].values[0]

    # merge
    rmols = fegrow.build_molecules(template_mol,
                                   attachment_index,
                                   [ethanol])

    # generate conformers
    rmols.generate_conformers(num_conf=20, minimum_conf_rms=0.1)

    # there should be multiple conformers
    assert rmols[0].GetNumConformers() > 2
