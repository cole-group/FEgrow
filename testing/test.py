
from rdkit import Chem
import fegrow
from fegrow import RGroups


def test_adding_ethanol_1mol():
	"""
	Check if adding one group to a molecule creates just one molecule.
	"""
	# load the SDF
	template_mol = Chem.SDMolSupplier('data/sarscov2_coreh.sdf', removeHs=False)[0]
	attachment_index = [40]

	# get a group
	groups = RGroups.dataframe
	R_group_ethanol = groups.loc[groups['Name']=='ethanol']['Mol'].values[0]

	# merge
	rmols = fegrow.build_molecules(template_mol, 
                               attachment_index, 
                               [R_group_ethanol])

	assert len(rmols) == 1, 'Did not generate 1 molecule'

def test_adding_ethanol_number_of_atoms():
	"""
	Check if merging ethanol with a molecule yields the right number of atoms. 
	"""
	# load the SDF
	template_mol = Chem.SDMolSupplier('data/sarscov2_coreh.sdf', removeHs=False)[0]
	template_atoms_num = template_mol.GetNumAtoms()
	attachment_index = [40]

	# get a group
	groups = RGroups.dataframe
	R_group_ethanol = groups.loc[groups['Name']=='ethanol']['Mol'].values[0]
	ethanol_atoms_num = R_group_ethanol.GetNumAtoms()

	# merge
	rmols = fegrow.build_molecules(template_mol, 
                               attachment_index, 
                               [R_group_ethanol])

	assert (template_atoms_num + ethanol_atoms_num - 2) == rmols[0].GetNumAtoms()

def test_growing_plural_groups():
	"""
	Check if adding two groups to a templates creates two molecules.
	"""
	# load the SDF
	template_mol = Chem.SDMolSupplier('data/sarscov2_coreh.sdf', removeHs=False)[0]
	attachment_index = [40]

	# get a group
	groups = RGroups.dataframe
	R_group_ethanol = groups.loc[groups['Name']=='ethanol']['Mol'].values[0]
	R_group_cyclopropane = groups.loc[groups['Name'] == 'cyclopropane' ]['Mol'].values[0]

	# merge
	rmols = fegrow.build_molecules(template_mol, 
                               attachment_index, 
                               [R_group_ethanol, R_group_cyclopropane])

	assert len(rmols) == 2

