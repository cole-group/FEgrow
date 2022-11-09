import copy
import prody
from rdkit import Chem
import fegrow
from fegrow import RGroups

init_mol = Chem.SDMolSupplier('sarscov2/coreh.sdf', removeHs=False)[0]
template_mol = fegrow.RMol(init_mol)
template_mol.rep2D(idx=True, size=(500, 500))
attachment_index = [40]
RGroups
interactive_rgroups = RGroups.get_selected()
groups = RGroups.dataframe
R_group_ethanol = groups.loc[groups['Name']=='*CCO']['Mol'].values[0]
R_group_cyclopropane = groups.loc[groups['Name'] == '*C1CC1' ]['Mol'].values[0]
R_group_propanol = Chem.MolFromMolFile('manual_rgroups/propan-1-ol-r.mol', removeHs=False)
selected_rgroups = [R_group_ethanol] + interactive_rgroups
selected_rgroups
