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

rmols = fegrow.build_molecules(template_mol, 
                               attachment_index, 
                               selected_rgroups)
rmols
rmols.rep2D()
rmols[0].rep3D()
rmols.toxicity()
rmols.generate_conformers(num_conf=5,
                          minimum_conf_rms=0.5, 
                          # flexible=[3, 18, 20])
                        )
sys = prody.parsePDB('7L10.pdb')
rec = sys.select('not (nucleic or hetatm or water)')
prody.writePDB('rec.pdb', rec)
fegrow.fix_receptor("rec.pdb", "rec_final.pdb")
rec_final = prody.parsePDB("rec_final.pdb")
rmols[0].rep3D(prody=rec_final)
rmols.remove_clashing_confs(rec_final)
rmols[0].rep3D(prody=rec_final)
