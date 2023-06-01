from fegrow import RList
from rdkit import Chem
import prody


# create a list of molecules from smiles
rdkit_mols = Chem.SmilesMolSupplier('sars_generated_smiles.txt')
# load the common core
core = Chem.SDMolSupplier('core.sdf')[0]

rlist = RList.from_mols(rdkit_mols)
rlist.set_common_core(core)

# generate conformers, etc
rlist[5].generate_conformers(100)
rec_final = prody.parsePDB("rec_final.pdb")
rlist[5].remove_clashing_confs(rec_final)
rlist[5].to_file('exampleIndex5.sdf')