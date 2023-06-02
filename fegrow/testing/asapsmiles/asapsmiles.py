from fegrow import RList, RMol
from rdkit import Chem
import prody


# load the common core
core = Chem.SDMolSupplier('core.sdf')[0]

# create RList of molecules from smiles
rlist = RList()
params = Chem.SmilesParserParams()
params.removeHs = False # keep the hydrogens
for smiles in open('sars_generated_smiles.txt').readlines():
    mol = Chem.MolFromSmiles(smiles.strip(), params=params)
    rlist.append(RMol(mol))

# ensure that the common core is indeed that
for rmol in rlist:
    # check if the core_mol is a substructure
    if not rmol.HasSubstructMatch(core):
        raise Exception('The core molecule is not a substructure of one of the RMols in the RList, '
                        'according to Mol.HasSubstructMatch')
    rmol._save_template(core)

# conformers and energies, example for one molecule
mol_id = 1 # 5 broken, wrong number of atoms
rlist[mol_id].generate_conformers(num_conf=100, minimum_conf_rms=0.5)
protein_filename = "rec_final.pdb"
rec_final = prody.parsePDB(protein_filename)
rlist[mol_id].remove_clashing_confs(rec_final)

# continue only if there are any conformers to be optimised
if rlist[mol_id].GetNumConformers() > 0:
    energies = rlist[mol_id].optimise_in_receptor(
        receptor_file=protein_filename,
        ligand_force_field="openff",
        use_ani=True,
        sigma_scale_factor=0.8,
        relative_permittivity=4,
        water_model=None,
        platform_name='CPU',
    )
    rlist[mol_id].sort_conformers(energy_range=5)

    affinities = rlist[mol_id].gnina(receptor_file=protein_filename)

    with Chem.SDWriter(f'Rmol{mol_id}_best_conformers.sdf') as SDW:
        SDW.write(rlist[mol_id])