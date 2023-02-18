import os
import tempfile
import requests
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import AllChem

path = "/tmp/"
os.chdir(path)


from rdkit import Chem

smiles = ["Br", "CCCC"]
CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
mols_list = []


def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text.lower()


def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon > 1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems


link = ["", "C", "CC", "CCC"]

library = []
for i in range(4):
    a = [x + link[i] for x in smiles]
    library.append(a)

flat_library = [
    item for sublist in library for item in sublist
]  # flatten list of lists


mols = []
for i in range(4):  # add all mols to a list
    for j in range(len(smiles)):
        mols.append(Chem.MolFromSmiles(library[i][j]))
# Chem.MolToMolFile(mols[50], str('test50.mol')) #working mol to molfile
print(mols)
for i, gen_mol in enumerate(mols):  # iterate over list
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mol2"
    )  # create a temp file to read into obabel
    # print(tmp.name)
    # Chem.AllChem.Compute2DCoords(gen_mol)
    filename = flat_library[i] + ".mol"
    Chem.MolToMolFile(gen_mol, filename)  # write mol to tmp file, works here
    Chem.MolFromMolFile(filename, removeHs=False)

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, filename)  # Open Babel will uncompress automatically
    mol.AddHydrogens()
    mol2 = tempfile.NamedTemporaryFile(suffix=".mol")
    obConversion.WriteFile(mol, flat_library[i] + "_h.mol")  # is a molecule with coords
    filename = flat_library[i] + "_h.mol"
    # mol = Chem.SDMolSupplier(smiles[i]+'_h.mol', removeHs=False) #is none here
    init_mol = Chem.SDMolSupplier(filename, removeHs=False)[0]
    params = AllChem.ETKDGv3()
    AllChem.EmbedMultipleConfs(init_mol, numConfs=1, params=params)
    Chem.SDWriter(filename).write(init_mol)

    # find atoms with only 1 heavy bonded atom
    atom_list = []
    for atom in init_mol.GetAtoms():
        heavy_neighbours = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
        if heavy_neighbours > 1:
            continue

        atom_list.append(atom)
    h_index = None
    # get any of the hydrogens on the heavy edge(y) atom
    for atom in atom_list:
        if len(atom_list) == 2:
            print(atom_list)
        for neighbour in atom.GetNeighbors():
            if neighbour.GetAtomicNum() == 1:
                h_index = neighbour.GetIdx()
                break
    if h_index is None:
        print("h_index is None!")
        print(filename)
        print("rings", GetRingSystems(init_mol))
        # select from the rings a carbon
        print("x")
        for idx in GetRingSystems(init_mol)[0]:
            atom = init_mol.GetAtomWithIdx(idx)
            # select any of carbons for the R group
            if atom.GetAtomicNum() == 6:
                for neighbour in atom.GetNeighbors():
                    if neighbour.GetAtomicNum() == 1:
                        print("found a h on ring")
                        h_index = neighbour.GetIdx()
                        break
                break
        else:
            print("No carbon for the attachement?!")
            continue

    print("Final h index", h_index)

    # replace the hydrogen with R
    emol = Chem.EditableMol(init_mol)
    emol.ReplaceAtom(h_index, Chem.Atom("*"))
    back = emol.GetMol()
    final_name = str(i) + "v2.mol"
    mols_list.append(back)
    try:
        Chem.SDWriter(smiles_to_iupac(flat_library[i])).write(
            back
        )  # lol @ using a webserver to go from smiles to iupac
    except:
        Chem.SDWriter(flat_library[i]).write(back)
