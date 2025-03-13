from bs4 import BeautifulSoup
from rdkit import Chem

soup = BeautifulSoup(open("top500_R_replacements.xml").read(), "html.parser")

lib = list(Chem.SDMolSupplier("library.sdf", removeHs=False))

for mol in lib:
    sm = Chem.MolToSmiles(Chem.RemoveHs(mol))

    found = False
    for top in soup.find_all("center"):
        refsm = Chem.MolToSmiles(Chem.MolFromSmiles(top.attrs["smiles"]))
        if sm == refsm:
            found = True
            mol.SetIntProp("rank", int(top.attrs["degree"]))
            break
    if found == False:
        print("Not found", sm)
        mol.SetIntProp("rank", -1)

reordered_mols = sorted(lib, reverse=True, key=lambda m: m.GetIntProp("rank"))

with Chem.SDWriter("reordered.sdf") as SD:
    for mol in reordered_mols:
        SD.write(mol)
