import pandas

from rdkit import Chem
from fegrow import ChemSpace


# check if two molecules were built with chemspace
chemspace = ChemSpace()

scaffold = Chem.SDMolSupplier("data/5R83_core.sdf", removeHs=False)[0]
chemspace.add_scaffold(scaffold, 6)

oracle = pandas.read_csv("data/cs50k_scored49578_unique47710.csv.zip")

# separate the Smiles to be scanned
smiles_list = oracle.Smiles.to_list()[:40]
chemspace.add_smiles(smiles_list, h=6)

# the protein here does not matter as we don't use it anyway
chemspace.add_protein("data/7L10_final.pdb")


def oracle_look_up(scaffold, h, smiles, *args, **kwargs):
    # mol, data
    return None, {"score": oracle[oracle.Smiles == smiles].iloc[0].cnnaffinity}


# select random molecules
random_pics = chemspace.active_learning(n=5, first_random=True)
chemspace.evaluate(random_pics, full_evaluation=oracle_look_up)

# set the results for the studied smiles
for i in range(2):
    picks = chemspace.active_learning(n=5)
    res = chemspace.evaluate(picks, full_evaluation=oracle_look_up)
    # filter out the penalties
    res = res[res.score != 0]
    print(
        f"AL cycle cnnaffinity. Mean: {res.score.mean():.2f}, Min: {res.score.min():.2f}, Max: {res.score.max():.2f}"
    )
