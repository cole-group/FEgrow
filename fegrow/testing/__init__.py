import pathlib

loc = pathlib.Path(__file__).parent

# for the tutorial, simplify the access to the scaffold and the smiles for it
core_5R83_path = str(loc / "data" / "5R83_core.sdf")
rec_5R83_path = str(loc / "data" / "5R83_rec.pdb")
smiles_5R83_path = str(loc / "data" / "cs50k_scored49578_unique47710.csv.zip")

loc = str(loc)
