import pathlib

import pandas
import pytest
from rdkit import Chem

from fegrow import RGroups, Linkers


# instantiate the libraries

rgroups = pandas.DataFrame(RGroups._load_data())
rlinkers = pandas.DataFrame(Linkers._load_data())

root = pathlib.Path(__file__).parent


@pytest.fixture
def RGroups():
    return rgroups


@pytest.fixture
def RLinkers():
    return rlinkers


@pytest.fixture
def sars_core_scaffold():
    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    scaffold = Chem.MolFromSmiles("[H]c1c([H])c([H])n([H])c(=O)c1[H]", params=params)
    Chem.AllChem.Compute2DCoords(scaffold)
    return scaffold


@pytest.fixture
def sars_scaffold_sdf():
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarscov2_coreh.sdf"), removeHs=False
    )[0]
    return template_mol


@pytest.fixture
def sars_scaffold_chunk_sdf():
    template_mol = Chem.SDMolSupplier(
        str(root / "data" / "sarcov2_coreh_cut.sdf"), removeHs=False
    )[0]
    return template_mol


@pytest.fixture
def rec_7l10_final_path():
    return str(root / "data" / "7L10_final.pdb")
