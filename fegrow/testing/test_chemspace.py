import pathlib
import pytest

import pandas

import fegrow
from fegrow import RGroups, Linkers, ChemSpace
import prody

# instantiate the libraries
RGroups = pandas.DataFrame(RGroups._load_data())
RLinkers = pandas.DataFrame(Linkers._load_data())

root = pathlib.Path(__file__).parent


def test_chem_space(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    chemspace = ChemSpace()

    R_group_ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    R_group_cyclopropane = RGroups[RGroups.Name == "*C1CC1"].Mol.values[0]

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([R_group_ethanol, R_group_cyclopropane])

    chemspace.generate_conformers(num_conf=10,
                                  minimum_conf_rms=0.5,
                                  # flexible=[3, 18, 20])
                                  )

    rec_final = prody.parsePDB(rec_7l10_final_path)
    chemspace.remove_clashing_confs(rec_final)