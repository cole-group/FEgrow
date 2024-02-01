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
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    R_group_ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    R_group_cyclopropane = RGroups[RGroups.Name == "*C1CC1"].Mol.values[0]

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([R_group_ethanol, R_group_cyclopropane])
    assert len(chemspace) == 2

    chemspace.generate_conformers(num_conf=10,
                                  minimum_conf_rms=0.5,
                                  # flexible=[3, 18, 20])
                                  )

    rec_final = prody.parsePDB(rec_7l10_final_path)
    chemspace.remove_clashing_confs(rec_final)
    chemspace.optimise_in_receptor(rec_7l10_final_path, "openff")
    cnnaff = chemspace.gnina(rec_7l10_final_path)

    # ensure unique IDs for each molecule
    assert len({i[0] for i in cnnaff.index}) == len(chemspace)


def test_pipeline_experimental(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    R_group_ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    R_group_cyclopropane = RGroups[RGroups.Name == "*C1CC1"].Mol.values[0]

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([R_group_ethanol, R_group_cyclopropane])

    chemspace.add_protein(rec_7l10_final_path)

    chemspace._evaluate_experimental([0, 1])


def test_pipeline_rgroups(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    R_group_ethanol = RGroups[RGroups.Name == "*CCO"].Mol.values[0]
    R_group_cyclopropane = RGroups[RGroups.Name == "*C1CC1"].Mol.values[0]

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([R_group_ethanol, R_group_cyclopropane])

    chemspace.add_protein(rec_7l10_final_path)

    chemspace.evaluate([1], skip_optimisation=True)

    assert chemspace.dataframe.iloc[1].CNNAffinity > 2.0


def test_pipeline_smiles(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]',
                          '[H]c1nc([H])c(C2([H])C([H])([H])C2([H])[H])c([H])c1[H]'])

    chemspace.add_protein(rec_7l10_final_path)

    chemspace.evaluate([1], skip_optimisation=True)

    assert chemspace.dataframe.iloc[1].CNNAffinity > 2.0
