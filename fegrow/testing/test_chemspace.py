import pathlib
import pytest
import random

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

    assert chemspace.dataframe.iloc[1].score > 2.0


def test_pipeline_smiles(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]',
                          '[H]c1nc([H])c(C2([H])C([H])([H])C2([H])[H])c([H])c1[H]'])

    chemspace.add_protein(rec_7l10_final_path)

    chemspace.evaluate([1], skip_optimisation=True)

    assert chemspace.dataframe.iloc[1].score > 2.0


def test_evaluate_scoring_function_works(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """
    Ensure that the passed functional form is used.

    :param RGroups:
    :param sars_scaffold_chunk_sdf:
    :param rec_7l10_final_path:
    :return:
    """

    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]'])
    chemspace.add_protein(rec_7l10_final_path)

    random_score = random.random()
    def scorer(rmol, pdb_filename, data):
        return random_score

    chemspace.evaluate([0], scoring_function=scorer, skip_optimisation=True)

    assert chemspace.dataframe.iloc[0].score == random_score


def test_evaluate_scoring_function_saves_data(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """
    Ensure that the passed functional form is used.

    :param RGroups:
    :param sars_scaffold_chunk_sdf:
    :param rec_7l10_final_path:
    :return:
    """

    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]'])
    chemspace.add_protein(rec_7l10_final_path)

    hello_world = "Hi Frank!"
    def scorer(rmol, pdb_filename, data):
        data["hello_world"] = hello_world
        return 5

    chemspace.evaluate([0], scoring_function=scorer, skip_optimisation=True)

    assert chemspace.dataframe.iloc[0].Mol.GetProp("hello_world") == hello_world


def test_evaluate_full_hijack(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """
    Ensure that the passed functional form is used.

    :param RGroups:
    :param sars_scaffold_chunk_sdf:
    :param rec_7l10_final_path:
    :return:
    """

    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]'])
    chemspace.add_protein(rec_7l10_final_path)

    def full_evaluation(scaffold,
                     h,
                     smiles,
                     pdb_filename,
                     *args,
                     **kwargs):
        # return: mol, data
        return None, {"score": 5}

    chemspace.evaluate([0], full_evaluation=full_evaluation)

    assert chemspace.dataframe.iloc[0].score == 5


@pytest.mark.skip(reason="requires the pydockingorg interface. ")
def test_adding_enamines(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """
    Ensure that the passed functional form is used.

    :param RGroups:
    :param sars_scaffold_chunk_sdf:
    :param rec_7l10_final_path:
    :return:
    """

    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]'], h=8)
    chemspace.add_protein(rec_7l10_final_path)
    def scorer(rmol, pdb_filename, data):
        return 5

    chemspace.evaluate([0], scoring_function=scorer, skip_optimisation=True)
    assert len(chemspace) == 1

    chemspace.add_enamine_molecules(results_per_search=10)

    # at least one extra one must have made it
    assert len(chemspace) > 1