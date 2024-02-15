import pathlib
import tempfile

import pytest
import random

import pandas

from rdkit import Chem
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

    R_group_ethanol = RGroups[RGroups.Name == "*CCO"].Mol.item()
    R_group_cyclopropane = RGroups[RGroups.Name == "*C1CC1"].Mol.item()

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

    assert chemspace.df.iloc[1].score > 2.0


def test_access_mol_directly(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    R_group_ethanol = RGroups[RGroups.Name == "*CCO"].Mol.item()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([R_group_ethanol])

    mol = chemspace[0]
    assert chemspace.df.loc[0].Mol == mol


def test_toxicity(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([RGroups[RGroups.Name == "*CCO"].Mol.item(),
                           RGroups[RGroups.Name == "*C1CC1"].Mol.item()])

    toxicity = chemspace.toxicity()
    assert len(toxicity) == 2

def test_writing(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_rgroups([RGroups[RGroups.Name == "*CCO"].Mol.item(),
                           RGroups[RGroups.Name == "*C1CC1"].Mol.item()])

    with tempfile.NamedTemporaryFile(suffix=".sdf") as TMP:
        chemspace.to_sdf(TMP.name)
        reimported_cs = ChemSpace.from_sdf(TMP.name)
        assert chemspace.df.Smiles == reimported_cs.df.Smiles


def test_pipeline_smiles(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    chemspace.add_scaffold(sars_scaffold_chunk_sdf, 8)
    # this could be a list of smiles, (but molecules would be automatically converted to smiles anyway)
    chemspace.add_smiles(['[H]OC([H])([H])C([H])([H])c1c([H])nc([H])c([H])c1[H]',
                          '[H]c1nc([H])c(C2([H])C([H])([H])C2([H])[H])c([H])c1[H]'])

    chemspace.add_protein(rec_7l10_final_path)

    chemspace.evaluate([1], skip_optimisation=True)

    assert chemspace.df.iloc[1].score > 2.0


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

    assert chemspace.df.iloc[0].score == random_score


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

    assert chemspace.df.iloc[0].Mol.GetProp("hello_world") == hello_world


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

    assert chemspace.df.iloc[0].score == 5


def test_al(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
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

    not_studied_smiles = ['[H]OC(=O)N([H])c1c([H])nc([H])c([H])c1[H]',
                          '[H]ON([H])c1c([H])nc([H])c([H])c1[H]']
    studied_smiles = ['[H]OC([H])([H])c1c([H])nc([H])c([H])c1[H]',
                      '[H]ON([H])C(=O)c1c([H])nc([H])c([H])c1[H]']
    chemspace.add_smiles(studied_smiles + not_studied_smiles)
    chemspace.add_protein(rec_7l10_final_path)

    # set the results for the studied smiles
    df = chemspace.df
    df.loc[df.index == 0, ['score', 'Training']] = [3.2475, True]
    df.loc[df.index == 1, ['score', 'Training']] = [3.57196, True]

    to_study = chemspace.active_learning(n=1)

    assert to_study.iloc[0].Smiles in not_studied_smiles


def test_al_local(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """
    Run a small active learning test.
    """

    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    scaffold = Chem.SDMolSupplier(str(root / "data/5R83_core.sdf"), removeHs=False)[0]
    chemspace.add_scaffold(scaffold, 6)

    oracle = pandas.read_csv(root / "data/cs50k_scored49578_unique47710.csv.zip")

    # separate the Smiles to be scanned
    smiles_list = oracle.Smiles.to_list()[:40]
    chemspace.add_smiles(smiles_list, h=6)

    # the protein here does not matter as we don't use it anyway
    chemspace.add_protein(rec_7l10_final_path)

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
        assert len(res) == 5

        # filter out the penalties
        res = res[res.score != 0]
        print(f"AL cycle cnnaffinity. Mean: {res.score.mean():.2f}, Min: {res.score.min():.2f}, Max: {res.score.max():.2f}")


def test_al_full(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """
    Run a small active learning test.
    """

    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    scaffold = Chem.SDMolSupplier(str(root / "data/5R83_core.sdf"), removeHs=False)[0]
    chemspace.add_scaffold(scaffold, 6)

    oracle = pandas.read_csv(root / "data/cs50k_scored49578_unique47710.csv.zip")

    # separate the Smiles to be scanned
    chemspace.add_smiles(oracle.Smiles.to_list()[:10], h=6)

    # the protein here does not matter as we don't use it anyway
    chemspace.add_protein(rec_7l10_final_path)

    def oracle_look_up(scaffold, h, smiles, *args, **kwargs):
        # mol, data
        return None, {"score": oracle[oracle.Smiles == smiles].iloc[0].cnnaffinity}

    # select random molecules
    random_pics = chemspace.active_learning(n=3, first_random=True)
    chemspace.evaluate(random_pics, full_evaluation=oracle_look_up)

    assert chemspace.df.score.count() == 3
    assert all(~chemspace.df.loc[random_pics.index].score.isna())

    # compute all
    chemspace.evaluate(full_evaluation=oracle_look_up)
    assert chemspace.df.score[chemspace.df.score.isna()].count() == 0


def test_al_manual_gp(RGroups, sars_scaffold_chunk_sdf, rec_7l10_final_path):
    """

    """
    # check if two molecules were built with chemspace
    chemspace = ChemSpace()

    scaffold = Chem.SDMolSupplier(str(root / "data/5R83_core.sdf"), removeHs=False)[0]
    chemspace.add_scaffold(scaffold, 6)

    oracle = pandas.read_csv(root / "data/cs50k_scored49578_unique47710.csv.zip")

    # separate the Smiles to be scanned
    smiles_list = oracle.Smiles.to_list()[:50]
    chemspace.add_smiles(smiles_list, h=6)

    # the protein here does not matter as we don't use it anyway
    chemspace.add_protein(rec_7l10_final_path)

    def oracle_look_up(scaffold, h, smiles, *args, **kwargs):
        # mol, data
        return None, {"score": oracle[oracle.Smiles == smiles].iloc[0].cnnaffinity}

    # select random molecules
    random_pics = chemspace.active_learning(n=5, first_random=True)
    chemspace.evaluate(random_pics, full_evaluation=oracle_look_up)

    # configure active learning
    from fegrow.al import Model, Query
    chemspace.model = Model.gaussian_process()

    chemspace.query = Query.UCB(beta=10)
    picks = chemspace.active_learning(n=5)
    evaluated = chemspace.evaluate(picks, full_evaluation=oracle_look_up)

    # another go without changing any settings
    picks = chemspace.active_learning(n=5)
    evaluated = chemspace.evaluate(picks, full_evaluation=oracle_look_up)

    # use every querrying strategy
    chemspace.query = Query.greedy()
    picks = chemspace.active_learning(n=5)
    evaluated = chemspace.evaluate(picks, full_evaluation=oracle_look_up)

    chemspace.query = Query.EI(tradeoff=0.1)
    picks = chemspace.active_learning(n=5)
    evaluated = chemspace.evaluate(picks, full_evaluation=oracle_look_up)

    chemspace.query = Query.PI(tradeoff=0.1)
    picks = chemspace.active_learning(n=5)
    evaluated = chemspace.evaluate(picks, full_evaluation=oracle_look_up)

    chemspace.model = Model.linear()
    chemspace.query = Query.greedy()
    picks = chemspace.active_learning(n=5)
    evaluated = chemspace.evaluate(picks, full_evaluation=oracle_look_up)


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