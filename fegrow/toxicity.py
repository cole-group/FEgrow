import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from .sascorer import calculateScore


def rule_of_five(mol):
    """
    Function to calculate the Ro5 properties for a molecule. Needs the new R group to be joined to form a single
    RDKit mol object as input. Returns a series containing the molecular weight, number of hydrogen bond donors and
    acceptors and the calculated LogP.

    A flag of True is returned if the molecule complies and False if it doesn't.
    """

    # Ro5 descriptors
    MW = round(Descriptors.MolWt(mol), 3)
    HBA = Descriptors.NumHAcceptors(mol)
    HBD = Descriptors.NumHDonors(mol)
    LogP = round(Descriptors.MolLogP(mol), 3)

    # Ro5 conditions
    conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]

    # passes Ro5 if no more than one out of four conditions is violated
    pass_ro5 = conditions.count(True) >= 3

    ro5 = {
        "MW": MW,
        "HBA": HBA,
        "HBD": HBD,
        "LogP": LogP,
        "Pass_Ro5": pass_ro5,
    }

    return ro5


def filter_mols(mol, catalog, filter_type):
    """
    Function to carry out a filter of a list of molecules dependent on the catalog supplied.
    Returns a flag of True if unwanted features are detected and False if none are found
    """

    entry = catalog.GetFirstMatch(mol)  # Get the first matching PAINS
    if entry is not None:
        flag = True  # true if mol contains filter
    else:
        flag = False  # false if not

    result = {filter_type: flag}  # return dict of values

    return result


def tox_props(data):
    """
    Function to get properties of a list of molecules and return a dataframe of results.
    """

    # initialize pains filter
    params_pains = FilterCatalogParams()
    params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog_pains = FilterCatalog(params_pains)

    # initialize unwanted substructures filter
    params_unwanted = FilterCatalogParams()
    params_unwanted.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_unwanted = FilterCatalog(params_unwanted)

    # initialise functional group filter
    params_nih = FilterCatalogParams()
    params_nih.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    catalog_nih = FilterCatalog(params_nih)

    # if a dataframe of mols is supplied
    if isinstance(data, pd.DataFrame):
        mols = list(data["ROMol"])
        ro5 = pd.DataFrame([rule_of_five(mol) for mol in mols])
        pains = pd.DataFrame(
            [filter_mols(mol, catalog_pains, "has_pains") for mol in mols]
        )
        unwanted_subs = pd.DataFrame(
            [filter_mols(mol, catalog_unwanted, "has_unwanted_subs") for mol in mols]
        )
        nih = pd.DataFrame(
            [filter_mols(mol, catalog_nih, "has_prob_fgs") for mol in mols]
        )
        sa_score = [round(calculateScore(mol), 3) for mol in mols]

        data = pd.concat(
            [data, ro5, pains, unwanted_subs, nih], axis=1
        )  # put results together
        data["synthetic_accessibility"] = sa_score

        return data

    # if a single molecule is supplied
    else:
        mol = data
        ro5 = pd.DataFrame([rule_of_five(data)])
        pains = pd.DataFrame([filter_mols(mol, catalog_pains, "has_pains")])
        unwanted_subs = pd.DataFrame(
            [filter_mols(mol, catalog_unwanted, "has_unwanted_subs")]
        )
        nih = pd.DataFrame([filter_mols(mol, catalog_nih, "has_prob_fgs")])
        sa_score = [calculateScore(mol)]

        data = pd.concat([ro5, pains, unwanted_subs, nih], axis=1)
        data["synthetic_accessibility"] = sa_score

        return data
