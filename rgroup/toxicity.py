from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def rule_of_five(mol):

    """
    Function to calculate the Ro5 properties for a molecule. Needs the new R group to be joined to form a single
    RDKit mol object as input. Returns a series containing the molecular weight, number of hydrogen bond donors and acceptors and the
    calculated LogP. A flag of True is returned if the molecule complies and False if it doesn't.
    """

    # Ro5 descriptors
    MW = Descriptors.ExactMolWt(mol)
    HBA = Descriptors.NumHAcceptors(mol)
    HBD = Descriptors.NumHDonors(mol)
    LogP = Descriptors.MolLogP(mol)

    # Ro5 conditions
    conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]

    # passes Ro5 if no more than one out of four conditions is violated
    if conditions.count(True) >= 3:
        pass_ro5 = True  # ro5 compliant
    else:
        pass_ro5 = False  # fails ro5

    ro5 = {
        "MW": MW,
        "HBA": HBA,
        "HBD": HBD,
        "LogP": LogP,
        "Pass_Ro5": pass_ro5,
    }  # return dict of values

    return ro5


def get_pains(mol, catalog):

    """
    Function to flag any PAINS within molecules. Returns a flag of True if PAINS are detected
    (and the structure) and False if none are found
    """

    entry = catalog.GetFirstMatch(mol)  # Get the first matching PAINS
    if entry is not None:
        flag = True  # true if mol contains PAINS
    else:
        flag = False  # false if no PAINS present

    pains = {"has_pains": flag}  # return dict of values

    return pains


def get_unwanted_subs(mol, substructures):

    """
    Function to flag any unwanted substructures. Returns a flag of True if unwanted substructures are detected
    (and the structure) and False if none are found
    """

    matches = []  # list of matching unwanted substructures
    for _, substructure in substructures.iterrows():
        if mol.HasSubstructMatch(substructure.rdkit_molecule):
            matches.append(substructure["name"])

    if len(matches) != 0:
        flag = True  # unwanted substructures present
    else:
        flag = False  # no unwanted substructures
        matches = "n/a"

    unwanted_subs = {"has_unwanted_subs": flag}  # return dict of values

    return unwanted_subs


def tox_props(data):

    """
    Function to get properties of a list of molecules and return a dataframe of results.
    """

    # initialize pains filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    # load unwanted substructure data
    substructures = pd.read_csv("data/unwanted_substructures.csv", sep="\s+")
    substructures["rdkit_molecule"] = substructures.smarts.apply(Chem.MolFromSmarts)

    # if a dataframe of mols is supplied
    if isinstance(data, pd.DataFrame):
        mols = list(data["ROMol"])
        ro5 = pd.DataFrame([rule_of_five(mol) for mol in mols])
        pains = pd.DataFrame([get_pains(mol, catalog) for mol in mols])
        unwanted_subs = pd.DataFrame(
            [get_unwanted_subs(mol, substructures) for mol in mols]
        )

        data = pd.concat(
            [data, ro5, pains, unwanted_subs], axis=1
        )  # put results together

        return data

    # if a single molecule is supplied
    else:
        ro5 = pd.DataFrame([rule_of_five(data)])
        pains = pd.DataFrame([get_pains(data, catalog)])
        unwanted_subs = pd.DataFrame([get_unwanted_subs(data, substructures)])

        data = pd.concat([ro5, pains, unwanted_subs], axis=1)

        return data
