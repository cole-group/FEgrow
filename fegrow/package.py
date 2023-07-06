import copy
import itertools
import stat
from typing import Optional, List, Union, Tuple
import os
import glob
import tempfile
import subprocess
import re
from pathlib import Path
from urllib.request import urlretrieve
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from prody.proteins.functions import showProtein, view3D
import py3Dmol
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem import PandasTools
import mols2grid
import pandas

from .conformers import generate_conformers
from .toxicity import tox_props

# default options
pandas.set_option('display.precision', 3)


def replace_atom(mol: Chem.Mol, target_idx: int, new_atom: int) -> Chem.Mol:
    edit_mol = Chem.RWMol(mol)
    for atom in edit_mol.GetAtoms():
        if atom.GetIdx() == target_idx:
            atom.SetAtomicNum(new_atom)
    return Chem.Mol(edit_mol)


def rep2D(mol, idx=-1, rdkit_mol=False, **kwargs):
    numbered = copy.deepcopy(mol)
    numbered.RemoveAllConformers()
    if idx:
        for atom in numbered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    AllChem.Compute2DCoords(numbered)

    if rdkit_mol:
        return numbered
    else:
        return Draw.MolToImage(numbered, **kwargs)


def rep3D(mol):
    viewer = py3Dmol.view(width=300, height=300, viewergrid=(1, 1))
    for i in range(mol.GetNumConformers()):
        mb = Chem.MolToMolBlock(mol, confId=i)
        viewer.addModel(mb, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer


def is_linker(rmol):
    """
    Check if the molecule is a linker by checking if it has 2 R-group points
    """
    if len([atom for atom in rmol.GetAtoms() if atom.GetAtomMapNum() in (1, 2)]) == 2:
        return True

    return False


def __getAttachmentVector(R_group):
    """In the R-group or a linker, search for the position of the attachment point (R atom)
    and extract the atom (currently only single bond supported). In case of the linker,
    the R1 atom is selected.
    rgroup: fragment passed as rdkit molecule
    return: tuple (ratom, ratom_neighbour)
    """

    # find the R groups in the molecule
    ratoms = [atom for atom in R_group.GetAtoms() if atom.GetAtomicNum() == 0]
    if not len(ratoms):
        raise Exception(
            "The R-group does not have R-atoms (Atoms with index == 0, visualised with a '*' character)"
        )

    # if it is a linker, it will have more than 1 R group, pick the one with index 1
    if len(ratoms) == 1:
        atom = ratoms[0]
    elif is_linker(R_group):
        # find the attachable point
        ratoms = [atom for atom in ratoms if atom.GetAtomMapNum() == 1]
        atom = ratoms[0]
    else:
        raise Exception(
            "Either missing R-atoms, or more than two R-atoms. "
            '"Atom.GetAtomicNum" should be 0 for the R-atoms, and in the case of the linker,  '
            '"Atom.GetAtomMapNum" has to specify the order (1,2) '
        )

    neighbours = atom.GetNeighbors()
    if len(neighbours) > 1:
        raise NotImplementedError(
            "The linking R atom in the R group has two or more attachment points. "
        )

    return atom, neighbours[0]


def merge_R_group(mol, R_group, replaceIndex):
    """function originally copied from
    https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py
    """

    # the linking R atom on the R group
    rgroup_R_atom, R_atom_neighbour = __getAttachmentVector(R_group)
    print(f"Rgroup atom index {rgroup_R_atom} neighbouring {R_atom_neighbour}")

    # atom to be replaced in the molecule
    replace_atom = mol.GetAtomWithIdx(replaceIndex)
    assert (
        len(replace_atom.GetNeighbors()) == 1
    ), "The atom being replaced on the molecule has more neighbour atoms than 1. Not supported."
    replace_atom_neighbour = replace_atom.GetNeighbors()[0]

    # align the Rgroup
    AlignMol(
        R_group,
        mol,
        atomMap=(
            (R_atom_neighbour.GetIdx(), replace_atom.GetIdx()),
            (rgroup_R_atom.GetIdx(), replace_atom_neighbour.GetIdx()),
        ),
    )

    # merge the two molecules
    combined = Chem.CombineMols(mol, R_group)
    emol = Chem.EditableMol(combined)

    # connect
    bond_order = rgroup_R_atom.GetBonds()[0].GetBondType()
    emol.AddBond(
        replace_atom_neighbour.GetIdx(),
        R_atom_neighbour.GetIdx() + mol.GetNumAtoms(),
        order=bond_order,
    )
    # -1 accounts for the removed linking atom on the template
    emol.RemoveAtom(rgroup_R_atom.GetIdx() + mol.GetNumAtoms())
    # remove the linking atom on the template
    emol.RemoveAtom(replace_atom.GetIdx())

    merged = emol.GetMol()
    Chem.SanitizeMol(merged)

    # if the molecule was previously merged
    if hasattr(mol, "template") and mol.template is not None:
        # mol already had a connected e.g. a linker, therefore we use the area without the linker
        template = mol.template
    else:
        # prepare the template
        etemp = Chem.EditableMol(mol)
        etemp.RemoveAtom(replace_atom.GetIdx())
        template = etemp.GetMol()

    with_template = RMol(merged)
    with_template._save_template(template)
    # save the group
    with_template._save_rgroup(R_group)

    if is_linker(R_group):
        # the linker's label = 1 was used for the merging,
        # rename label = 2 to 0 to turn it into a simple R-group
        for atom in with_template.GetAtoms():
            if atom.GetAtomMapNum() == 2:
                atom.SetAtomMapNum(0)

    return with_template


def ic50(x):
    return 10 ** (-x - -9)


class RInterface:
    """
    This is a shared interface for a molecule and a list of molecules.

    The main purpose is to allow using the same functions on a single molecule and on a group of them.
    """

    def rep2D(self, **kwargs):
        pass

    def toxicity(self):
        pass

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        pass

    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        pass


class RMol(rdkit.Chem.rdchem.Mol, RInterface):
    """
    RMol is essentially a wrapper around RDKit Mol with
    tailored functionalities for attaching R groups, etc.

    :param rmol: when provided, energies and additional metadata is preserved.
    :type rmol: RMol
    :param template: Provide the original molecule template
        used for this RMol.
    """

    gnina_dir = None

    def __init__(self, *args, id=None, template=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(args[0], RMol) or isinstance(args[0], rdkit.Chem.Mol):
            self.template = args[0].template if hasattr(args[0], "template") else None
            self.rgroup = args[0].rgroup if hasattr(args[0], "rgroup") else None
            self.opt_energies = (
                args[0].opt_energies if hasattr(args[0], "opt_energies") else None
            )
            self.id = args[0].id if hasattr(args[0], "id") else None
        else:
            self.template = template
            self.rgroup = None
            self.opt_energies = None
            self.id = id

    def _save_template(self, mol):
        self.template = RMol(copy.deepcopy(mol))

    def _save_rgroup(self, rgroup):
        self.rgroup = rgroup

    def _save_opt_energies(self, energies):
        self.opt_energies = energies

    def toxicity(self):
        """
        Assessed various ADMET properties, including
         - Lipinksi rule of 5 properties,
         - the presence of unwanted substructures
         - problematic functional groups
         - synthetic accessibility

         :return: a row of a dataframe with the descriptors
         :rtype: dataframe
        """
        df = tox_props(self)
        # add an index column to the front
        df.insert(0, "ID", self.id)
        df.set_index("ID", inplace=True)

        # add a column with smiles
        df = df.assign(Smiles=[Chem.MolToSmiles(self)])

        return df

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        """
        Generate conformers using the RDKIT's ETKDG. The generated conformers
        are embedded into the template structure. In other words,
        any atoms that are common with the template structure,
        should have the same coordinates.

        :param num_conf: fixme
        :param minimum_conf_rms: The minimum acceptable difference in the RMS in any new generated conformer.
            Conformers that are too similar are discarded.
        :type minimum_conf_rms: float
        :param flexible: A list of indices that are common with the template molecule
            that should have new coordinates.
        :type flexible: List[int]
        """
        cons = generate_conformers(self, num_conf, minimum_conf_rms, **kwargs)
        self.RemoveAllConformers()
        [self.AddConformer(con, assignId=True) for con in cons.GetConformers()]

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Enumerate the conformers inside of the receptor by employing
        ANI2x, a hybrid machine learning / molecular mechanics (ML/MM) approach.
        ANI2x is neural nework potential for the ligand energetics
        but works only for the following atoms: H, C, N, O, F, S, Cl.

        Open Force Field Parsley force field is used for intermolecular interactions with the receptor.

        :param sigma_scale_factor: is used to scale the Lennard-Jones radii of the atoms.
        :param relative_permittivity: is used to scale the electrostatic interactions with the protein.
        :param water_model: can be used to set the force field for any water molecules present in the binding site.
        """
        if self.GetNumConformers() == 0:
            print("Warning: no conformers so cannot optimise_in_receptor. Ignoring.")
            return

        from .receptor import ForceField, optimise_in_receptor

        opt_mol, energies = optimise_in_receptor(self, *args, **kwargs)
        # replace the conformers with the optimised ones
        self.RemoveAllConformers()
        [
            self.AddConformer(conformer, assignId=True)
            for conformer in opt_mol.GetConformers()
        ]
        # save the energies
        self._save_opt_energies(energies)

        # build a dataframe with the molecules
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(energies),
                "Conformer": conformer_ids,
                "Energy": energies,
            }
        )

        return df

    def sort_conformers(self, energy_range=5):
        """
        For the given molecule and the conformer energies order the energies
         and only keep any conformers with in the energy range of the
         lowest energy conformer.

        :param energy_range: The energy range (kcal/mol),
            above the minimum, for which conformers should be kept.
        """
        if self.GetNumConformers() == 0:
            print("An rmol doesn't have any conformers. Ignoring.")
            return None
        elif self.opt_energies is None:
            raise AttributeError(
                "Please run the optimise_in_receptor in order to generate the energies first. "
            )

        from .receptor import sort_conformers

        final_mol, final_energies = sort_conformers(
            self, self.opt_energies, energy_range=energy_range
        )
        # overwrite the current confs
        self.RemoveAllConformers()
        [
            self.AddConformer(conformer, assignId=True)
            for conformer in final_mol.GetConformers()
        ]
        self._save_opt_energies(final_energies)

        # build a dataframe with the molecules
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(final_energies),
                "Conformer": conformer_ids,
                "Energy": final_energies,
            }
        )

        return df

    def rep2D(self, **kwargs):
        """
        Use RDKit and get a 2D diagram.
        Uses Compute2DCoords and Draw.MolToImage function

        Works with IPython Notebook.

        :param **kwargs: are passed further to Draw.MolToImage function.
        """
        return rep2D(self, **kwargs)

    def rep3D(
        self, view=None, prody=None, template=False, confIds: Optional[List[int]] = None
    ):
        """
        Use py3Dmol to obtain the 3D view of the molecule.

        Works with IPython Notebook.

        :param view: a view to which add the visualisation. Useful if one wants to 3D view
            multiple conformers in one view.
        :type view: py3Dmol view instance (None)
        :param prody: A prody protein around which a view 3D can be created
        :type prody: Prody instance (Default: None)
        :param template: Whether to visualise the original 3D template as well from which the molecule was made.
        :type template: bool (False)
        :param confIds: Select the conformations for display.
        :type confIds: List[int]
        """
        if prody is not None:
            view = view3D(prody)

        if view is None:
            view = py3Dmol.view(width=400, height=400, viewergrid=(1, 1))

        for conf in self.GetConformers():
            # ignore the confIds we've not asked for
            if confIds is not None and conf.GetId() not in confIds:
                continue

            mb = Chem.MolToMolBlock(self, confId=conf.GetId())
            view.addModel(mb, "lig")

            # use reverse indexing to reference the just added conformer
            # http://3dmol.csb.pitt.edu/doc/types.html#AtomSelectionSpec
            # cmap = plt.get_cmap("tab20c")
            # hex = to_hex(cmap.colors[i]).split('#')[-1]
            view.setStyle({"model": -1}, {"stick": {}})

        if template:
            mb = Chem.MolToMolBlock(self.template)
            view.addModel(mb, "template")
            # show as sticks
            view.setStyle({"model": -1}, {"stick": {"color": "0xAF10AB"}})

        # zoom to the last added model
        view.zoomTo({"model": -1})
        return view

    def remove_clashing_confs(self, prot, min_dst_allowed=1.0):
        """
        Removing conformations that class with the protein.
        Note that the original conformer should be well docked into the protein,
        ideally with some space between the area of growth and the protein,
        so that any growth on the template doesn't automatically cause
        clashes.

        :param prot: The protein against which the conformers should be tested.
        :type prot: Prody instance
        :param min_dst_allowed: If any atom is within this distance in a conformer, the
         conformer will be deleted.
        :type min_dst_allowed: float in Angstroms
        """
        prot_coords = prot.getCoords()

        counter = 0
        for conf in list(self.GetConformers())[::-1]:
            confid = conf.GetId()

            # for each atom check how far it is from the protein atoms
            eacheach_shortest = []
            for coord in conf.GetPositions():
                shortest = np.min(np.sqrt(np.sum((coord - prot_coords) ** 2, axis=1)))
                eacheach_shortest.append(shortest)

            min_dst = np.min(eacheach_shortest)

            if min_dst < min_dst_allowed:
                self.RemoveConformer(confid)
                print(f"Clash with the protein. Removing conformer id: {confid}")

    @staticmethod
    def set_gnina(loc):
        """
        Set the location of the binary file gnina. This could be your own compiled directory,
        or a directory where you'd like it to be downloaded.

        By default, gnina path is to the working directory (~500MB).

        :param loc: path to gnina binary file. E.g. /dir/path/gnina. Note that right now gnina should
         be a binary file with that specific filename "gnina".
        :type loc: str
        """
        # set gnina location
        path = Path(loc)
        if path.is_file():
            assert path.name == "gnina", 'Please ensure gnina binary is named "gnina"'
            RMol.gnina_dir = path.parent
        else:
            raise Exception("The path is not the binary file gnina")
        # extend this with running a binary check

    @staticmethod
    def _check_download_gnina():
        """
        Check if gnina works. Otherwise, download it.
        """
        if RMol.gnina_dir is None:
            # assume it is in the current directory
            RMol.gnina_dir = os.getcwd()

        # check if gnina works
        try:
            subprocess.run(
                ["./gnina", "--help"], capture_output=True, cwd=RMol.gnina_dir
            )
            return
        except FileNotFoundError as E:
            pass

        # gnina is not found, try downloading it
        print(f"Gnina not found or set. Download gnina (~500MB) into {RMol.gnina_dir}")
        gnina = os.path.join(RMol.gnina_dir, "gnina")
        # fixme - currently download to the working directory (Home could be more applicable).
        urlretrieve(
            "https://github.com/gnina/gnina/releases/download/v1.0.1/gnina",
            filename=gnina,
        )
        # make executable (chmod +x)
        mode = os.stat(gnina).st_mode
        os.chmod(gnina, mode | stat.S_IEXEC)

        # check if it works
        subprocess.run(["./gnina", "--help"], capture_output=True, cwd=RMol.gnina_dir)

    def gnina(self, receptor_file):
        """
        Use gnina to extract CNNaffinity, and convert it into IC50.

        LIMITATION: currenly the gnina binaries do not support Mac.

        :param receptor_file: Path to the receptor file.
        :type receptor_file: str
        """
        self._check_download_gnina()

        # obtain the absolute file to the receptor
        receptor = Path(receptor_file)
        if not receptor.exists():
            raise ValueError(f'Your receptor "{receptor_file}" does not seem to exist.')

        # make a temporary sdf file for gnina
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sdf")
        with Chem.SDWriter(tmp.name) as w:
            for conformer in self.GetConformers():
                w.write(self, confId=conformer.GetId())

        # run the code on the sdf
        process = subprocess.run(
            [
                "./gnina",
                "--score_only",
                "-l",
                tmp.name,
                "-r",
                receptor.absolute(),
                "--seed",
                "0",
                "--stripH",
                "False",
            ],
            capture_output=True,
            cwd=RMol.gnina_dir,
        )
        output = process.stdout.decode("utf-8")
        CNNaffinities_str = re.findall(r"CNNaffinity: (-?\d+.\d+)", output)

        # convert to floats
        CNNaffinities = list(map(float, CNNaffinities_str))

        # generate IC50 from the CNNaffinities
        ic50s = list(map(ic50, CNNaffinities))

        # create a dataframe
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(CNNaffinities),
                "Conformer": conformer_ids,
                "CNNaffinity": CNNaffinities,
                "CNNaffinity->IC50s": ic50s,
            }
        )

        return df

    def to_file(self, file_name: str):
        """
        Write the molecule and all conformers to file.

        Note:
            The file type is worked out from the name extension by splitting on `.`.
        """
        file_type = file_name.split(".")[-1]

        # SDF has its own multi-frame writer
        if file_type.lower() == 'sdf':
            with Chem.SDWriter(file_name) as SD:
                for conformer in self.GetConformers():
                    SD.write(self, confId=conformer.GetId())
            return

        write_functions = {
            "mol": Chem.MolToMolBlock,
            "pdb": Chem.MolToPDBBlock,
            "xyz": Chem.MolToXYZBlock,
        }

        func = write_functions.get(file_type, None)
        if func is None:
            raise RuntimeError(
                f"The file type {file_type} is not support please chose from {write_functions.keys()}"
            )

        with open(file_name, "w") as output:
            for conformer in self.GetConformers():
                output.write(func(self, confId=conformer.GetId()))

    def df(self):
        """
        Generate a pandas dataframe row for this molecule with SMILES.

        :returns: pandas dataframe row.
        """
        df = pandas.DataFrame(
            {
                "ID": [self.id],
                "Smiles": [Chem.MolToSmiles(self)],
            }
        )
        # attach energies if they're present
        if self.opt_energies:
            df = df.assign(
                Energies=", ".join([str(e) for e in sorted(self.opt_energies)])
            )

        df.set_index(["ID"], inplace=True)
        return df

    def _repr_html_(self):
        # return a dataframe with the rdkit visualisation

        df = self.df()

        # add a visualisation column
        PandasTools.AddMoleculeColumnToFrame(
            df, "Smiles", "Molecule", includeFingerprints=True
        )
        return df._repr_html_()


class RGroupGrid(mols2grid.MolGrid):
    """
    A wrapper around the mols to grid class to load and process the r group folders locally.
    """

    def __init__(self):
        dataframe = RGroupGrid._load_molecules()

        super(RGroupGrid, self).__init__(
            dataframe, removeHs=True, mol_col="Mol", use_coords=False, name="m2"
        )

    @staticmethod
    def _load_molecules() -> pandas.DataFrame:
        """
        Load the local r groups into rdkit molecules
        """
        molecules = []
        names = []

        builtin_rgroups = Path(__file__).parent / "data" / "rgroups" / "library.sdf"
        print(open(str(builtin_rgroups)).read())
        for rgroup in Chem.SDMolSupplier(str(builtin_rgroups.resolve()), removeHs=False):
            molecules.append(rgroup)
            names.append(rgroup.GetProp('SMILES'))

            # highlight the attachment atom
            for atom in rgroup.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    setattr(rgroup, "__sssAtoms", [atom.GetIdx()])

        return pandas.DataFrame({"Mol": molecules, "Name": names})

    def _ipython_display_(self):
        from IPython.display import display, update_display, display_html

        subset = ["img", "Name", "mols2grid-id"]
        display_html(self.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        # use the new API
        df = self.get_selection()
        # now get a list of the molecules
        return list(df["Mol"])


class RLinkerGrid(mols2grid.MolGrid):
    """
    A wrapper around the mols to grid class to load and process the linker folders locally.
    """

    def __init__(self):
        dataframe = self._load_molecules()

        super(RLinkerGrid, self).__init__(
            dataframe,
            removeHs=True,
            mol_col="Mol",
            use_coords=False,
            name="m1",
            prerender=False,
        )

    def _load_molecules(self) -> pandas.DataFrame:
        """
        Load the local linkers into rdkit molecules
        """
        builtin_rlinkers = Path(__file__).parent / "data" / "linkers" / "library.sdf"
        linker_mols = list(Chem.SDMolSupplier(str(builtin_rlinkers.resolve()), removeHs=False))

        # sort the linkers so that [R1]C[R2] is next to [R2]C[R1] in the grid
        # fixme: preorder them in the .sdf and use that order instead
        linker_mols = sorted(
            linker_mols,
            key=lambda l: l.GetProp('SMILES').replace("[*:1]", "R").replace("[*:2]", "R"),
        )

        linkers = []
        for mol in linker_mols:
            # generate a searchable name in the form of a simple SMILE without hydrogens
            name = (
                Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=False)
                .replace("[*:1]", "R1")
                .replace("[*:2]", "R2")
            )

            # simplify the names for the user
            linkers.append(
                {
                    "Mol": mol,
                    "Name": name,
                    "Common": mol.GetIntProp(
                        "SmileIndex"
                    ),  # extract the index property from the original publication
                }
            )

        # presort using the original publication index
        linkers = sorted(linkers, key=lambda i: i["Common"])

        return pandas.DataFrame(linkers)

    def _ipython_display_(self):
        from IPython.display import display

        subset = ["img", "Name", "mols2grid-id"]
        return display(self.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        # use the new API
        df = self.get_selection()
        # now get a list of the molecules
        return list(df["Mol"])


def link(Rgroups, Rlinkers, One2One=False):
    """

    :param Rgroups:
    :param Rlinkers:
    :param One2One: If True, for each selected RGroup, a selected linker with the same index is added.
        Alternatively, each linker is added to each RGroup resulting in n x m number of combinsions, with n being
        the number of RGroups and m the number of linkers.
    :return:
    """

    # convert rgroups/linkers to smiles
    rgroup_smiles = [Chem.MolToSmiles(mol) for mol in Rgroups]
    linker_smiles = [Chem.MolToSmiles(mol) for mol in Rlinkers]

    # loop over and combine smiles
    combined = []
    if One2One:
        for linker, rgroup in zip(linker_smiles, rgroup_smiles):
            combined.append(rgroup.replace("*", "*" + linker))
    else:
        for linker in linker_smiles:
            for rgroup in rgroup_smiles:
                combined.append(rgroup.replace("*", "*" + linker))

    # generate 3D conformers for each molecule
    mols = []
    for smile in combined:
        mol = Chem.MolFromSmiles(smile)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, params=AllChem.ETKDGv3())
        mols.append(mol)
    return mols


class RList(RInterface, list):
    """
    Streamline working with RMol by presenting the same interface on the list,
    and allowing to dispatch the functions to any single member.
    """

    def rep2D(self, subImgSize=(400, 400), **kwargs):
        return Draw.MolsToGridImage(
            [mol.rep2D(rdkit_mol=True, **kwargs) for mol in self], subImgSize=subImgSize
        )

    @staticmethod
    def _append_jupyter_visualisation(df):
        if "Smiles" not in df:
            return

        # add a column with the visualisation
        PandasTools.AddMoleculeColumnToFrame(
            df, "Smiles", "Molecule", includeFingerprints=True
        )

    def toxicity(self):
        df = pandas.concat([m.toxicity() for m in self] + [pandas.DataFrame()])
        RList._append_jupyter_visualisation(df)
        return df

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            rmol.generate_conformers(num_conf, minimum_conf_rms, **kwargs)

    def GetNumConformers(self):
        return [rmol.GetNumConformers() for rmol in self]

    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            rmol.remove_clashing_confs(prot, min_dst_allowed=min_dst_allowed)

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Replace the current molecule with the optimised one. Return lists of energies.
        """
        # return pandas.concat([m.toxicity() for m in self])

        dfs = []
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            dfs.append(rmol.optimise_in_receptor(*args, **kwargs))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def sort_conformers(self, energy_range=5):
        dfs = []
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            dfs.append(rmol.sort_conformers(energy_range))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def gnina(self, receptor_file):
        dfs = []
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            dfs.append(rmol.gnina(receptor_file))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def discard_missing(self):
        """
        Remove from this list the molecules that have no conformers
        """
        removed = []
        for rmol in self[::-1]:
            if rmol.GetNumConformers() == 0:
                rmindex = self.index(rmol)
                print(
                    f"Discarding a molecule (id {rmindex}) due to the lack of conformers. "
                )
                self.remove(rmol)
                removed.append(rmindex)
        return removed

    @property
    def dataframe(self):
        return pandas.concat([rmol.df() for rmol in self] + [pandas.DataFrame()])

    def _repr_html_(self):
        # return the dataframe with the visualisation column of the dataframe
        df = self.dataframe
        RList._append_jupyter_visualisation(df)
        return df._repr_html_()


def build_molecules(
    templates: Union[Chem.Mol, List[Chem.Mol]],
    r_groups: Union[Chem.Mol, List[Chem.Mol], int],
    attachment_points: Optional[List[int]] = None,
):
    """
    For the given core molecule and list of attachment points
     and r groups enumerate the possible molecules and
     return a list of them.

    :param template: The core scaffold molecule to attach the r groups to, or a list of them.
    :param r_group: The list of rdkit molecules which should be considered
      r groups or the RGroup Grid with highlighted molecules.
    :param attachment_points: The list of atom index in the core ligand
      that the r groups should be attached to. If it is empty, connecting points are sought out and matched.
    """

    # fixme - special case after changing the API: remove in the future
    # This is a temporary warning about the change in the interface.
    # This change is because there are situations where the attachment_points do not need to be passed to the function.
    if (
        isinstance(r_groups, list)
        and len(r_groups) > 0
        and isinstance(r_groups[0], int)
    ):
        print(
            'Warning: second argument is detected to be an integer. It is now "r_groups" '
            "whereas attachement_points are provided as the 3rd argument. "
        )
        raise Exception(
            "Please note that after adding the linker to FEgrow (version 1.1), "
            'the "build_molecules" function interface has changed to'
            ' "build_molecules(core_ligand, r_groups, attachment_points)". '
        )

    # ensure template and r_group are lists
    if not issubclass(templates.__class__, List):
        templates = [templates]
    if not issubclass(r_groups.__class__, List):
        r_groups = [r_groups]

    # make a deep copy of r_groups/linkers to ensure we don't modify the library
    templates = [copy.deepcopy(mol) for mol in templates]
    r_groups = [copy.deepcopy(mol) for mol in r_groups]

    # get attachment points for each template
    if not attachment_points:
        # attempt to generate the attachment points by picking the joining molecule
        # case: a list of templates previously joined with linkers requires iterating over them
        attachment_points = [__getAttachmentVector(lig)[0].GetIdx() for lig in templates]

    if not attachment_points:
        raise Exception("Could not find attachement points. ")

    if len(attachment_points) != len(templates):
        raise Exception(
            f"There must be one attachment point for each template. "
            f"Provided attachement points = {len(attachment_points)} "
            f"with templates number: {len(templates)}"
        )

    combined_mols = RList()
    id_counter = 0
    for atom_idx, core_ligand in zip(attachment_points, templates):
        for r_mol in r_groups:
            core_mol = RMol(copy.deepcopy(core_ligand))
            merged_mol = merge_R_group(
                mol=core_mol, R_group=r_mol, replaceIndex=atom_idx
            )
            # assign the identifying index to the molecule
            merged_mol.id = id_counter
            combined_mols.append(merged_mol)
            id_counter += 1

    return combined_mols
