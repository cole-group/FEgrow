import abc
import copy
import functools
import itertools
import logging
import os
import typing
import warnings
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import List, Optional, Union, Sequence
import urllib
import time

import functools

import pandas as pd
import requests
import numpy as np
import mols2grid
import openmm
import openmm.app
import pandas
import pint_pandas
import prody as prody_package
import py3Dmol
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
import dask
from dask.distributed import LocalCluster, Client, Scheduler, Worker
import modAL
from sklearn import gaussian_process

from .builder import build_molecules_with_rdkit
from .conformers import generate_conformers
from .receptor import optimise_in_receptor, sort_conformers
from .toxicity import tox_props

# default options
pandas.set_option("display.precision", 3)

logger = logging.getLogger(__name__)

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


class RInterface:
    """
    This is a shared interface for a molecule and a list of molecules.

    The main purpose is to allow using the same functions on a single molecule and on a group of them.
    """

    @abc.abstractmethod
    def rep2D(self, **kwargs):
        ...

    @abc.abstractmethod
    def toxicity(self):
        pass

    @abc.abstractmethod
    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        pass

    @abc.abstractmethod
    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        pass


class RMol(RInterface, rdkit.Chem.rdchem.Mol):
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

    def rep2D(self, idx=-1, rdkit_mol=False, h=True, **kwargs):
        """
        Use RDKit and get a 2D diagram.
        Uses Compute2DCoords and Draw.MolToImage function

        Works with IPython Notebook.

        :param **kwargs: are passed further to Draw.MolToImage function.
        """
        numbered = copy.deepcopy(self)

        if not h:
            numbered = Chem.RemoveHs(numbered)

        numbered.RemoveAllConformers()
        if idx:
            for atom in numbered.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
        Chem.AllChem.Compute2DCoords(numbered)

        if rdkit_mol:
            return numbered
        else:
            return Draw.MolToImage(numbered, **kwargs)

    def rep3D(
        self,
        view=None,
        prody=None,
        template=False,
        confIds: Optional[List[int]] = None,
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
            view = prody_package.proteins.functions.view3D(prody)

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

    def remove_clashing_confs(self,
                              protein: Union[str, openmm.app.PDBFile], min_dst_allowed=1.0):
        """
        Removing conformations that class with the protein.
        Note that the original conformer should be well docked into the protein,
        ideally with some space between the area of growth and the protein,
        so that any growth on the template doesn't automatically cause
        clashes.

        :param protein: The protein against which the conformers should be tested.
        :type protein: filename or the openmm PDBFile instance or prody instance
        :param min_dst_allowed: If any atom is within this distance in a conformer, the
         conformer will be deleted.
        :type min_dst_allowed: float in Angstroms
        """
        if type(protein) is str:
            protein = openmm.app.PDBFile(protein)

        if type(protein) is openmm.app.PDBFile:
            protein_coords = protein.getPositions(asNumpy=True).in_units_of(openmm.unit.angstrom)._value
        else:
            protein_coords = protein.getCoords()

        rm_counter = 0
        for conf in list(self.GetConformers()):
            # for each atom check how far it is from the protein atoms
            min_dst = 999_999_999  # arbitrary large distance

            for point in conf.GetPositions():
                shortest = np.min(
                    np.sqrt(np.sum((point - protein_coords) ** 2, axis=1))
                )
                min_dst = min(min_dst, shortest)

                if min_dst < min_dst_allowed:
                    self.RemoveConformer(conf.GetId())
                    logger.debug(
                        f"Clash with the protein. Removing conformer id: {conf.GetId()}"
                    )
                    rm_counter += 1
                    break
        print(f"Removed {rm_counter} conformers. ")

        # return self for Dask
        return self


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
        urllib.request.urlretrieve(
            "https://github.com/gnina/gnina/releases/download/v1.0.1/gnina",
            filename=gnina,
        )
        # make executable (chmod +x)
        mode = os.stat(gnina).st_mode
        os.chmod(gnina, mode | stat.S_IEXEC)

        # check if it works
        subprocess.run(
            ["./gnina", "--help"], capture_output=True, check=True, cwd=RMol.gnina_dir
        )

    def gnina(self, receptor_file, gnina_gpu=False):
        """
        Use GNINA to extract CNNaffinity, which we also recalculate to Kd (nM)

        LIMITATION: The GNINA binary does not support MAC/Windows.

        Please cite GNINA accordingly:
        McNutt, Andrew T., Paul Francoeur, Rishal Aggarwal, Tomohide Masuda, Rocco Meli, Matthew Ragoza,
        Jocelyn Sunseri, and David Ryan Koes. "GNINA 1.0: molecular docking with deep learning."
        Journal of cheminformatics 13, no. 1 (2021): 1-20.

        :param receptor_file: Path to the receptor file.
        :type receptor_file: str
        """
        RMol._check_download_gnina()
        gnina_path = os.path.join(RMol.gnina_dir, 'gnina')

        if not isinstance(receptor_file, str) and not isinstance(receptor_file, Path):
            raise ValueError(f"gnina function requires a file path to the receptor. Instead, was given: {type(receptor_file)}")

        # get the absolute path
        receptor = Path(receptor_file)
        if not receptor.exists():
            raise ValueError(f'Your receptor "{receptor_file}" does not seem to exist.')

        _, CNNaffinities = gnina(self, receptor, gnina_path, gnina_gpu=gnina_gpu)

        return RMol._parse_gnina_cnnaffinities(self, CNNaffinities)

    @staticmethod
    def _ic50(x):
        return 10 ** (-x - -9)

    @staticmethod
    def _parse_gnina_cnnaffinities(mol, CNNAffinities, mol_id=0):
        # generate IC50 from the CNNaffinities
        ic50s = list(map(RMol._ic50, CNNAffinities))

        # add nM units
        ic50s_nM = pandas.Series(ic50s, dtype="pint[nM]")

        # create a dataframe
        conformer_ids = [c.GetId() for c in mol.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [mol_id] * len(CNNAffinities),
                "Conformer": conformer_ids,
                "CNNaffinity": CNNAffinities,
                "Kd": ic50s_nM,
            }
        )

        return df

    def to_file(self, filename: str):
        """
        Write the molecule and all conformers to file.

        Note:
            The file type is worked out from the name extension by splitting on `.`.
        """
        file_type = Path(filename).suffix.lower()

        writers = {
            ".mol": Chem.MolToMolFile,
            ".sdf": Chem.SDWriter,
            ".pdb": functools.partial(Chem.PDBWriter, flavor=1),
            ".xyz": Chem.MolToXYZFile,
        }

        func = writers.get(file_type, None)
        if func is None:
            raise RuntimeError(
                f"The file type {file_type} is not support please chose from {writers.keys()}"
            )

        if file_type in ['.pdb', '.sdf']:
            # multi-frame writers

            with writers[file_type](filename) as WRITER:
                for conformer in self.GetConformers():
                    WRITER.write(self, confId=conformer.GetId())
            return

        writers[file_type](self, filename)


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


class DaskTasks:
    @staticmethod
    @dask.delayed
    def obabel_protonate(smi):
        return subprocess.run(['obabel', f'-:{smi}', '-osmi', '-p', '7', '-xh'],
                              capture_output=True).stdout.decode().strip()

    @staticmethod
    @dask.delayed
    def scaffold_check(smih, scaffold):
        """

        :param smih:
        :param scaffold:
        :return: [has_scaffold_bool, protonated_smiles]
        """
        params = Chem.SmilesParserParams()
        params.removeHs = False

        mol = Chem.MolFromSmiles(smih, params=params)
        if mol is None:
            return False, None

        if mol.HasSubstructMatch(scaffold):
            return True, smih

        return False, None

class ChemSpace: # RInterface
    """
    Streamline working with many RMols or a specific chemical space by employing a pandas dataframe,
    in combination with Dask for parallellisation.
    """

    def rep2D(self, subImgSize=(400, 400), **kwargs):
        return Draw.MolsToGridImage(
            [row.Mol.rep2D(rdkit_mol=True, **kwargs) for i, row in self.df.iterrows()], subImgSize=subImgSize
        )

    _dask_cluster = None
    _dask_client = None

    DATAFRAME_DEFAULT_VALUES = {"Smiles": [], # Smiles will always be replaced when inserting new data.
                                "Mol": pandas.NA,
                                "score": pandas.NA, # any scoring function, by default cnnaffinity
                                "h": pandas.NA,
                                "Training": False, # will be used for AL
                                "Success": pandas.NA, # true if built, false if built unsuccessfully
                                "enamine_searched": False,
                                "enamine_id": pandas.NA}

    def __init__(self, data=None, data_indices=None, dask_cluster=None, dask_local_cluster_kwargs={}):
        if data is None:
            data = ChemSpace.DATAFRAME_DEFAULT_VALUES

        self.df = pandas.DataFrame(data, index=data_indices)

        ChemSpace._dask_cluster = dask_cluster

        if ChemSpace._dask_cluster is None:
            logger.info("No Dask cluster configured. Creating a local cluster of threads. ")
            warnings.warn("ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. "
                          "Use a Dask cluster with processes as a work around "
                          "(see the documentation for an example of this workaround) .")

            kwargs = {"n_workers": None,
                      "processes": False,  # turn off Nanny to avoid the problem
                                           # with loading of the main file (ie executing it)
                      "dashboard_address": ":8989",
                      **dask_local_cluster_kwargs
                      }
            ChemSpace._dask_cluster = LocalCluster(**kwargs)
            # ChemSpace._dask_cluster = Scheduler()
            # ChemSpace._dask_cluster = LocalCluster(preload_nanny=["print('Hi Nanny')"],
            #                                        preload=["pint"], n_workers=1
            #                                        ) #asynchronous=True)

        ChemSpace._dask_client = Client(ChemSpace._dask_cluster) #ChemSpace._dask_cluster, asynchronous=True)
        print(f"Dask can be watched on {ChemSpace._dask_client.dashboard_link}")

        self._scaffolds = []
        self._model = None
        self._query = None
        self._query_label = None

    def set_dask_caching(self, bytes_num=4e9):
        # Leverage 4 gigabytes of memory
        from dask.cache import Cache
        self.cache = Cache(bytes_num)
        self.cache.register()

    @property
    def dask_client(self):
        if self._dask_client is None:
            print("Initialising Local Dask")
            self._dask_cluster = LocalCluster()
            self._dask_client = Client(ChemSpace._dask_cluster)

        return self._dask_client


    @staticmethod
    def _add_smiles_2D_visualisation(df):
        if "Smiles" not in df:
            return

        # add a column with the visualisation
        Chem.PandasTools.AddMoleculeColumnToFrame(
            df, "Smiles", "Molecule", includeFingerprints=True
        )

    def toxicity(self):
        df = pandas.concat([m.toxicity() for m in self] + [pandas.DataFrame()])
        ChemSpace._add_smiles_2D_visualisation(df)
        return df

    def generate_conformers(
            self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        # prepare the dask parameters to be send
        num_conf = dask.delayed(num_conf)
        minimum_conf_rms = dask.delayed(minimum_conf_rms)

        # create the dask jobs
        delayed_generate_conformers = dask.delayed(generate_conformers)
        jobs = {}
        for i, row in self.df.iterrows():
            jobs[row.Mol] = (delayed_generate_conformers(row.Mol, num_conf, minimum_conf_rms, **kwargs))

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        for mol, result in results.items():
            new_mol = result.result()
            mol.RemoveAllConformers()
            [mol.AddConformer(c, assignId=True) for c in new_mol.GetConformers()]

    def GetNumConformers(self):
        return [rmol.GetNumConformers() for rmol in self]

    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        prot = dask.delayed(prot)
        min_dst_allowed = dask.delayed(min_dst_allowed)

        # create the dask jobs
        delayed_remove_clashing_confs = dask.delayed(RMol.remove_clashing_confs)
        jobs = {}
        for i, row in self.df.iterrows():
            jobs[row.Mol] = delayed_remove_clashing_confs(row.Mol, prot, min_dst_allowed=min_dst_allowed)

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        for mol, result in results.items():
            unclashed_mol = result.result()
            mol.RemoveAllConformers()
            [mol.AddConformer(c) for c in unclashed_mol.GetConformers()]

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Return lists of energies.
        """

        # daskify parameters
        args = [dask.delayed(arg) for arg in args]
        kwargs = {k: dask.delayed(v) for k, v in kwargs.items()}

        # create the dask jobs
        delayed_optimise_in_receptor = dask.delayed(optimise_in_receptor)
        jobs = {}
        for i, row in self.df.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(f"Warning: mol {i} has no conformers. Ignoring receptor optimisation.")
                continue

            jobs[row.Mol] = delayed_optimise_in_receptor(row.Mol, *args, **kwargs)

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        dfs = []
        for mol, result in results.items():
            opt_mol, energies = result.result()
            mol.RemoveAllConformers()
            # replace the conformers with the optimised ones
            [mol.AddConformer(c) for c in opt_mol.GetConformers()]

            mol.SetProp("energies", str(energies))
            dfs.append(pandas.DataFrame({}))
            mol._save_opt_energies(energies)

    def sort_conformers(self, energy_range=5):
        dfs = []
        for i, row in self.df.iterrows():
            print(f"RMol index {i}")
            dfs.append(row.Mol.sort_conformers(energy_range))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def gnina(self, receptor_file):
        # daskify objects
        receptor_file = dask.delayed(receptor_file)
        RMol._check_download_gnina()
        gnina_path = dask.delayed(os.path.join(RMol.gnina_dir, 'gnina'))

        # create the dask jobs
        delayed_gnina = dask.delayed(gnina)
        jobs = {}
        for i, row in self.df.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(f"Warning: mol {i} has no conformers. Ignoring gnina.")
                continue

            jobs[i] = delayed_gnina(row.Mol, receptor_file, gnina_path)

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        dfs = []
        for i, result in results.items():
            _, cnnaffinities = result.result()
            df = RMol._parse_gnina_cnnaffinities(self.df.Mol[i], cnnaffinities, mol_id=i)
            dfs.append(df)

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def discard_missing(self):
        """
        Remove from this list the molecules that have no conformers
        """
        ids_to_remove = []
        for i, row in self.df.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(
                    f"Discarding a molecule (id {i}) due to the lack of conformers. "
                )
                ids_to_remove.append(i)

        self.df = self.df[~self.df.index.isin(ids_to_remove)]
        return ids_to_remove

    # @property
    # def dataframe(self):
    #     return pandas.concat([rmol.df() for rmol in self] + [pandas.DataFrame()])

    # def _ipython_display_(self):
    #     from IPython.display import display
    #
    #     df = self.dataframe
    #     RList._add_smiles_2D_visualisation(df)
    #     return display(df)

    def add_scaffold(self, template, atom_replacement_index=None):

        # check if any atom is marked for joining
        if atom_replacement_index is None:
            if not any(atom.GetAtomicNum() == 0 for atom in template.GetAtoms()):
                warnings.warn("The template does not have an attachement (Atoms with index 0, "
                                 "or in case of Smiles the * character. )")
        else:
            # mark the right atom for replacement by assigning it atomic number == 0
            template.GetAtomWithIdx(atom_replacement_index).SetAtomicNum(0)

        self._scaffolds.append(template)

    def add_rgroups(self, rgroups_linkers, rgroups2=None, alltoall=False):
        """
        Note that if they are Smiles:
         - if they have an * atom (e.g. RDKit atom.SetAtomicNum(0)), this will be used for attachment to the scaffold
         - if they don't have an * atom, the scaffold will be fitted as a substructure

        First link the linker to the scaffold. Then add the rgroups.

        :param rgroups2: A list of Smiles. Molecules will be accepted and converted to Smiles.
        :param linker: A molecule. Ideally it has 2 atatchement points.
        :return:
        """
        scaffold = dask.delayed(self._scaffolds[0])

        if not isinstance(rgroups_linkers, typing.Iterable):
            rgroups_linkers = [rgroups_linkers]

        if rgroups2 is not None and not isinstance(rgroups2, typing.Iterable):
            rgroups2 = [rgroups2]

        # create the dask jobs
        delayed_build_molecule = dask.delayed(build_molecule)

        jobs = [delayed_build_molecule(scaffold, linker) for linker in rgroups_linkers]

        # if linkers and rgroups are attached, add them in two iterations
        if rgroups2 is not None and not alltoall:
            # for each attached linker, attach an rgroup with the same position
            jobs = [delayed_build_molecule(scaffold_linked, rgroup)
                    for rgroup, scaffold_linked in
                    itertools.zip_longest(rgroups2, jobs, fillvalue=jobs[0])]
        elif rgroups2 is not None and alltoall:
            jobs = [delayed_build_molecule(scaffold_linked, rgroup)
                    for rgroup, scaffold_linked in
                    itertools.product(rgroups2, jobs)]

        results = self.dask_client.compute(jobs)
        built_mols = [r.result() for r in results]

        # get Smiles
        built_mols_smiles = [Chem.MolToSmiles(mol) for mol in built_mols]

        # extract the H indices used for attaching the scaffold
        hs = [mol.GetIntProp('attachment_point') for mol in built_mols]

        self.add_data({"Smiles": built_mols_smiles, "Mol": built_mols, "h": hs})

    def add_data(self, data):
        """

        :param data: dictionary {"Smiles": [], "h": [], ... }
        :return:
        """

        # ensure that the new indices start at the end
        last_index = max([self.df.index.max() + 1])
        if np.isnan(last_index):
            last_index = 0

        # ensure correct default values in the new rows
        data_with_defaults = ChemSpace.DATAFRAME_DEFAULT_VALUES.copy()
        data_with_defaults.update(data)

        # update the internal dataframe
        new_indices = range(last_index, last_index + len(data_with_defaults['Smiles']))
        prepared_data = pandas.DataFrame(data_with_defaults, index=new_indices)
        self.df = pandas.concat([self.df, prepared_data])
        return prepared_data

    def add_smiles(self, smiles_list, h=pandas.NA, protonate=False):
        """
        Add a list of Smiles into this ChemicalSpace

        :param h: which h was used to connect to the
        :param protonate: use openbabel to protonate each smile
        :return:
        """

        if protonate:
            delayed_protonations = [DaskTasks.obabel_protonate(smi) for smi in smiles_list]
            jobs = self.dask_client.compute(delayed_protonations)
            smiles_list = [job.result() for job in jobs]

        # convert the Smiles into molecules
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mols = [Chem.MolFromSmiles(smiles, params=params) for smiles in smiles_list]

        self.add_data({"Smiles": smiles_list, "Mol": mols, "h":h})

    def _evaluate_experimental(self, indices=None, num_conf=10, minimum_conf_rms=0.5, min_dst_allowed=1):
        """
        Generate the conformers and score the subset of molecules.

        E.g.
        :param indices: The indices in the dataframe to be run through the pipeline.
            If None, all molecules are evaluated.
        :return:
        """

        if indices is None:
            # evaluate all molecules
            rgroups = list(self.df.Mol)

        if len(self._scaffolds) == 0:
            print("Please add scaffolds to the system for the evaluation. ")
        elif len(self._scaffolds) > 1:
            raise NotImplementedError("For now we only allow working with one scaffold. ")

        # carry out the full pipeline, generate conformers, etc.
        # note that we have to find a way to pass all the molecules
        # this means creating a pipeline of tasks for Dask,
        # a pipeline that is well conditional

        ## GENERATE CONFORMERS
        num_conf = dask.delayed(num_conf)
        minimum_conf_rms = dask.delayed(minimum_conf_rms)
        protein = dask.delayed(prody_package.parsePDB(self._protein_filename))
        protein_file = dask.delayed(self._protein_filename)
        min_dst_allowed = dask.delayed(min_dst_allowed)
        RMol._check_download_gnina()
        gnina_path = dask.delayed(os.path.join(RMol.gnina_dir, 'gnina'))

        # functions
        delayed_generate_conformers = dask.delayed(generate_conformers)
        delayed_remove_clashing_confs = dask.delayed(RMol.remove_clashing_confs)
        delayed_gnina = dask.delayed(gnina)

        # create dask jobs
        jobs = {}
        for i, row in self.df.iterrows():
            generated_confs = delayed_generate_conformers(row.Mol, num_conf, minimum_conf_rms)
            removed_clashes = delayed_remove_clashing_confs(generated_confs, protein,
                                                                                 min_dst_allowed=min_dst_allowed)
            jobs[i] = delayed_gnina(removed_clashes, protein_file, gnina_path)

        # run all jobs
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # gather the results
        for i, result in results.items():
            mol, cnnaffinities = result.result()

            # extract the conformers
            input_mol = self.df.Mol[i]
            input_mol.RemoveAllConformers()
            [input_mol.AddConformer(c) for c in mol.GetConformers()]

            # save the affinities so that one can retrace which conformer has the best energy
            input_mol.SetProp("cnnaffinities", str(cnnaffinities))

            self.df.score[i] = max(cnnaffinities)

        logger.info(f"Evaluated {len(results)} cases")

    def evaluate(self,
                 indices : Union[Sequence[int], pandas.DataFrame]=None,
                 scoring_function=None,
                 gnina_path=None,
                 gnina_gpu=False,
                 num_conf=50,
                 minimum_conf_rms=0.5,
                 penalty=pd.NA,
                 al_ignore_penalty=True,
                 **kwargs):
        """

        :param indices:
        :param scoring_function:
        :param gnina_path:
        :param gnina_gpu:
        :param num_conf:
        :param minimum_conf_rms:
        :param penalty:
        :param al_ignore_penalty:
        :param kwargs:
        :return:
        """

        # evaluate all molecules if no indices are picked
        if indices is None:
            indices = slice(None)

        if isinstance(indices, pandas.DataFrame):
            indices = indices.index

        selected_rows = self.df.loc[indices]

        # discard computed rows
        selected_rows = selected_rows[selected_rows.score.isna()]

        if len(self._scaffolds) == 0:
            print("Please add scaffolds to the system for the evaluation. ")
        elif len(self._scaffolds) > 1:
            raise NotImplementedError("For now we only allow working with one scaffold. ")

        # should be enough to do it once, shared
        ## GENERATE CONFORMERS

        if gnina_path is not None:
            # gnina_path = delayed(os.path.join(RMol.gnina_dir, 'gnina'))
            RMol.set_gnina(os.path.join(RMol.gnina_dir, 'gnina'))
        RMol._check_download_gnina()

        num_conf = dask.delayed(num_conf)
        minimum_conf_rms = dask.delayed(minimum_conf_rms)
        protein_file = dask.delayed(self._protein_filename)
        RMol._check_download_gnina()

        scaffold = dask.delayed(self._scaffolds[0])
        # extract which hydrogen was used for the attachement
        h_attachements = [a.GetIdx() for a in self._scaffolds[0].GetAtoms() if a.GetAtomicNum() == 0]

        h_attachement_index = None
        if len(h_attachements) > 0:
            h_attachement_index = h_attachements[0]

        # create dask jobs
        delayed_evaluate = dask.delayed(_evaluate_atomic)
        jobs = {}
        for i, row in selected_rows.iterrows():
            jobs[i] = delayed_evaluate(scaffold,
                                        row.Smiles,
                                        protein_file,
                                        h=h_attachement_index,
                                        num_conf=num_conf,
                                        minimum_conf_rms=minimum_conf_rms,
                                        scoring_function=scoring_function,
                                        gnina_gpu=gnina_gpu,
                                        **kwargs
                                        )

        # run all
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # gather the results
        for i, result in results.items():
            Training = True
            build_succeeded = True

            try:
                mol, data = result.result()

                # save all data generated
                for k, v in data.items():
                    mol.SetProp(k, str(v))

                # replace the original molecule with the new one
                self.df.at[i, "Mol"] = mol

                # extract the score
                score = data["score"]
            except subprocess.CalledProcessError as E:
                logger.error("Failed Process", E, E.cmd, E.output, E.stdout, E.stderr)
                score = penalty
                build_succeeded = False

                if al_ignore_penalty:
                    Training = False
            except Exception as E:
                # failed to finish the protocol, set the penalty
                score = penalty
                build_succeeded = False

                if al_ignore_penalty:
                    Training = False

            self.df.loc[i, ["score", "Training", "Success"]] = score, Training, build_succeeded

        logger.info(f"Evaluated {len(results)} cases")
        return self.df.loc[indices]

    def umap(self, filename="umap_out.html"):
        print("Please cite UMAP (umap-learn pacakge) if you're using it: https://arxiv.org/abs/1802.03426 , "
              "https://umap-learn.readthedocs.io/en/latest/index.html")

        from umap import UMAP
        fps = self.compute_fps(tuple(self.df.Smiles))
        # convert to a list of 1D arrays
        fps = [fp.flatten() for fp in np.split(fps, len(fps))]
        def tanimoto_dist(a, b):
            dotprod = np.dot(a, b)
            tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
            return 1.0 - tc

        reducer = UMAP(metric=tanimoto_dist)
        res = reducer.fit_transform(fps)

        df = self.df.copy()
        df["x"] = res[:, 0]
        df["y"] = res[:, 1]

        # Bokeh visualization
        # remove columns that bokeh cannot work with (and which are not needed)
        # df = df.drop(columns=["ROMol", "fp"])

        # Tooltip for hover functionality
        TOOLTIP = """
            <div>
                @svg{safe}
                sf1 Value: @sf1_values<br>
                cycle: @cycle <br>
                cnnaff: @cnnaffinity <br>
                plip: @plip <br>
                enamine id: @enamine_id <br>
                al exp: @run <br>
            </div>
            """

        # make the circles smaller for the noise (==0) in the cluster
        # df["sizes"] = [2 if c == 0 else 10 for c in picked_df.cluster]

        from bokeh.plotting import figure, output_file, show
        fig = figure(width=1000, height=500, # tooltips=TOOLTIP
                   title="UMAP Projection of Molecular Fingerprints")

        # colors = df["cluster"].astype('float').values
        from bokeh import palettes
        from bokeh.transform import linear_cmap
        # mapper = linear_cmap(field_name='cluster', palette=palettes.Turbo256, low=0, high=20)

        fig.circle('x', 'y', source=df, alpha=0.9)

        # Create a color bar based on sf1 values
        # color_mapper = LinearColorMapper(palette=Viridis256, low=min(colors), high=max(colors))
        # color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title='sf1 Value')

        # Add the color bar to the plot
        # p.add_layout(color_bar, 'right')
        output_file(filename=filename)
        show(fig)
        # import matplotlib.pyplot as plt
        # plt.show()


    def add_enamine_molecules(self, n_best=1, results_per_search=100, remove_scaffold_h=False):
        """
        For the best scoring molecules, find similar molecules in Enamine REAL database
         and add them to the dataset.

        Make sure you have the permission/license to use https://sw.docking.org/search.html
            this way.

        @scaffold: The scaffold molecule that has to be present in the found molecules.
            If None, this requirement will be ignored.
        @molecules_per_smile: How many top results (molecules) per Smiles searched.
        """

        from pydockingorg import Enamine

        if len(self._scaffolds) > 1:
            raise NotImplementedError("Only one scaffold is supported atm.")

        scaffold = self._scaffolds[0]

        # get the best performing molecules
        vl = self.df.sort_values(by="score", ascending=False)
        best_vl_for_searching = vl[:n_best]

        # nothing to search for yet
        if len(best_vl_for_searching[~best_vl_for_searching.score.isna()]) == 0:
            return

        if len(set(best_vl_for_searching.h)) > 1:
            raise NotImplementedError('Multiple growth vectors are used. ')

        # filter out previously queried molecules
        new_searches = best_vl_for_searching[best_vl_for_searching.enamine_searched == False]
        smiles_to_search = list(new_searches.Smiles)

        start = time.time()
        print(f'Querying Enamine REAL. Looking up {len(smiles_to_search)} smiles.')
        try:
            with Enamine() as DB:
                results: pandas.DataFrame = DB.search_smiles(smiles_to_search, remove_duplicates=True,
                                                             results_per_search=results_per_search)
        except requests.exceptions.HTTPError as HTTPError:
            print("Enamine API call failed. ", HTTPError)
            return
        print(f"Enamine returned with {len(results)} rows in {time.time() - start:.1f}s.")

        # update the database that this molecule has been searched
        self.df.loc[new_searches.index, 'enamine_searched'] = True

        if len(results) == 0:
            print("The server did not return a single Smiles!")
            return

        # prepare the scaffold for testing its presence
        # specifically, the hydrogen was replaced and has to be removed
        # for now we assume we only are growing one vector at a time - fixme
        if remove_scaffold_h:
            scaffold_noh = Chem.EditableMol(scaffold)
            scaffold_noh.RemoveAtom(int(best_vl_for_searching.iloc[0].h))
            scaffold = scaffold_noh.GetMol()

        dask_scaffold = dask.delayed(scaffold)

        start = time.time()
        # protonate and check for scaffold
        delayed_protonations = [DaskTasks.obabel_protonate(smi.rsplit(maxsplit=1)[0])
                                for smi in results.hitSmiles.values]
        jobs = self.dask_client.compute([DaskTasks.scaffold_check(smih, dask_scaffold)
                                         for smih in delayed_protonations])
        scaffold_test_results = [job.result() for job in jobs]
        scaffold_mask = [r[0] for r in scaffold_test_results]
        # smiles None means that the molecule did not have our scaffold
        protonated_smiles = [r[1] for r in scaffold_test_results if r[1] is not None]
        print(f"Dask obabel protonation + scaffold test finished in {time.time() - start:.2f}s.")
        print(f"Tested scaffold presence. Kept {sum(scaffold_mask)}/{len(scaffold_mask)}.")

        if len(scaffold_mask) > 0:
            similar = results[scaffold_mask]
            similar.hitSmiles = protonated_smiles
        else:
            similar = pandas.DataFrame(columns=results.columns)

        # filter out Enamine molecules which were previously added
        new_enamines = similar[~similar.id.isin(vl.enamine_id)]

        warnings.warn(f"Only one H vector is assumed and used. Picking {vl.h[0]} hydrogen on the scaffold. ")
        new_data = {
            'Smiles': list(new_enamines.hitSmiles.values),
            'h': vl.h[0], # fixme: for now assume that only one vector is used
            'enamine_id': list(new_enamines.id.values)
        }

        print("Adding: ", len(new_enamines.hitSmiles.values))
        return self.add_data(new_data)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, query):
        self._query = query

        if 'fegrow_label' in query.keywords:
            self._query_label = query.keywords.pop('fegrow_label')

    @property
    def query_label(self):
        return self._query_label

    def active_learning(self,
                        n=1,
                        first_random=True,
                        score_higher_better=True,
                        model=None,
                        query=None,
                        learner_type=None,
                        ):
        """
        Model the data using the Training subset. Then use the active learning query method.

        See properties "model" and "query" for finer control.

        It's better to save the FPs in the dataframe. Or in the underlying system.
        :return:
        """

        training = self.df[self.df.Training]
        selection = self.df[~self.df.Training]

        if training.empty:
            if first_random:
                warnings.warn("Selecting randomly the first samples to be studied (no score data yet). ")
                return selection.sample(n)
            else:
                raise ValueError("There is no scores for active learning. Please use the \"first_random\" property. ")

        # get the scored subset
        # fixme - multitarget?
        train_targets = training["score"].to_numpy(dtype=float)

        library_features = self.compute_fps(tuple(self.df.Smiles))
        train_features = library_features[training.index]
        selection_features = library_features[selection.index]

        import fegrow.al

        if model is not None:
            self.model = model
        if self.model is None:
            self.model = fegrow.al.Model.gaussian_process()

        if query is not None:
            self.query = query

        # employ Greedy query by default
        if self.query is None:
            self.query = fegrow.al.Query.Greedy()

        # update on how many to querry
        query = functools.partial(self.query, n_instances=n)

        target_multiplier = 1
        if score_higher_better is True:
            target_multiplier = -1

        if self.query_label in ['greedy', 'thompson', 'EI', 'PI']:
            target_multiplier *= 1
        elif self.query_label == 'UCB':
            target_multiplier *= -1

        train_targets = train_targets * target_multiplier

        # only GP uses Bayesian Optimizer
        if learner_type is not None:
            learner = learner_type(
                estimator=self.model,
                X_training=train_features,
                y_training=train_targets,
                query_strategy=query)
        elif isinstance(self.model, gaussian_process.GaussianProcessRegressor):
            learner = modAL.models.BayesianOptimizer(
                estimator=self.model,
                X_training=train_features,
                y_training=train_targets,
                query_strategy=query)
        else:
            learner = modAL.models.ActiveLearner(
                estimator=self.model,
                X_training=train_features,
                y_training=train_targets,
                query_strategy=query)

        inference = learner.predict(library_features) * target_multiplier

        self.df['regression'] = inference.T.tolist()

        selection_idx, _ = learner.query(selection_features)

        return selection.iloc[selection_idx]

    @staticmethod
    def _compute_fp_from_smiles(smiles, radius=3, size=2048):
        mol = Chem.MolFromSmiles(smiles)
        return np.array(Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=size))

    @functools.cache
    def compute_fps(self, smiles_tuple):
        """
        :param smiles_tuple: It has to be a tuple to be hashable (to work with caching).
        :return:
        """
        futures = self._dask_client.map(ChemSpace._compute_fp_from_smiles, smiles_tuple)
        fps = np.array([r.result() for r in futures])

        return fps

    def __str__(self):
        return f"Chemical Space with {len(self._dataframe)} smiles and {len(self._scaffolds)} scaffolds. "

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, item):
        return self.df.loc[item].Mol

    def toxicity(self):
        # return the toxicity of all of them
        toxicities = []
        for i, row in self.df.iterrows():
            toxicity = row.Mol.toxicity()

            # set the index to map the molecules back to the main dataframe
            toxicity.index = [i]

            toxicities.append(toxicity)

        return pandas.concat(toxicities)

    def to_sdf(self, filename, failed=False, unbuilt=True):
        """
        Write every molecule and all its fields as properties, to an SDF file.

        :return:
        """
        with Chem.SDWriter(filename) as SD:
            columns = self.df.columns.to_list()
            columns.remove("Mol")

            for i, row in self.df.iterrows():

                # ignore this molecule because it failed during the build
                if failed is False and row.Success is False:
                    continue

                # ignore this molecule because it was not built yet
                if unbuilt is False and row.Success is pandas.NA:
                    continue

                mol = row.Mol
                mol.SetIntProp("index", i)
                for column in columns:
                    value = getattr(row, column)
                    mol.SetProp(column, str(value))

                mol.ClearProp("attachement_point")
                SD.write(mol)

    @staticmethod
    def from_sdf(filename):
        items = []
        keys = {}
        for mol in Chem.SDMolSupplier(filename):
            props = mol.GetPropsAsDict()
            props['Mol'] = mol
            items.append(props)
            keys = set(props.keys()).union(keys)

        # convert into {key: list, key: list}
        data = {key: [] for key in keys}
        for item in items:
            for key in keys:
                if key in item:
                    data[key].append(item[key])
                else:
                    # some values might be missing
                    # for example, "cnnaffinities" are not saved when an error occurs
                    # in that case the value will be missing and the pentalty is assigned
                    data[key].append(pandas.NA)

        indices = data.pop('index')

        defaults = ChemSpace.DATAFRAME_DEFAULT_VALUES.copy()
        defaults.update(data)
        return ChemSpace(data=defaults, data_indices=indices)

    @property
    def df(self):
        return self._dataframe

    @df.setter
    def df(self, df):
        self._dataframe = df

    def add_protein(self, protein_filename):
        self._protein_filename = protein_filename

    def _ipython_display_(self):
        from IPython.display import display_html

        # exclude the 3D structures
        df = self.df.loc[:, self.df.columns != 'Mol']
        Chem.PandasTools.AddMoleculeColumnToFrame(df, smilesCol="Smiles", molCol="2D")
        return display_html(df)

class RGroups(pandas.DataFrame):
    """
    The default R-Group library with visualisation (mols2grid).
    """

    def __init__(self):
        data = RGroups._load_data()
        super().__init__(data)

        self._fegrow_grid = mols2grid.MolGrid(self, removeHs=True, mol_col="Mol", use_coords=False, name="m2")

    @staticmethod
    def _load_data() -> pandas.DataFrame:
        """
        Load the default R-Group library

        The R-groups were largely extracted from (please cite accordingly):
        Takeuchi, Kosuke, Ryo Kunimoto, and Jürgen Bajorath. "R-group replacement database for medicinal chemistry." Future Science OA 7.8 (2021): FSO742.
        """
        molecules = []
        names = []

        builtin_rgroups = Path(__file__).parent / "data" / "rgroups" / "library.sdf"
        for rgroup in Chem.SDMolSupplier(str(builtin_rgroups), removeHs=False):
            molecules.append(rgroup)
            names.append(rgroup.GetProp("SMILES"))

            # highlight the attachment atom
            for atom in rgroup.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    setattr(rgroup, "__sssAtoms", [atom.GetIdx()])

        return {"Mol": molecules, "Name": names}

    def _ipython_display_(self):
        from IPython.display import display_html

        subset = ["img", "Name", "mols2grid-id"]
        display_html(self._fegrow_grid.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        df = self._fegrow_grid.get_selection()
        return list(df["Mol"])


class Linkers(pandas.DataFrame):
    """
    A linker library presented as a grid molecules using mols2grid library.
    """

    def __init__(self):
        # initialise self dataframe
        data = Linkers._load_data()
        super().__init__(data)

        self._fegrow_grid = mols2grid.MolGrid(
            self,
            removeHs=True,
            mol_col="Mol",
            use_coords=False,
            name="m1",
            prerender=False,
        )

    @staticmethod
    def _load_data():
        # note that the linkers are pre-sorted so that:
        #  - [R1]C[R2] is next to [R2]C[R1]
        #  - according to how common they are (See the original publication) as described with SmileIndex
        builtin_rlinkers = Path(__file__).parent / "data" / "linkers" / "library.sdf"

        mols = []
        display_names = []
        smile_indices = []
        for mol in Chem.SDMolSupplier(str(builtin_rlinkers), removeHs=False):
            mols.append(mol)

            # use easier searchable SMILES, e.g. [*:1] was replaced with R1
            display_names.append(mol.GetProp("display_smiles"))

            # extract the index property from the original publication
            smile_indices.append(mol.GetIntProp("SmileIndex"))

        return {"Mol": mols, "Name": display_names, "Common": smile_indices}

    def _ipython_display_(self):
        from IPython.display import display

        subset = ["img", "Name", "mols2grid-id"]
        return display(self._fegrow_grid.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        df = self._fegrow_grid.get_selection()
        return list(df["Mol"])


def gnina(mol, receptor, gnina_path, gnina_gpu=False):

    extras = []
    if gnina_gpu is False:
        extras.append("--no_gpu")

    # make a temporary sdf file for gnina
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf") as TMP_SDF:
        with Chem.SDWriter(TMP_SDF.name) as w:
            for conformer in mol.GetConformers():
                w.write(mol, confId=conformer.GetId())

        # run the code on the sdf
        process = subprocess.run(
            [
                gnina_path,
                "--score_only",
                "-l",
                TMP_SDF.name,
                "-r",
                receptor,
                "--seed",
                "0",
                "--stripH",
                "False",
            ] + extras
            ,
            capture_output=True,
            check=True,
        )

    output = process.stdout.decode("utf-8")
    CNNaffinities_str = re.findall(r"CNNaffinity: (-?\d+.\d+)", output)

    # convert to floats
    CNNaffinities = list(map(float, CNNaffinities_str))

    return mol, CNNaffinities

def build_molecules(*args, **kwargs):
    raise NotImplementedError("This function was removed. "
                              "Please use the new simple \"build_molecule\" instead, "
                              "which now does not work with lists. ")

def build_molecule(
    scaffolds: Chem.Mol,
    r_group: Union[Chem.Mol, str],
    scaffold_point: Optional[int] = None,
    rgroup_point: Optional[int] = None,
    keep: Optional[int] = None,
):
    """

    :param scaffolds:
    :param r_groups:
    :param scaffold_point: attachement point on the scaffold
    :param keep: When the scaffold is grown from an internal atom that divides the molecules into separate
        submolecules, keep the submolecule with this atom index.
    :return:
    """

    if isinstance(r_group, list) and len(r_group) == 0:
        raise ValueError("Empty list received. Please pass any R-groups or R-linkers. ")

    if isinstance(scaffold_point, list) or isinstance(scaffolds, list):
        raise ValueError("Only one scaffold and rgroup at at time is permitted. ")

    # scaffolds were created earlier, they are most likely templates combined with linkers,
    if isinstance(scaffolds, ChemSpace):
        # fixme - these should become "the cores", it's simple with one mol, and tricky with more of them,
        scaffolds = [mol for idx, mol in scaffolds.dataframe.Mol.items()]

    # convert smiles into a molecule
    if isinstance(r_group, str):
        if '*' not in r_group and rgroup_point is None:
            raise ValueError("The SMILES used for the R-Group has to have an R-group atom. "
                             "That is the character * in Smiles, or you can use the RDKit function .SetAtomicNum(0) ")
        params = Chem.SmilesParserParams()
        params.removeHs = False
        r_group = Chem.MolFromSmiles(r_group, params=params)

        # set the attachement point on the R-group
        if rgroup_point is not None:
            r_group.GetAtomWithIdx(rgroup_point).SetAtomicNum(0)

    built_mols = build_molecules_with_rdkit(
        scaffolds, r_group, scaffold_point, keep
    )

    mol, scaffold, scaffold_no_attachement = built_mols
    rmol = RMol(mol)

    if hasattr(scaffold, 'template') and isinstance(scaffold.template, rdkit.Chem.Mol):
        # save the original scaffold (e.g. before the linker was added)
        # this means that conformer generation will always have to regenerate the previously added R-groups/linkers
        rmol._save_template(scaffold.template)
    else:
        rmol._save_template(scaffold_no_attachement)

    return rmol


def _evaluate_atomic(scaffold,
                     smiles,
                     pdb_filename,
                     h=None,
                     scoring_function=None,
                     num_conf=50,
                     minimum_conf_rms=0.5,
                     ani=True,
                     platform="CPU",
                     gnina_gpu=False,
                     skip_optimisation=False,
                     full_evaluation=None
                     ):
    """

    :param scaffold:
    :param h:
    :param smiles: Full Smiles.
    :param scoring_function:
    :param pdb_filename:
    :param gnina_path:
    :return:
    """

    if full_evaluation is not None:
        return full_evaluation(scaffold,
                     h,
                     smiles,
                     pdb_filename,
                     scoring_function=None,
                     num_conf=50,
                     minimum_conf_rms=0.5,
                     ani=ani,
                     platform="CPU",
                     skip_optimisation=False)

    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    rmol = RMol(Chem.MolFromSmiles(smiles, params=params))

    # remove the h
    # this is to help the rdkit's HasSubstructMatch
    if h is not None:
        scaffold = copy.deepcopy(scaffold)
        scaffold_m = Chem.EditableMol(scaffold)
        scaffold_m.RemoveAtom(int(h))
        scaffold = scaffold_m.GetMol()

    rmol._save_template(scaffold)

    rmol.generate_conformers(num_conf=num_conf, minimum_conf_rms=minimum_conf_rms)
    rmol.remove_clashing_confs(pdb_filename)
    if not skip_optimisation:
        rmol.optimise_in_receptor(
            receptor_file=pdb_filename,
            ligand_force_field="openff",
            use_ani=ani,
            sigma_scale_factor=0.8,
            relative_permittivity=4,
            water_model=None,
            platform_name=platform,
        )

        if rmol.GetNumConformers() == 0:
            raise Exception("No Conformers")

        rmol.sort_conformers(energy_range=2)  # kcal/mol

    data = {}
    if scoring_function is None:
        cnnaffinities = rmol.gnina(receptor_file=pdb_filename, gnina_gpu=gnina_gpu)
        data = {"cnnaffinities": [float(affinity) for affinity in cnnaffinities.CNNaffinity]}
        score = data["cnnaffinities"][0]
    else:
        score = scoring_function(rmol, pdb_filename, data)

    data["score"] = score
    return rmol, data
