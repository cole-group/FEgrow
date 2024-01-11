import abc
import copy
import functools
import logging
import os
import warnings
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import List, Optional, Union
import urllib

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
from dask import delayed
from dask.distributed import LocalCluster, Client, Scheduler, Worker

from .builder import build_molecules_with_rdkit
from .conformers import generate_conformers
from .receptor import optimise_in_receptor, sort_conformers
from .toxicity import tox_props

# default options
pandas.set_option("display.precision", 3)

logger = logging.getLogger(__name__)


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

    def rep2D(self, idx=-1, rdkit_mol=False, **kwargs):
        """
        Use RDKit and get a 2D diagram.
        Uses Compute2DCoords and Draw.MolToImage function

        Works with IPython Notebook.

        :param **kwargs: are passed further to Draw.MolToImage function.
        """
        numbered = copy.deepcopy(self)
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

    def gnina(self, receptor_file):
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

        _, CNNaffinities = gnina(self, receptor, gnina_path)

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


class ChemSpace: # RInterface
    """
    Streamline working with many RMols or a specific chemical space by employing a pandas dataframe,
    in combination with Dask for parallellisation.
    """

    def rep2D(self, subImgSize=(400, 400), **kwargs):
        return Draw.MolsToGridImage(
            [row.Mol.rep2D(rdkit_mol=True, **kwargs) for i, row in self.dataframe.iterrows()], subImgSize=subImgSize
        )

    # def __getitem__(self, item):
    #     """
    #     Provide list like behaviour that returns the molecule.
    #     ps. Mol column always has to be present.
    #     """
    #     return self.loc[item].Mol
    _dask_cluster = None
    _rmol_functions = {}

    def __init__(self, data={"Smiles": [], "Mol": [], "CNNAffinity": []}):
        self.dataframe = pandas.DataFrame(data)

        # PandasTools.ChangeMoleculeRendering(self)
        if ChemSpace._dask_cluster is None:
            print('None cluster')
            import asyncio
            # silence_logs=logging.DEBUG
            ChemSpace._dask_cluster = LocalCluster(n_workers=2,
                                                   processes=False, # turn off Nanny to avoid the problem
                                                   # with loading of the main file (ie executing it)
                                                   )
            # ChemSpace._dask_cluster = Scheduler()
            # ChemSpace._dask_cluster = LocalCluster(preload_nanny=["print('Hi Nanny')"],
            #                                        preload=["pint"], n_workers=1
            #                                        ) #asynchronous=True)
            ChemSpace._dask_client = Client(ChemSpace._dask_cluster) #ChemSpace._dask_cluster, asynchronous=True)

            # prepare the functions for dask
            for name, function in \
                    [("generate_conformers", generate_conformers),
                     ("remove_clashing_confs", RMol.remove_clashing_confs),
                     ("optimise_in_receptor", optimise_in_receptor),
                     ("gnina", gnina),
                     ("build_molecule", build_molecule)
                     ]:
                ChemSpace._rmol_functions[name] = delayed(function)

        # self._dask_client = None
        # self._dask_cluster = None

        #
        self._scaffolds = []

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
        num_conf = delayed(num_conf)
        minimum_conf_rms = delayed(minimum_conf_rms)

        # create the dask jobs
        jobs = {}
        for i, row in self.dataframe.iterrows():
            jobs[row.Mol] = (
                ChemSpace._rmol_functions['generate_conformers'](row.Mol, num_conf, minimum_conf_rms, **kwargs))

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
        prot = delayed(prot)
        min_dst_allowed = delayed(min_dst_allowed)

        # create the dask jobs
        jobs = {}
        for i, row in self.dataframe.iterrows():
            jobs[row.Mol] = ChemSpace._rmol_functions['remove_clashing_confs'](row.Mol, prot,
                                                                               min_dst_allowed=min_dst_allowed)

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        for mol, result in results.items():
            unclashed_mol = result.result()
            mol.RemoveAllConformers()
            [mol.AddConformer(c) for c in unclashed_mol.GetConformers()]

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Replace the current molecule with the optimised one. Return lists of energies.
        """

        # dfs = []
        # for i, rmol in enumerate(self):
        #     print(f"RMol index {i}")
        #     dfs.append(rmol.optimise_in_receptor(*args, **kwargs))
        #
        # df = pandas.concat(dfs)
        # df.set_index(["ID", "Conformer"], inplace=True)
        # return df

        # daskify objects
        args = [delayed(arg) for arg in args]
        kwargs = {k: delayed(v) for k, v in kwargs.items()}

        # create the dask jobs
        jobs = {}
        for i, row in self.dataframe.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(f"Warning: mol {i} has no conformers. Ignoring receptor optimisation.")
                continue

            jobs[row.Mol] = ChemSpace._rmol_functions['optimise_in_receptor'](row.Mol, *args, **kwargs)

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        for mol, result in results.items():
            opt_mol, energies = result.result()
            mol.RemoveAllConformers()
            # replace the conformers with the optimised ones
            [mol.AddConformer(c) for c in opt_mol.GetConformers()]

            mol._save_opt_energies(energies)

        # build a dataframe with the molecules
        # conformer_ids = [c.GetId() for c in self.GetConformers()]
        # df = pandas.DataFrame(
        #     {
        #         "ID": [self.id] * len(energies),
        #         "Conformer": conformer_ids,
        #         "Energy": energies,
        #     }
        # )

        # return df

    def sort_conformers(self, energy_range=5):
        dfs = []
        for i, row in self.dataframe.iterrows():
            print(f"RMol index {i}")
            dfs.append(row.Mol.sort_conformers(energy_range))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def gnina(self, receptor_file):
        # dfs = []
        # for i, rmol in enumerate(self):
        #     print(f"RMol index {i}")
        #     dfs.append(rmol.gnina(receptor_file))
        #
        # df = pandas.concat(dfs)
        # df.set_index(["ID", "Conformer"], inplace=True)
        # return df

        # daskify objects
        receptor_file = delayed(receptor_file)
        RMol._check_download_gnina()
        gnina_path = delayed(os.path.join(RMol.gnina_dir, 'gnina'))

        # create the dask jobs
        jobs = {}
        for i, row in self.dataframe.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(f"Warning: mol {i} has no conformers. Ignoring gnina.")
                continue

            jobs[i] = ChemSpace._rmol_functions['gnina'](row.Mol, receptor_file, gnina_path)

        # dask batch compute
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # extract results
        dfs = []
        for i, result in results.items():
            _, cnnaffinities = result.result()
            df = RMol._parse_gnina_cnnaffinities(self.dataframe.Mol[i], cnnaffinities, mol_id=i)
            dfs.append(df)


        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def discard_missing(self):
        """
        Remove from this list the molecules that have no conformers
        """
        ids_to_remove = []
        for i, row in self.dataframe.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(
                    f"Discarding a molecule (id {i}) due to the lack of conformers. "
                )
                ids_to_remove.append(i)

        self.dataframe = self.dataframe[~self.dataframe.index.isin(ids_to_remove)]
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
                raise ValueError("The template ")
        else:
            # mark the right atom for replacement by assigning it atomic number == 0
            template.GetAtomWithIdx(atom_replacement_index).SetAtomicNum(0)

        self._scaffolds.append(template)

    def add_rgroups(self, rgroups):
        """
        Note that if they are Smiles:
         - if they have an * atom (e.g. RDKit atom.SetAtomicNum(0)), this will be used for attachment to the scaffold
         - if they don't have an * atom, the scaffold will be fitted as a substructure

        # fixme - add support for smiles

        :param rgroups: A list of Smiles. Molecules will be accepted and converted to Smiles.
        :return:
        """

        # convert molecules into smiles
        # if isinstance(smi_list[0], Chem.Mol):
        #     smi_list = [Chem.MolToSmiles(mol) for mol in smi_list]

        scaffold = delayed(self._scaffolds[0])

        # create the dask jobs
        jobs = [ChemSpace._rmol_functions["build_molecule"](scaffold, rgroup) for rgroup in rgroups]
        results = self.dask_client.compute(jobs)
        built_mols = [r.result() for r in results]

        # get Smiles
        built_mols_smiles = [Chem.MolToSmiles(mol) for mol in built_mols]

        # update the internal dataframe
        rgroups = pandas.DataFrame({"Smiles": built_mols_smiles, "Mol": built_mols})
        self.dataframe = pandas.concat([self.dataframe, rgroups])

    def evaluate(self, indices=None, num_conf=10, minimum_conf_rms=0.5, min_dst_allowed=1):
        """
        Generate the conformers and score the subset of molecules.

        E.g.
        :param indices: The indices in the dataframe to be run through the pipeline.
            If None, all molecules are evaluated.
        :return:
        """

        if indices is None:
            # evaluate all molecules
            rgroups = list(self.dataframe.Mol)

        if len(self._scaffolds) == 0:
            print("Please add scaffolds to the system for the evaluation. ")
        elif len(self._scaffolds) > 1:
            raise NotImplementedError("For now we only allow working with one scaffold. ")

        # carry out the full pipeline, generate conformers, etc.
        # note that we have to find a way to pass all the molecules
        # this means creating a pipeline of tasks for Dask,
        # a pipeline that is well conditional

        ## GENERATE CONFORMERS
        # self.generate_conformers(num_conf=num_conf, minimum_conf_rms=minimum_conf_rms)
        num_conf = delayed(num_conf)
        minimum_conf_rms = delayed(minimum_conf_rms)
        protein = delayed(prody_package.parsePDB(self._protein_filename))
        protein_file = delayed(self._protein_filename)
        min_dst_allowed = delayed(min_dst_allowed)
        RMol._check_download_gnina()
        gnina_path = delayed(os.path.join(RMol.gnina_dir, 'gnina'))

        # create dask jobs
        jobs = {}
        for i, row in self.dataframe.iterrows():
            generated_confs = ChemSpace._rmol_functions['generate_conformers'](row.Mol, num_conf, minimum_conf_rms)
            removed_clashes = ChemSpace._rmol_functions['remove_clashing_confs'](generated_confs, protein,
                                                                               min_dst_allowed=min_dst_allowed)
            jobs[i] = ChemSpace._rmol_functions['gnina'](removed_clashes, protein_file, gnina_path)

        # run all jobs
        results = dict(zip(jobs.keys(), self.dask_client.compute(list(jobs.values()))))

        # gather the results
        for i, result in results.items():
            mol, cnnaffinities = result.result()

            # extract the conformers
            input_mol = self.dataframe.Mol[i]
            input_mol.RemoveAllConformers()
            [input_mol.AddConformer(c) for c in mol.GetConformers()]

            # save the affinities so that one can retrace which conformer has the best energy
            input_mol.SetProp("cnnaffinities", str(cnnaffinities))

            self.dataframe.CNNAffinity[i] = max(cnnaffinities)

        print(f"Evaluated {len(results)} cases")

    def __str__(self):
        return f"Chemical Space with {len(self.dataframe)} smiles and {len(self._scaffolds)} scaffolds. "

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.dataframe)

    @property
    def df(self):
        return self.dataframe

    def add_protein(self, protein_filename):
        self._protein_filename = protein_filename


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


def gnina(mol, receptor, gnina_path):
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
            ],
            capture_output=True,
            check=True,
        )

    output = process.stdout.decode("utf-8")
    CNNaffinities_str = re.findall(r"CNNaffinity: (-?\d+.\d+)", output)

    # convert to floats
    CNNaffinities = list(map(float, CNNaffinities_str))

    return mol, CNNaffinities


def build_molecule(
    scaffolds: Chem.Mol,
    r_group: Chem.Mol,
    attachment_point: Optional[int] = None,
    keep_component: Optional[int] = None,
):
    """

    :param scaffolds:
    :param r_groups:
    :param attachment_point:
    :param keep_component: When the scaffold is grown from an internal atom that divides the molecules into separate
        submolecules, keep the submolecule with this atom index.
    :return:
    """

    if isinstance(r_group, list) and len(r_group) == 0:
        raise ValueError("Empty list received. Please pass any R-groups or R-linkers. ")

    # scaffolds were created earlier, they are most likely templates combined with linkers,
    if isinstance(scaffolds, ChemSpace):
        # fixme - these should become "the cores", it's simple with one mol, and tricky with more of them,
        scaffolds = [mol for idx, mol in scaffolds.dataframe.Mol.items()]

    built_mols = build_molecules_with_rdkit(
        scaffolds, r_group, attachment_point, keep_component
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
