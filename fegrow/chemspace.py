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

from fegrow.builder import build_molecules_with_rdkit
from fegrow.conformers import generate_conformers
from fegrow.receptor import optimise_in_receptor, sort_conformers
from fegrow.toxicity import tox_props

# default options
pandas.set_option("display.precision", 3)

logger = logging.getLogger(__name__)

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

class Executor:

    @ABC.abstractmethod
    def execute(self, func, *args, **kwargs):
        ...

class SequentialExecutor(Executor):

    def execute(self, df, fn, *args, **kwargs):
        # run one by one
        ...


class DaskExecutor(Executor):

    def __init__(self, client, cluster):
        # this might be even a better place to create the cluster
        # should it be shared across the "classes"?
        self._dask_cluster = dask_cluster

        if ChemSpace._dask_cluster is None:
            logger.info("No Dask cluster configured. Creating a local cluster of threads. ")
            warnings.warn("ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. "
                          "Use a Dask cluster with processes as a work around "
                          "(see the documentation for an example of this workaround) .")

            kwargs = {"n_workers": None,
                      "processes": False,  # turn off Nanny to avoid the problem
                      # with loading of the main file (ie re-executing it)
                      # creating an infinite loop
                      "dashboard_address": ":8989",
                      **dask_local_cluster_kwargs
                      }
            self._dask_cluster = LocalCluster(**kwargs)

        self._dask_client = Client(self._dask_cluster)
        print(f"Dask can be watched on {self._dask_client.dashboard_link}")

    def execute(self, df, fn, *args, **kwargs):
        # daskify
        d_fn = dask.delayed(fn)

        # prepare the dask args
        d_args = [dask.delayed(arg) for arg in args]

        # assemble the dask jobs
        jobs = {}
        for i, row in self.df.iterrows():
            jobs[delayed_generate_conformers(row.Mol, num_conf, minimum_conf_rms, **kwargs)] = (i, row)

        # batch compute
        self.dask_client.compute(list(jobs.keys()))

        # gather results
        results = {}
        for job, (i, row) in jobs.items():
            results[i] = job.result()

        return results

class ChemSpace: # RInterface
    """
    Streamline working with many RMols or a specific chemical space by employing a pandas dataframe,
    in combination with Dask for parallellisation.
    """

    def rep2D(self, subImgSize=(400, 400), **kwargs):
        return Draw.MolsToGridImage(
            [row.Mol.rep2D(rdkit_mol=True, **kwargs) for i, row in self.df.iterrows()], subImgSize=subImgSize
        )

    DATAFRAME_DEFAULT_VALUES = {"Smiles": [], # Smiles will always be replaced when inserting new data.
                                "Mol": pandas.NA,
                                "score": pandas.NA, # any scoring function, by default cnnaffinity
                                "h": pandas.NA,
                                "Training": False, # will be used for AL
                                "Success": pandas.NA, # true if built, false if built unsuccessfully
                                "enamine_searched": False,
                                "enamine_id": pandas.NA}

    executor = None

    def __init__(self, data=None,
                 data_indices=None,
                 force_sequential=True,
                 dask_cluster=None):
        if data is None:
            data = ChemSpace.DATAFRAME_DEFAULT_VALUES

        self.df = pandas.DataFrame(data, index=data_indices)

        self._scaffolds = []
        self._model = None
        self._query = None
        self._query_label = None

        if force_sequential:
            self.executor = SequentialExecutor()
        else:
            self.executor = DaskExecutor(dask_cluster, dask_client)

    def set_dask_caching(self, bytes_num=4e9):
        # use max 4 gigabytes of memory
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
        # give dataset,
        results = self.executor.execute(self.df, generate_conformers, num_conf, minimum_conf_rms, **kwargs)

        # extract results
        for i, new_mol in results.items():
            # fetch the original input mol
            mol = self.df[i].mol
            mol.RemoveAllConformers()

            # take the generated conformers
            for c in new_mol.GetConformers():
                mol.AddConformer(c, assignId=True)

    def GetNumConformers(self):
        return [rmol.GetNumConformers() for rmol in self]

    def remove_clashing_confs(self, prot, min_dst_allowed=1):

        results = self.executor.execute(self.df, RMol.remove_clashing_confs,
                                        prot, min_dst_allowed, **kwargs)

        # extract results
        for i, unclashy_mol in results.items():
            mol.RemoveAllConformers()
            for c in unclashy_mol.GetConformers():
                mol.AddConformer(c)

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Return lists of energies.
        """

        results = self.executor.execute(self.df, optimise_in_receptor,
                                        *args, **kwargs)

        jobs = {}
        for i, row in self.df.iterrows():
            if row.Mol.GetNumConformers() == 0:
                print(f"Warning: mol {i} has no conformers. Ignoring receptor optimisation.")
                continue

            jobs[row.Mol] = delayed_optimise_in_receptor(row.Mol, *args, **kwargs)

        # extract results
        dfs = []
        for i, (opt_mol, energies) in results.items():
            opt_mol, energies = result.result()
            mol.RemoveAllConformers()
            # replace the conformers with the optimised ones
            for c in opt_mol.GetConformers():
                mol.AddConformer(c)

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
        RMol._check_download_gnina()

        results = self.executor.execute(self.df, gnina,
                                        receptor_file, **kwargs)

        gnina_path = dask.delayed(os.path.join(RMol.gnina_dir, 'gnina'))

        # create the dask jobs
        delayed_gnina = dask.delayed()
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

        # prepare the scaffold for tests its presence
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

