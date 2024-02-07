import pathlib

import click
from typing import Optional


@click.command()
@click.option(
    "-c",
    "--core-ligand",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="The path to the SDF file of the core ligand which will be used to restrain the geometries of the scored ligands this should be a substructure of the ligands to be scored.",
)
@click.option(
    "-l",
    "--ligands",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="The path to the ligands to be scored can be in any supported format by RDKit such as csv. smiles or SDF.",
)
@click.option(
    "-r",
    "--receptor",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="The path of the receptor PDB file, this should only contain the receptor and not the reference ligand.",
)
@click.option(
    "-s",
    "--settings",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="The path of the settings file which configures the scoring run.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=True, path_type=pathlib.Path),
    help="The name of the output folder.",
)
@click.option(
    "-g",
    "--gnina-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True, resolve_path=True, path_type=pathlib.Path),
    help="The path to the gnina executable which will override the settings.",
)
@click.option(
    '-t',
    '--threads',
    default=1,
    show_default=True,
    type=click.INT,
    help='The number of threads per worker.'
)
@click.option(
    '-w',
    '--workers',
    default=4,
    show_default=True,
    type=click.INT,
    help='The number of workers to use.'

)
def score(
    core_ligand: pathlib.Path,
    ligands: pathlib.Path,
    receptor: pathlib.Path,
    output: pathlib.Path,
    settings: Optional[pathlib.Path] = None,
    gnina_path: Optional[pathlib.Path] = None,
    threads: int = 1,
    workers: int = 4,
):
    """
    Score the list of input ligands using Gnina after optimising in the receptor. Tasks are distributed using Dask.
    """
    from dask.distributed import Client
    from rdkit import Chem
    import traceback
    import dask
    import tqdm
    import warnings
    import os
    import logging
    from fegrow.cli.utils import score_ligand, Settings, load_target_ligands
    from fegrow.package import RMol
    from dask.distributed import LocalCluster
    import time


    # hide warnings and logs from openff-toolkit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        logging.getLogger('fegrow').setLevel(logging.ERROR)

        client = Client(LocalCluster(threads_per_worker=threads, n_workers=workers))
        # create the cluster
        click.echo(f"Client created {client}")

        if settings is not None:
            config = Settings.parse_file(settings)
            if gnina_path is not None:
                config.gnina_path = gnina_path.as_posix()
        else:
            # build the base settings object this needs the gnina path
            config = Settings(gnina_path=gnina_path.as_posix())
        # set the gnina directory
        RMol.set_gnina(loc=config.gnina_path)
        click.echo(f'Setting Gnina path to: {config.gnina_path}')

        click.echo(f"Loading core ligand from {core_ligand}")
        # we remove all Hs rather than specific ones at the attachment point
        core = Chem.MolFromMolFile(core_ligand.as_posix(), removeHs=True)
        core_dask = dask.delayed(core)

        # load the target ligands
        target_smiles = load_target_ligands(ligand_file=ligands)

        # build a list of tasks to submit
        for_submission = [
            score_ligand(
                core_ligand=core_dask,
                target_smiles=smiles,
                receptor=receptor,
                settings=config,
            )
            for smiles in target_smiles
        ]

        submitted = client.compute(for_submission)

        output.mkdir(exist_ok=True)

        molecule_output = Chem.SDWriter(output.joinpath("scored_molecules.sdf").as_posix())
        with tqdm.tqdm(total=len(submitted), desc="Scoring molecules...", ncols=80) as pbar:
            while len(submitted) > 0:
                to_remove = []
                for job in submitted:
                    if not job.done():
                        continue

                    # remove the job
                    to_remove.append(job)
                    pbar.update(1)

                    try:
                        mol_data = job.result()
                        # if the molecule could not be scored as it had no conformers do not save the geometry
                        if mol_data['cnnaffinity'] == 0:
                            continue

                        rmol = mol_data.pop("molecule")
                        # recover the properties (they are not passed with serialisation)
                        [rmol.SetProp(k, str(v)) for k, v in mol_data.items()]
                        # write the molecule out when they complete incase we crash
                        molecule_output.write(rmol)
                    except Exception as E:
                        print("error for molecule")
                        traceback.print_exc()

                # remove jobs before next round
                for job in to_remove:
                    submitted.remove(job)
                time.sleep(5)

        click.echo("All molecules scored")
