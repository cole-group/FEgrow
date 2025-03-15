import pathlib
import time
from typing import Optional

import click


@click.command()
@click.option(
    "-c",
    "--core-ligand",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The path to the SDF file of the core ligand which will be used to restrain the geometries of the scored ligands this should be a substructure of the ligands to be scored.",
)
@click.option(
    "-l--ligands",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The path to the ligands to be scored can be in any supported format by RDKit such as csv. smiles or SDF.",
)
@click.option(
    "-r",
    "--receptor",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The path of the receptor PDB file, this should only contain the receptor and not the reference ligand.",
)
@click.option(
    "-s",
    "--settings",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="The path of the settings file which configures the scoring run.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=True),
    help="The name of the output folder.",
)
@click.option(
    "-g",
    "--gnina-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True),
    help="The path to the gnina executable which will override the settings.",
)
def score(
    core_ligand: pathlib.Path,
    ligands: pathlib.Path,
    receptor: pathlib.Path,
    output: pathlib.Path,
    settings: Optional[pathlib.Path] = None,
    gnina_path: Optional[pathlib.Path] = None,
):
    """
    Score the list of input ligands using Gnina after optimising in the receptor.
    """
    import traceback

    import dask
    import tqdm
    from dask.distributed import Client
    from rdkit import Chem

    from fegrow.cli.utils import Settings, load_target_ligands, score_ligand

    try:
        from mycluster import create_cluster
    except ImportError:

        def create_cluster():
            from dask.distributed import LocalCluster

            return LocalCluster()

    client = Client(create_cluster())
    # create the cluster
    click.echo(f"Client created {client}")

    if settings is not None:
        config = Settings.parse_file(settings)
        if gnina_path is not None:
            config.gnina_path = gnina_path
    else:
        # build the base settings object this needs the gnina path
        config = Settings(gnina_path=gnina_path)

    click.echo(f"Loading core ligand from {core_ligand}")
    # we remove all Hs rather than specific ones at the attachment point
    core = Chem.MolFromMolFile(core_ligand, removeHs=True)
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
    jobs = dict((job, i) for i, job in enumerate(submitted))

    output_path = pathlib.Path(output)
    output_path.mkdir(exist_ok=True)

    molecule_output = Chem.SDWriter(output_path.joinpath("scored_molecules.sdf"))
    with tqdm.tqdm(total=len(submitted), desc="Scoring molecules...", ncols=80) as pbar:
        while len(jobs) > 0:
            for job, index in jobs.items():
                if not job.done():
                    continue

                # remove the job
                del jobs[job]
                pbar.update(1)

                try:
                    mol_data = job.result()
                    rmol = mol_data.pop("molecule")
                    # recover the properties (they are not passed with serialisation)
                    [rmol.SetProp(k, str(v)) for k, v in mol_data.items()]
                    # write the molecule out when they complete incase we crash
                    molecule_output.write(rmol)
                except Exception:
                    print("error for index, ", index)
                    traceback.print_exc()

            time.sleep(5)

    click.echo("All molecules scored")
