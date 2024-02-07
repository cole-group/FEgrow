import click
import pathlib

@click.command()
@click.option(
    '-l',
    '--ligands',
    help='The sdf file which contains the scored ligands to create the report for.',
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path, exists=True),
    multiple=True
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=True, path_type=pathlib.Path),
    help="The name of the output folder.",
)
@click.option(
    '-t',
    '--title',
    type=click.STRING,
    help='The title of the dataset which will be used for the report and sdf file names'
)
def report(
        ligands: list[pathlib.Path],
        output: pathlib.Path,
        title: str
):
    """
    Generate an interactive HTML report of the molecules from the scored SDF files and an SDF file ordered by the top
    performing molecules.
    """
    from rdkit import Chem
    from fegrow.cli.utils import draw_mol
    import pandas as pd
    import tqdm
    import panel
    import bokeh.models.widgets.tables

    output.mkdir(exist_ok=True, parents=True)
    # load all best scoring ligands
    molecules_and_affinities = []
    for file in ligands:
        # load the low energy ligands and the predicted affinity
        supplier = Chem.SDMolSupplier(file.as_posix(), removeHs=True)
        for molecule in supplier:
            molecules_and_affinities.append((molecule, float(molecule.GetProp('cnnaffinity'))))

    click.echo(f'Total number of molecules and predictions: {len(molecules_and_affinities)}')

    # sort all molecules by predicted affinity
    ranked_all_mols = sorted(molecules_and_affinities, key=lambda x: x[1])
    # write out all molecules to a single sdf sorted by affinity
    output_file = output.joinpath(f'{title}.sdf')
    with Chem.SDWriter(output_file.as_posix()) as sdf_out:
        rows = []
        for mol, affinity in tqdm.tqdm(ranked_all_mols, desc="Creating report...", ncols=80):
            try:
                smiles = mol.GetProp('smiles')
            except KeyError:
                smiles = Chem.MolToSmiles(mol)
                mol.SetProp('smiles', smiles)
            rows.append(
                {
                    "Smiles": smiles,
                    "Molecule": draw_mol(smiles),
                    "CNNaffinity": affinity,
                    "IC50 (nM)": mol.GetProp('cnnaffinityIC50').split()[0],
                }
            )
            sdf_out.write(mol)
    # create the report
    df = pd.DataFrame(rows)

    number_format = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")
    layout = panel.Column(
        panel.widgets.Tabulator(
            df,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters={"Smiles": "html", "Molecule": "html", "CNNaffinity": number_format,
                        "IC50 (nM)": number_format},
            configuration={"rowHeight": 400},
        ),
        sizing_mode="stretch_width",
        scroll=True
    )

    layout.save(output.joinpath(f'{title}.html').as_posix(), title=title, embed=True)
