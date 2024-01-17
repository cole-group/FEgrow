import pathlib

import click
from fegrow.cli.scoring import score
from fegrow.cli.utils import Settings


@click.group()
def cli():
    pass


cli.add_command(score)


@cli.command()
@click.option(
    "-g",
    "--gnina-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True),
    help="The path to the gnina executable which will override the settings.",
)
def settings(gnina_path: pathlib.Path):
    """
    Create a runtime settings object for scoring runs which can be configured.
    """
    config = Settings(gnina_path=gnina_path)
    with open("settings.json", "w") as output:
        output.write(config.json(indent=2))

