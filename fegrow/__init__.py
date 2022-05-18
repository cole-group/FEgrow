from pathlib import Path

from .conformers import generate_conformers
from .package import RList, RMol, rep2D, build_molecules, ic50, RGroupGrid, RLinkerGrid
from .receptor import fix_receptor, optimise_in_receptor, sort_conformers
from .toxicity import tox_props
from .version import __version__

RGroups = RGroupGrid()
RLinkers = RLinkerGrid()

__version__ = open(Path(__file__).parent / 'version.txt').read().strip()

__all__ = [RMol,
           generate_conformers,
           rep2D, fix_receptor,
           optimise_in_receptor,
           tox_props,
           sort_conformers,
           RGroups,
           RLinkers,
           build_molecules,
           ic50,
           __version__]