from .conformers import generate_conformers
from .package import RList, RMol, rep2D, build_molecules, ic50, RGroupGrid, RLinkerGrid, link
from .receptor import fix_receptor, optimise_in_receptor, sort_conformers
from .toxicity import tox_props

RGroups = RGroupGrid()
RLinkers = RLinkerGrid()

# get the version
__version__ = open('version.txt').read().strip()

__all__ = [RMol,
           generate_conformers,
           rep2D, fix_receptor,
           optimise_in_receptor,
           tox_props,
           sort_conformers,
           RGroups,
           RLinkers,
           link,
           build_molecules,
           ic50,
           __version__]