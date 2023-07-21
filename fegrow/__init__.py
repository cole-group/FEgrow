from pathlib import Path

from .conformers import generate_conformers, WrongCoreForMolecule
from .package import (
    RList,
    RMol,
    rep2D,
    build_molecules,
    ic50,
    RGroupGrid,
    RLinkerGrid,
    link,
)
from .receptor import (
    fix_receptor,
    optimise_in_receptor,
    sort_conformers
)
from .toxicity import tox_props

# get the version
__version__ = open(Path(__file__).parent / "version.txt").read().strip()

__all__ = [
    RMol,
    generate_conformers,
    rep2D,
    fix_receptor,
    optimise_in_receptor,
    tox_props,
    sort_conformers,
    RGroupGrid,
    RLinkerGrid,
    link,
    build_molecules,
    ic50,
    __version__,
    WrongCoreForMolecule
]
