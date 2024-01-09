from pathlib import Path

from .conformers import generate_conformers, WrongCoreForMolecule
from .package import (
    ChemSpace,
    RMol,
    build_molecules,
    RGroups,
    Linkers,
)
from .receptor import fix_receptor, optimise_in_receptor, sort_conformers
from .toxicity import tox_props

# get the version
__version__ = open(Path(__file__).parent / "version.txt").read().strip()

__all__ = [
    RMol,
    generate_conformers,
    fix_receptor,
    optimise_in_receptor,
    tox_props,
    sort_conformers,
    RGroups,
    Linkers,
    build_molecules,
    __version__,
    WrongCoreForMolecule,
]
