"""Functionality for working with ML force fields."""

import atexit
import os
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from typing import Literal

from openff.toolkit import Molecule
from openmmml import MLPotential as OMMMLPotential

__all__ = [
    "AVAILABLE_ML_FORCE_FIELDS",
    "ANI2X",
    "MACEOFF23Small",
    "MACEOFF23Medium",
    "MACEOFF23Large",
    "EGRET1",
]

# Model accessed 24/05/25
_EGRET1_MODEL_URL = (
    "https://github.com/rowansci/egret-public/raw/227d6641e6851"
    "eb1037d48712462e4ce61c1518f/compiled_models/EGRET_1.model"
)

AVAILABLE_ML_FORCE_FIELDS = Literal[
    "ani2x",
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "egret-1",
]


def _check_mace_installed() -> None:
    """Check that mace-torch is installed."""
    try:
        import mace  # noqa: F401

    except ImportError:
        msg = (
            "Using a MACE force field requires the `mace_torch` package. "
            "Please install it with `pip install mace-torch`."
        )
        raise ImportError(msg)


class _MLForceField(ABC):
    """Abstract base class for machine learning force fields."""

    name: str
    allowed_elements: frozenset[str]
    allow_charged: bool = False

    @classmethod
    @abstractmethod
    def get_potential(cls) -> OMMMLPotential:
        """Get the OpenMM MLPotential for this force field."""

    @classmethod
    def is_compatible_with_molecule(cls, molecule: Molecule) -> bool:
        """
        Check if the force field is compatible with the given molecule.

        Args:
            molecule (OFFMolecule): The molecule to check compatibility with.

        Returns:
            bool: True if compatible, False otherwise.
        """
        mol_elements = set(atom.symbol for atom in molecule.atoms)
        if not mol_elements.issubset(cls.allowed_elements):
            return False
        # Work out the total formal charge
        rdkit_mol = molecule.to_rdkit()
        total_charge = sum(atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms())
        if not cls.allow_charged and total_charge != 0:
            return False
        return True


class ANI2X(_MLForceField):
    name = "ani2x"
    allowed_elements = frozenset(["H", "C", "N", "O", "S", "F", "Cl"])
    allow_charged = False

    @classmethod
    def get_potential(cls) -> OMMMLPotential:
        """Get the ANI2x potential."""
        return OMMMLPotential(cls.name)


class _MACE_BASE(_MLForceField):
    """Base class for MACE force fields."""

    allowed_elements = frozenset(["C", "H", "N", "O", "S", "F", "Cl", "Br"])
    allow_charged = False

    @classmethod
    def get_potential(cls) -> OMMMLPotential:
        """Get the MACE potential."""

        cls._check_available()

        print(
            "MACE models are distributed under the ASL which"
            "does not permit commercial use."
        )

        return OMMMLPotential(cls.name)

    @staticmethod
    def _check_available() -> None:
        """Check if the MACE force field is available."""
        _check_mace_installed()


class MACEOFF23Small(_MACE_BASE):
    """MACE-OFF 23 Small force field."""

    name = "mace-off23-small"


class MACEOFF23Medium(_MACE_BASE):
    """MACE-OFF 23 Medium force field."""

    name = "mace-off23-medium"


class MACEOFF23Large(_MACE_BASE):
    """MACE-OFF 23 Large force field."""

    name = "mace-off23-large"


class EGRET1(_MLForceField):
    """EGRET-1 force field."""

    name = "egret-1"
    allowed_elements = frozenset(["C", "H", "N", "O", "S", "F", "Cl", "Br"])
    allow_charged = False

    @classmethod
    def get_potential(cls) -> OMMMLPotential:
        """Get the EGRET-1 potential."""

        cls._check_available()

        tmp_file = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
        tmp_file.close()  # Close so urllib can write to it
        print(f"Downloading Egret-1 model from {_EGRET1_MODEL_URL}")
        urllib.request.urlretrieve(_EGRET1_MODEL_URL, filename=tmp_file.name)

        # Register file for deletion at program exit
        atexit.register(
            lambda: os.remove(tmp_file.name) if os.path.exists(tmp_file.name) else None
        )

        return OMMMLPotential("mace", modelPath=tmp_file.name)

    @staticmethod
    def _check_available() -> None:
        """Check if the EGRET-1 MLP is available."""
        _check_mace_installed()


_MLFF_NAME_TO_CLASS: dict[AVAILABLE_ML_FORCE_FIELDS, type[_MLForceField]] = {
    "ani2x": ANI2X,
    "mace-off23-small": MACEOFF23Small,
    "mace-off23-medium": MACEOFF23Medium,
    "mace-off23-large": MACEOFF23Large,
    "egret-1": EGRET1,
}
