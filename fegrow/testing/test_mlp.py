"""Tests for the mlp module."""

from unittest.mock import MagicMock, patch

import pytest
from openff.toolkit import Molecule
from openmmml import MLPotential

from fegrow.mlp import (
    ANI2X,
    EGRET1,
    MACEOFF23Large,
    MACEOFF23Medium,
    MACEOFF23Small,
    _MLFF_NAME_TO_CLASS,
    _check_mace_installed,
)


class TestCheckMaceInstalled:
    """Tests for _check_mace_installed function."""

    @patch.dict("sys.modules", {"mace": MagicMock()})
    def test_check_mace_installed_success(self):
        """Test that no error is raised when mace is installed."""
        # Should not raise any exception
        _check_mace_installed()

    def test_check_mace_not_installed(self):
        """Test that ImportError is raised when mace is not installed."""
        with patch.dict("sys.modules", {"mace": None}):
            with pytest.raises(ImportError, match="mace_torch"):
                # Force re-import to trigger the import error
                import importlib
                import fegrow.mlp as mlp_module

                importlib.reload(mlp_module)
                mlp_module._check_mace_installed()


class TestANI2X:
    """Tests for the ANI2X force field."""

    def test_name(self):
        """Test that the name is correct."""
        assert ANI2X.name == "ani2x"

    def test_allowed_elements(self):
        """Test that allowed elements are correct."""
        expected = frozenset(["H", "C", "N", "O", "S", "F", "Cl"])
        assert ANI2X.allowed_elements == expected

    def test_allow_charged(self):
        """Test that charged molecules are not allowed."""
        assert ANI2X.allow_charged is False

    def test_get_potential(self):
        """Test that get_potential returns an MLPotential."""
        potential = ANI2X.get_potential()
        assert isinstance(potential, MLPotential)

    def test_compatible_with_methane(self):
        """Test compatibility with a simple molecule (methane)."""
        methane = Molecule.from_smiles("C")
        assert ANI2X.is_compatible_with_molecule(methane) is True

    def test_compatible_with_ethanol(self):
        """Test compatibility with ethanol."""
        ethanol = Molecule.from_smiles("CCO")
        assert ANI2X.is_compatible_with_molecule(ethanol) is True

    def test_compatible_with_sulfur(self):
        """Test compatibility with sulfur-containing molecule."""
        dimethyl_sulfide = Molecule.from_smiles("CSC")
        assert ANI2X.is_compatible_with_molecule(dimethyl_sulfide) is True

    def test_compatible_with_fluorine(self):
        """Test compatibility with fluorine-containing molecule."""
        fluoromethane = Molecule.from_smiles("CF")
        assert ANI2X.is_compatible_with_molecule(fluoromethane) is True

    def test_compatible_with_chlorine(self):
        """Test compatibility with chlorine-containing molecule."""
        chloromethane = Molecule.from_smiles("CCl")
        assert ANI2X.is_compatible_with_molecule(chloromethane) is True

    def test_incompatible_with_bromine(self):
        """Test incompatibility with bromine (not in allowed elements)."""
        bromomethane = Molecule.from_smiles("CBr")
        assert ANI2X.is_compatible_with_molecule(bromomethane) is False

    def test_incompatible_with_iodine(self):
        """Test incompatibility with iodine (not in allowed elements)."""
        iodomethane = Molecule.from_smiles("CI")
        assert ANI2X.is_compatible_with_molecule(iodomethane) is False

    def test_incompatible_with_charged_molecule(self):
        """Test incompatibility with charged molecules."""
        # Methylammonium (positively charged)
        charged_mol = Molecule.from_smiles("[NH3+]C")
        assert ANI2X.is_compatible_with_molecule(charged_mol) is False

    def test_incompatible_with_negatively_charged_molecule(self):
        """Test incompatibility with negatively charged molecules."""
        # Acetate (negatively charged)
        charged_mol = Molecule.from_smiles("CC([O-])=O")
        assert ANI2X.is_compatible_with_molecule(charged_mol) is False


class TestMACEForceFields:
    """Tests for MACE force field classes."""

    @pytest.mark.parametrize(
        "mace_class,expected_name",
        [
            (MACEOFF23Small, "mace-off23-small"),
            (MACEOFF23Medium, "mace-off23-medium"),
            (MACEOFF23Large, "mace-off23-large"),
        ],
    )
    def test_names(self, mace_class, expected_name):
        """Test that MACE force field names are correct."""
        assert mace_class.name == expected_name

    @pytest.mark.parametrize(
        "mace_class",
        [MACEOFF23Small, MACEOFF23Medium, MACEOFF23Large],
    )
    def test_allowed_elements(self, mace_class):
        """Test that MACE allowed elements are correct."""
        expected = frozenset(["C", "H", "N", "O", "S", "F", "Cl", "Br"])
        assert mace_class.allowed_elements == expected

    @pytest.mark.parametrize(
        "mace_class",
        [MACEOFF23Small, MACEOFF23Medium, MACEOFF23Large],
    )
    def test_allow_charged(self, mace_class):
        """Test that MACE force fields don't allow charged molecules."""
        assert mace_class.allow_charged is False

    @pytest.mark.parametrize(
        "mace_class",
        [MACEOFF23Small, MACEOFF23Medium, MACEOFF23Large],
    )
    @patch("fegrow.mlp._check_mace_installed")
    @patch("builtins.print")
    def test_get_potential(self, mock_print, mock_check, mace_class):
        """Test that get_potential returns an MLPotential for MACE models."""
        potential = mace_class.get_potential()
        assert isinstance(potential, MLPotential)
        mock_check.assert_called_once()
        # Check that the ASL license warning is printed
        mock_print.assert_called_once()
        assert "ASL" in mock_print.call_args[0][0]

    @pytest.mark.parametrize(
        "mace_class",
        [MACEOFF23Small, MACEOFF23Medium, MACEOFF23Large],
    )
    def test_compatible_with_bromine(self, mace_class):
        """Test MACE compatibility with bromine (allowed in MACE)."""
        bromomethane = Molecule.from_smiles("CBr")
        assert mace_class.is_compatible_with_molecule(bromomethane) is True

    @pytest.mark.parametrize(
        "mace_class",
        [MACEOFF23Small, MACEOFF23Medium, MACEOFF23Large],
    )
    def test_incompatible_with_iodine(self, mace_class):
        """Test MACE incompatibility with iodine (not in allowed elements)."""
        iodomethane = Molecule.from_smiles("CI")
        assert mace_class.is_compatible_with_molecule(iodomethane) is False

    @pytest.mark.parametrize(
        "mace_class",
        [MACEOFF23Small, MACEOFF23Medium, MACEOFF23Large],
    )
    def test_incompatible_with_charged_molecule(self, mace_class):
        """Test MACE incompatibility with charged molecules."""
        charged_mol = Molecule.from_smiles("[NH3+]C")
        assert mace_class.is_compatible_with_molecule(charged_mol) is False


class TestEGRET1:
    """Tests for the EGRET-1 force field."""

    def test_name(self):
        """Test that the name is correct."""
        assert EGRET1.name == "egret-1"

    def test_allowed_elements(self):
        """Test that allowed elements are correct."""
        expected = frozenset(["C", "H", "N", "O", "S", "F", "Cl", "Br"])
        assert EGRET1.allowed_elements == expected

    def test_allow_charged(self):
        """Test that charged molecules are not allowed."""
        assert EGRET1.allow_charged is False

    @patch("fegrow.mlp._check_mace_installed")
    @patch("fegrow.mlp.urllib.request.urlretrieve")
    @patch("fegrow.mlp.tempfile.NamedTemporaryFile")
    @patch("fegrow.mlp.atexit.register")
    @patch("builtins.print")
    def test_get_potential(
        self,
        mock_print,
        mock_atexit,
        mock_tempfile,
        mock_urlretrieve,
        mock_check,
    ):
        """Test that get_potential downloads model and returns
        an MLPotential.
        """
        # Mock the temporary file
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_egret.model"
        mock_tempfile.return_value = mock_file

        potential = EGRET1.get_potential()

        assert isinstance(potential, MLPotential)
        mock_check.assert_called_once()
        mock_urlretrieve.assert_called_once()
        mock_atexit.assert_called_once()
        # Check that download message is printed
        mock_print.assert_called_once()
        assert "Downloading" in mock_print.call_args[0][0]

    def test_compatible_with_bromine(self):
        """Test EGRET-1 compatibility with bromine."""
        bromomethane = Molecule.from_smiles("CBr")
        assert EGRET1.is_compatible_with_molecule(bromomethane) is True

    def test_incompatible_with_iodine(self):
        """Test EGRET-1 incompatibility with iodine."""
        iodomethane = Molecule.from_smiles("CI")
        assert EGRET1.is_compatible_with_molecule(iodomethane) is False

    def test_incompatible_with_charged_molecule(self):
        """Test EGRET-1 incompatibility with charged molecules."""
        charged_mol = Molecule.from_smiles("[NH3+]C")
        assert EGRET1.is_compatible_with_molecule(charged_mol) is False


class TestMLFFNameToClass:
    """Tests for the _MLFF_NAME_TO_CLASS mapping."""

    def test_mapping_contains_all_models(self):
        """Test that the mapping contains all available models."""
        expected_keys = {
            "ani2x",
            "mace-off23-small",
            "mace-off23-medium",
            "mace-off23-large",
            "egret-1",
        }
        assert set(_MLFF_NAME_TO_CLASS.keys()) == expected_keys

    def test_mapping_values(self):
        """Test that the mapping values are correct classes."""
        assert _MLFF_NAME_TO_CLASS["ani2x"] == ANI2X
        assert _MLFF_NAME_TO_CLASS["mace-off23-small"] == MACEOFF23Small
        assert _MLFF_NAME_TO_CLASS["mace-off23-medium"] == MACEOFF23Medium
        assert _MLFF_NAME_TO_CLASS["mace-off23-large"] == MACEOFF23Large
        assert _MLFF_NAME_TO_CLASS["egret-1"] == EGRET1


class TestCompatibilityEdgeCases:
    """Tests for edge cases in molecule compatibility checking."""

    def test_complex_molecule_compatible(self):
        """Test compatibility with a more complex molecule."""
        # Aspirin
        aspirin = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert ANI2X.is_compatible_with_molecule(aspirin) is True
        assert MACEOFF23Medium.is_compatible_with_molecule(aspirin) is True
        assert EGRET1.is_compatible_with_molecule(aspirin) is True

    def test_molecule_with_all_allowed_elements_ani2x(self):
        """Test molecule containing all ANI2X allowed elements."""
        # Create a molecule with H, C, N, O, S, F, Cl
        # Hypothetical molecule for testing
        mol = Molecule.from_smiles("FC(Cl)C(=O)N(C)CS", allow_undefined_stereo=True)
        assert ANI2X.is_compatible_with_molecule(mol) is True

    def test_molecule_with_all_allowed_elements_mace(self):
        """Test molecule containing all MACE allowed elements."""
        # Create a molecule with C, H, N, O, S, F, Cl, Br
        mol = Molecule.from_smiles("FC(Cl)C(=O)N(C)CSBr", allow_undefined_stereo=True)
        assert MACEOFF23Medium.is_compatible_with_molecule(mol) is True
        assert EGRET1.is_compatible_with_molecule(mol) is True

    def test_empty_molecule_elements(self):
        """Test that single atoms are compatible if in allowed set."""
        # Single carbon atom
        carbon = Molecule.from_smiles("C")
        assert ANI2X.is_compatible_with_molecule(carbon) is True
        assert MACEOFF23Medium.is_compatible_with_molecule(carbon) is True
        assert EGRET1.is_compatible_with_molecule(carbon) is True
