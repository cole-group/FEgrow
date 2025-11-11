import numpy as np
import prody
from rdkit import Chem

from fegrow import build_molecule


def test_mcs_atom_freezing(sars_scaffold_chunk_sdf, rec_7l10_final_path):
    rgroup = Chem.AddHs(Chem.MolFromSmiles("[*]N[CH3]"))
    rmol = build_molecule(sars_scaffold_chunk_sdf, rgroup, 8)

    rmol.generate_conformers(
        num_conf=50,
    )

    rec_final = prody.parsePDB(rec_7l10_final_path)
    rmol.remove_clashing_confs(rec_final)

    unmin_pos = rmol.GetConformer().GetPositions()
    scaffold_atoms = [a.GetIdx() for a in sars_scaffold_chunk_sdf.GetAtoms()]
    rmol.optimise_in_receptor(
        receptor_file=rec_7l10_final_path,
        ligand_force_field="openff",
        ligand_intramolecular_mlp=None,
        water_model=None,
        platform_name="CPU",
        ligand_indices_to_freeze=scaffold_atoms,
    )

    # check if the freezing worked
    min_pos = rmol.GetConformer().GetPositions()
    np.testing.assert_almost_equal(min_pos[scaffold_atoms], unmin_pos[scaffold_atoms])

    ## and reversely, check if the exception
    ## test optimisation without freezing the common area
    rmol.GetConformer().SetPositions(unmin_pos)
    rmol.optimise_in_receptor(
        receptor_file=rec_7l10_final_path,
        ligand_force_field="openff",
        ligand_intramolecular_mlp=None,
        water_model=None,
        platform_name="CPU",
    )

    with np.testing.assert_raises(AssertionError):
        min_pos_unfrozen = rmol.GetConformer().GetPositions()
        np.testing.assert_almost_equal(
            min_pos_unfrozen[scaffold_atoms], unmin_pos[scaffold_atoms]
        )
