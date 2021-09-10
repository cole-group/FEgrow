from copy import deepcopy
from typing import List, Tuple

import parmed
from openmmforcefields.generators import SystemGenerator
from openmmml import MLPotential
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from tqdm import tqdm
from typing_extensions import Literal

from .package import Mol

# fix for new openmm versions
try:
    from openmm import app, openmm, unit
except (ImportError, ModuleNotFoundError):
    from simtk.openmm import app, openmm
    from simtk import unit

from openff.toolkit.topology import Molecule as OFFMolecule


def fix_receptor(input_file: str, output_file: str, ph: float = 7.0):
    """
    Use PDBFixer to correct the input and add hydrogen.

    Args:
        input_file:
            The name of the pdb file which contains the receptor.
        output_file:
            The name of the pdb file the fixed receptor should be wrote to.
        ph:
            The ph the pronation state should be fixed for.
    """
    fixer = PDBFixer(filename=input_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file, "w"))


def _can_use_ani2x(molecule: OFFMolecule) -> bool:
    """
    Check if ani2x can be used for this molecule by inspecting the elements.
    """
    mol_elements = set([atom.element.symbol for atom in molecule.atoms])
    ani2x_elements = {"H", "C", "N", "O", "S", "F", "Cl"}
    if (
        mol_elements - ani2x_elements
        or molecule.total_charge.value_in_unit(unit.elementary_charge) != 0
    ):
        # if there is any difference in the sets or a net charge ani2x can not be used.
        return False
    return True


ForceField = Literal["openff", "gaff"]


def optimise_in_receptor(
    ligand: Mol,
    receptor_file: str,
    ligand_force_field: ForceField,
    use_ani: bool = True,
) -> Tuple[Mol, List[float]]:
    """
    For each of the input molecule conformers optimise the system using the chosen force field with the receptor held fixed.

    Args:
        ligand:
            The ligand with starting conformers already filtered for clashes with the receptor.
        receptor_file:
            The pdb file of the fixed and pronated receptor.
        ligand_force_field:
            The base ligand force field that should be used.
        use_ani:
            If we should try and use ani2x for the internal energy of the ligand.

    Returns:
        A copy of the input molecule with the optimised positions.
    """

    ligand_force_fields = {
        "openff": "openff_unconstrained-1.3.0.offxml",
        "gaff": "gaff-2.11",
    }

    # assume the receptor has already been fixed and hydrogens have been added.
    receptor = app.PDBFile(receptor_file)
    # receptor forcefield
    receptor_ff = "amber14/protein.ff14SB.xml"

    # load the molecule into openff
    openff_mol = OFFMolecule.from_rdkit(ligand, allow_undefined_stereo=True)

    # now we need to make our parameterised system, make a system generator
    system_generator = SystemGenerator(
        forcefields=[receptor_ff],
        small_molecule_forcefield=ligand_force_fields[ligand_force_field],
        cache="db.json",
        molecules=openff_mol,
    )

    # now make a combined receptor and ligand topology
    parmed_receptor = parmed.openmm.load_topology(
        receptor.topology, xyz=receptor.positions
    )
    parmed_ligand = parmed.openmm.load_topology(
        openff_mol.to_topology().to_openmm(), xyz=openff_mol.conformers[0]
    )
    complex_structure = parmed_receptor + parmed_ligand
    # build the complex system
    system = system_generator.create_system(complex_structure.topology)

    # work out the index of the atoms in the ligand assuming the are the last chain?
    # is this always true if we add the ligand to the receptor?
    ligand_res = list(complex_structure.topology.residues())[-1]
    ligand_idx = [atom.index for atom in ligand_res.atoms()]
    # now set all atom mass to 0 if not in the ligand
    for i in range(system.getNumParticles()):
        if i not in ligand_idx:
            system.setParticleMass(i, 0)
    # if we want to use ani2x check we can and adapt the system
    if use_ani and _can_use_ani2x(openff_mol):
        print("using ani2x")
        potential = MLPotential("ani2x")
        complex_system = potential.createMixedSystem(
            complex_structure.topology, system, ligand_idx
        )
    else:
        complex_system = system

    # propagate the System with Langevin dynamics note integrator note used.
    time_step = 1 * unit.femtoseconds  # simulation timestep
    temperature = 300 * unit.kelvin  # simulation temperature
    friction = 1 / unit.picosecond  # collision rate
    integrator_min = openmm.LangevinIntegrator(temperature, friction, time_step)

    # set up an openmm simulation
    simulation = app.Simulation(
        complex_structure.topology, complex_system, integrator_min
    )

    # save the receptor coords as they should be consistent
    receptor_coords = parmed_receptor.positions

    # loop over the conformers and energy minimise and store the final positions
    final_mol = Mol(deepcopy(ligand))
    final_mol.save_template(ligand.template)
    final_mol.RemoveAllConformers()
    energies = []
    for i, conformer in enumerate(
        tqdm(openff_mol.conformers, desc="Optimising conformer: ", ncols=80)
    ):
        # now we need to make the ligand coords
        lig_coords = conformer.value_in_unit(unit.angstrom)
        # make the vec3 list
        lig_vec = [openmm.Vec3(*i) for i in lig_coords] * unit.angstrom
        complex_coords = receptor_coords + lig_vec
        # write out the starting positions
        # with open(f"system_start_conformer_{i}.pdb", "w") as outfile:
        #     app.PDBFile.writeFile(complex_structure.topology, complex_coords, outfile)
        # set the initial positions
        simulation.context.setPositions(complex_coords)
        # now minimize the energy
        simulation.minimizeEnergy()

        # now write out the final coords
        min_state = simulation.context.getState(getPositions=True, getEnergy=True)
        energies.append(
            min_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        )
        positions = min_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        final_conformer = Chem.Conformer()
        for j, coord in enumerate(positions[ligand_idx[0]:]):
            atom_position = Point3D(*coord)
            final_conformer.SetAtomPosition(j, atom_position)
        final_mol.AddConformer(final_conformer, assignId=True)

        # with open(f"system_min_conformer_{i}.pdb", "w") as outfile:
        #     app.PDBFile.writeFile(complex_structure.topology, min_state.getPositions().value_in_unit(unit.angstrom), outfile)

    return final_mol, energies
