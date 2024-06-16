#!/usr/bin/env python
# coding: utf-8

# # FEgrow: An Open-Source Molecular Builder and Free Energy Preparation Workflow
# 
# **Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**
# 
# * Add chemical functional groups (R-groups) in user-defined positions
# * Output ADMET properties
# * Perform constrained optimisation
# * Score poses
# * Output structures for free energy calculations

# ## Overview
# 
# This notebook demonstrates the entire `FEgrow` workflow for generating a series of ligands with a common core for a specific binding site, via the addition of a user-defined set of R-groups. 
# 
# These *de novo* ligands are then subjected to ADMET analysis. Valid conformers of the added R-groups are enumerated, and optimised in the context of the receptor binding pocket, optionally using hybrid machine learning / molecular mechanics potentials (ML/MM).
# 
# An ensemble of low energy conformers is generated for each ligand, and scored using the `gnina` convolutional neural network (CNN). Output structures are saved as `pdb` files ready for use in free energy calculations.
# 
# The target for this tutorial is the main protease (Mpro) of SARS-CoV-2, and the core and receptor structures are taken from a [recent study by Jorgensen & co-workers](https://doi.org/10.1021/acscentsci.1c00039).

# In[ ]:


import prody
from rdkit import Chem

import fegrow
from fegrow import RGroups, Linkers

rgroups = RGroups()
linkers = Linkers()


# # Prepare the ligand template

# The provided core structure `lig.pdb` has been extracted from a crystal structure of Mpro in complex with compound **4** from the Jorgensen study (PDB: 7L10), and a Cl atom has been removed to allow growth into the S3/S4 pocket. The template structure of the ligand is protonated with [Open Babel](http://openbabel.org/wiki/Main_Page):

# In[ ]:


#get_ipython().system('obabel sarscov2/lig.pdb -O sarscov2/coreh.sdf -p 7')


# Load the protonated ligand into FEgrow:

# In[ ]:


init_mol = Chem.SDMolSupplier('sarscov2/coreh.sdf', removeHs=False)[0]

# get the FEgrow representation of the rdkit Mol
template = fegrow.RMol(init_mol)


# Show the 2D (with indices) representation of the core. This is used to select the desired growth vector.

# In[ ]:




# Using the 2D drawing, select an index for the growth vector. Note that it is currently only possible to grow from hydrogen atom positions. In this case, we are selecting the hydrogen atom labelled H:40 to enable growth into the S3/S4 pocket of Mpro.

# In[ ]:


attachment_index = 40


# # Optional: insert a linker
# We have added a library of linkers suggested by [Erti et al](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/63845d130949e1fd5f589527/original/the-most-common-linkers-in-bioactive-molecules-and-their-bioisosteric-replacement-network.pdf).
# If you wish to extend your R groups selection via a linker, select them below. *:1 is defined to be attached to the core (there exists a mirror image of each linker i.e. *:1 & *:2 swapped).
# 
# Linkers combinatorially augment chosen R groups, so if you choose 2 linkers and 3 R groups, this will result in 6 molecules being built.
# 
# ### Note : If you want to use linkers make sure that you use the correct function below, in cell [11].


# or select one programmatically
selected_linker = linkers.loc[linkers['Name']=='R1CR2'].Mol.item()

# create just one template merged with a linker
template_with_linker = fegrow.build_molecule(template, selected_linker, attachment_index)


# note that the linker leaves the second attachement point prespecified (* character)


# # Select RGroups for your template

# R-groups can be selected interactively or programmaticaly.
# 
# We have provided a set of common R-groups (see `fegrow/data/rgroups/library`), which can be browsed and selected interactively below.
# 
# Molecules from the library can alternatively be selected by name, as demonstrated below.
# 
# Finally, user-defined R-groups may be provided as `.mol` files. *In this case, the hydrogen atom selected for attachment should be replaced by the element symbol R.* See the directory `manual_rgroups` for examples.

# In[ ]:

# you can also directly access the built-in dataframe programmatically
R_group_ethanol = rgroups.loc[rgroups['Name']=='*CCO'].Mol.item()


# make a list of R-group molecule


# # Build a congeneric series

# Now that the R-groups have been selected, we merge them with the ligand core:
# 
# ### Note : Use `rmols = fegrow.build_molecules(template_with_linker, selected_rgroups)` if using a linker.

# In[ ]:


# we can either use the original template (so no linker)
# in this case we have to specify the attachment index
rmol = fegrow.build_molecule(template, R_group_ethanol, attachment_index)

# or we can use the template merged with the linker
# in which case the attachement point is not needed (R* atom is used)
# rmols = fegrow.build_molecules(template_with_linker, selected_rgroups)


# In[ ]:


rmol


# The R-group library can also be viewed as a 2D grid, or individual molecules can be selected for 3D view (note that the conformation of the R-group has not yet been optimised):

# In[ ]:




# In[ ]:




# Once the ligands have been generated, they can be assessed for various ADMET properties, including Lipinksi rule of 5 properties, the presence of unwanted substructures or problematic functional groups, and synthetic accessibility.

# In[ ]:


rmol.toxicity()


# For each ligand, a specified number of conformers (`num_conf`) is generated by using the RDKit [ETKDG algorithm](https://doi.org/10.1021/acs.jcim.5b00654). Conformers that are too similar to an existing structure are discarded. Empirically, we have found that `num_conf=200` gives an exhaustive search, and `num_conf=50` gives a reasonable, fast search, in most cases.
# 
# If required, a third argument can be added `flexible=[0,1,...]`, which provides a list of additional atoms in the core that are allowed to be flexible. This is useful, for example, if growing from a methyl group and you would like the added R-group to freely rotate.

# In[ ]:


rmol.generate_conformers(num_conf=10,
                          minimum_conf_rms=0.5, 
                          # flexible=[3, 18, 20])
                        )


# ### Prepare the protein

# The protein-ligand complex structure is downloaded, and [PDBFixer](https://github.com/openmm/pdbfixer) is used to protonate the protein, and perform other simple repair:

# In[ ]:


# get the protein-ligand complex structure
#get_ipython().system('wget -nc https://files.rcsb.org/download/7L10.pdb')

# load the complex with the ligand
sys = prody.parsePDB('7L10.pdb')

# remove any unwanted molecules
rec = sys.select('not (nucleic or hetatm or water)')

# save the processed protein
prody.writePDB('rec.pdb', rec)

# fix the receptor file (missing residues, protonation, etc)
fegrow.fix_receptor("rec.pdb", "rec_final.pdb")

# load back into prody
rec_final = prody.parsePDB("rec_final.pdb")


# View enumerated conformers in complex with protein:

# In[ ]:




# Any conformers that clash with the protein (any atom-atom distance less than 1 Angstrom), are removed.

# In[ ]:


rmol.remove_clashing_confs(rec_final)


# In[ ]:




# ### Optimise conformers in context of protein

# The remaining conformers are optimised using hybrid machine learning / molecular mechanics (ML/MM), using the [ANI2x](https://doi.org/10.1021/acs.jctc.0c00121) neural nework potential for the ligand energetics (as long as it contains only the atoms H, C, N, O, F, S, Cl). Note that the Open Force Field [Parsley](https://doi.org/10.1021/acs.jctc.1c00571) force field is used for intermolecular interactions with the receptor.
# 
# `sigma_scale_factor`: is used to scale the Lennard-Jones radii of the atoms.
# 
# `relative_permittivity`: is used to scale the electrostatic interactions with the protein.
# 
# `water_model`: can be used to set the force field for any water molecules present in the binding site.

# In[ ]:


# opt_mol, energies
energies = rmol.optimise_in_receptor(
    receptor_file="rec_final.pdb",
    ligand_force_field="openff", 
    use_ani=True,
    sigma_scale_factor=0.8,
    relative_permittivity=4,
    water_model = None
)

final_energies = rmol.sort_conformers(energy_range=5)
rmol.to_file(f"best_conformers.pdb")
print(final_energies)

affinities = rmol.gnina(receptor_file="rec_final.pdb")
affinities




