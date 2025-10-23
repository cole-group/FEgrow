import os
from glob import glob
from chimera import runCommand as rc

# Input folder ("" means current directory)
input_folder = ""

# Output folder
output_folder = "output_with_hydrogens"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Find all .pdb files
pdb_files = glob(os.path.join(input_folder, "*.pdb"))

for pdb_file in pdb_files:
    print("Processing: {}".format(pdb_file))
    rc("open {}".format(pdb_file))
    
    rc("addh")
    
    base_name = os.path.basename(pdb_file).replace(".pdb", "_withH.pdb")
    output_path = os.path.join(output_folder, base_name)
    
    rc("write format pdb 0 {}".format(output_path))
    rc("close all")

rc("stop now")
