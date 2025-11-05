#!/usr/bin/env python
# coding: utf-8

# # FEgrow: An Open-Source Molecular Builder and Free Energy Preparation Workflow
# 
# **Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

# ## Overview
# 
# Building and scoring molecules can be further streamlined by employing our established protocol. Here we show how to quickly build a library and score the entire library. 

# In[1]:

import os   # afk
from glob import glob  #afk 

import pandas as pd
import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace

#from fegrow.testing import core_5R83_path, rec_5R83_path, data_5R83_path

from dask.distributed import LocalCluster


import os
import shutil


def main():

    OUTPUT_DIR = "fegrow_result"
    
     

 


    lc = LocalCluster(processes=True, n_workers=None, threads_per_worker=1)    
        
    counter = 1    
        

    input_folder = "./Receptors"        
            
        # Find all .pdb files in the input folder
    pdb_files = glob(os.path.join(input_folder, "*.pdb")    )        

    for pdb_file in pdb_files:    
        # Prepare the ligand template
        print(f" pdb file {counter} read in")    
            

        #!grep "XEY" 7l10.pdb > in.pdb    
        #!obabel -ipdb lig-rebuilt.pdb -O in-H.sdf -p 7   


            


        #scaffold = Chem.SDMolSupplier(core_5R83_path)[0]    
        scaffold = Chem.SDMolSupplier('coreh.sdf')[0]


        with open('smiles-test-MERS.txt') as f:    
            
            mols = f.read().splitlines()    

            #pattern = scaffold

            #for i in range(len(mols)):
            #    mol = Chem.MolFromSmiles(mols[i])
            #    if mol.HasSubstructMatch(pattern) == False:
            #        print(i, mols[i])

            print(f"loading core finished round {counter}")
            print(f"creating chemspace with dask round {counter}")

            # create the chemical space
            cs = ChemSpace(dask_cluster=lc)   

            #cs._dask_cluster

            # we're not growing the scaffold, we're superimposing bigger molecules on it
            cs.add_scaffold(scaffold)

            smiles = mols[0:]

            # here we add Smiles which should already have been matched
            # to the scaffold (rdkit Mol.HasSubstructureMatch)
            #cs.add_smiles(smiles.to_list(), protonate=True)
            cs.add_smiles(smiles, protonate=True)
            cs

            # get the protein-ligand complex structure
            #!wget -nc https://files.rcsb.org/download/3vf6.pdb

            

            # load the complex with the ligand
            sys = prody.parsePDB(pdb_file)

            # remove any unwanted molecules
            rec = sys.select('not (nucleic or hetatm or water)')

            # save the processed protein
            prody.writePDB('rec.pdb', rec)

            # fix the receptor file (missing residues, protonation, etc)
            #f os.path.exists(OUTPUT_DIR):
                #   shutil.rmtree(OUTPUT_DIR)  # Optional: remove old results
            
            os.makedirs(OUTPUT_DIR, exist_ok=True) 
            fegrow.fix_receptor("rec.pdb", f"{OUTPUT_DIR}/rec_final_{counter}.pdb")
            print(f"pdb file into rec_final {counter}")
            # load back into prody
            #rec_final = prody.parsePDB("rec_final.pdb")
            #rec_final = prody.parsePDB("out.pdb")

            # fix the receptor file (missing residues, protonation, etc)
            ##fegrow.fix_receptor("7t79-H-prep.pdb", "rec_final.pdb")

            # load back into prody
            ##rec_final = prody.parsePDB("rec_final.pdb")

            #!grep "ATOM" ../structures/7t79-H.pdb > rec_final.pdb
            #cs.add_protein(rec_5R83_path)
            cs.add_protein(f"{OUTPUT_DIR}/rec_final_{counter}.pdb")
            print(f"successfully added pdb {counter} to chemspace to evaluate conformers on it")

            cs.evaluate(num_conf=500, gnina_gpu=False, penalty=0.0, al_ignore_penalty=False)


            cs.to_sdf(f"cs_optimised_molecules_in_rec_{counter}.sdf")


            for i in range (len(cs)):
                try:
                    cs[i].to_file("best_conformers_in_rec_{0}_{1}.pdb".format(counter,i)) # afk
                except AttributeError:
                    print("No conformer for molecule", i)

                for i in range(len(cs)):
                    pdb_filename = "best_conformers_in_rec_{0}_{1}.pdb".format(counter, i)
                    sdf_filename = os.path.join(OUTPUT_DIR, "rec_{0}_mol{1}.sdf".format(counter, i))
                    pdb_first_model = "tmp_first_model_{0}_{1}.pdb".format(counter, i)

                    try:
                        cs[i].to_file(pdb_filename)

                        # Extract first MODEL block (first conformer)
                        with open(pdb_filename, 'r') as infile:
                            lines = infile.readlines()

                        inside_model = False
                        first_model_lines = []
                        for line in lines:
                            if line.startswith("MODEL"):
                                if inside_model:
                                    break  # already captured one MODEL
                                inside_model = True
                            if inside_model:
                                first_model_lines.append(line)
                            if line.startswith("ENDMDL") and inside_model:
                                break  # stop after first MODEL

                        # If no MODEL blocks are found, fallback to entire file (may be single conformer)
                        if not first_model_lines:
                            first_model_lines = lines

                        # Write the first conformer to a new temporary file
                        with open(pdb_first_model, 'w') as outfile:
                            outfile.writelines(first_model_lines)

                        # Convert to .sdf using obabel
                        os.system(f"obabel -ipdb {pdb_first_model} -O {sdf_filename}")

                        os.remove(pdb_first_model)

                    except AttributeError:
                        print("No conformer for molecule", i)

        cs.df.to_csv('MERS-out.csv', index=True)
    
        counter += 1

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Especially needed for frozen executables
    main()
