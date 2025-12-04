#!/usr/bin/env python
# coding: utf-8

# # FEgrow: An Open-Source Molecular Builder and Free Energy Preparation Workflow
#
# **Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

# ## Overview
#
# Building and scoring molecules can be further streamlined by employing our established protocol. Here we show how to quickly build a library and score the entire library automatically for more than one recepetor.

import os  # afk
from glob import glob  # afk

import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace


from dask.distributed import LocalCluster


def main():
    OUTPUT_DIR = "fegrow_result"

    lc = LocalCluster(processes=True, n_workers=None, threads_per_worker=1)

    counter = 1

    input_folder = "./autofegrow-inputs/receptors_with_hydrogens"

    # Find all .pdb files in the input folder
    pdb_files = glob(os.path.join(input_folder, "*.pdb"))

    for pdb_file in pdb_files:
        # Prepare the ligand template
        print(" pdb file {} read in".format(counter))

        # scaffold = Chem.SDMolSupplier(core_5R83_path)[0]
        scaffold = Chem.SDMolSupplier("autofegrow-inputs/coreh.sdf")[0]

        with open("autofegrow-inputs/smiles-test-MERS.txt") as f:
            mols = f.read().splitlines()

            print("loading core finished round {}".format(counter))
            print("creating chemspace with dask round {}".format(counter))

            # create the chemical space
            cs = ChemSpace(dask_cluster=lc)
            cs.add_scaffold(scaffold)

            smiles = mols[0:]
            cs.add_smiles(smiles, protonate=True)
            cs

            sys = prody.parsePDB(pdb_file)
            rec = sys.select("not (nucleic or hetatm or water)")
            prody.writePDB("rec.pdb", rec)

            os.makedirs(OUTPUT_DIR) if not os.path.exists(OUTPUT_DIR) else None
            fegrow.fix_receptor(
                "rec.pdb", "{}/rec_final_{}.pdb".format(OUTPUT_DIR, counter)
            )
            print("pdb file into rec_final {}".format(counter))

            cs.add_protein("{}/rec_final_{}.pdb".format(OUTPUT_DIR, counter))
            print(
                "successfully added pdb {} to chemspace to evaluate conformers on it".format(
                    counter
                )
            )

            cs.evaluate(
                num_conf=500, gnina_gpu=False, penalty=0.0, al_ignore_penalty=False
            )

            cs.to_sdf("cs_optimised_molecules_in_rec_{}.sdf".format(counter))

            for i in range(len(cs)):
                try:
                    cs[i].to_file(
                        "best_conformers_in_rec_{0}_{1}.pdb".format(counter, i)
                    )  # afk
                except AttributeError:
                    print("No conformer for molecule", i)

                for i in range(len(cs)):
                    pdb_filename = "best_conformers_in_rec_{0}_{1}.pdb".format(
                        counter, i
                    )
                    sdf_filename = os.path.join(
                        OUTPUT_DIR, "rec_{0}_mol{1}.sdf".format(counter, i)
                    )
                    pdb_first_model = "tmp_first_model_{0}_{1}.pdb".format(counter, i)

                    try:
                        cs[i].to_file(pdb_filename)

                        with open(pdb_filename, "r") as infile:
                            lines = infile.readlines()

                        inside_model = False
                        first_model_lines = []
                        for line in lines:
                            if line.startswith("MODEL"):
                                if inside_model:
                                    break
                                inside_model = True
                            if inside_model:
                                first_model_lines.append(line)
                            if line.startswith("ENDMDL") and inside_model:
                                break

                        if not first_model_lines:
                            first_model_lines = lines

                        with open(pdb_first_model, "w") as outfile:
                            outfile.writelines(first_model_lines)

                        os.system(
                            "obabel -ipdb {} -O {}".format(
                                pdb_first_model, sdf_filename
                            )
                        )

                        os.remove(pdb_first_model)

                    except AttributeError:
                        print("No conformer for molecule", i)

        cs.df.to_csv("MERS-out.csv", index=True)

        counter += 1


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # Especially needed for frozen executables
    main()
