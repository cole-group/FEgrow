



class DaskTasks:
    @staticmethod
    @dask.delayed
    def obabel_protonate(smi):
        return subprocess.run(['obabel', f'-:{smi}', '-osmi', '-p', '7', '-xh'],
                              capture_output=True).stdout.decode().strip()

    @staticmethod
    @dask.delayed
    def scaffold_check(smih, scaffold):
        """

        :param smih:
        :param scaffold:
        :return: [has_scaffold_bool, protonated_smiles]
        """
        params = Chem.SmilesParserParams()
        params.removeHs = False

        mol = Chem.MolFromSmiles(smih, params=params)
        if mol is None:
            return False, None

        if mol.HasSubstructMatch(scaffold):
            return True, smih

        return False, None


def gnina(mol, receptor, gnina_path, gnina_gpu=False):

    extras = []
    if gnina_gpu is False:
        extras.append("--no_gpu")

    # make a temporary sdf file for gnina
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf") as TMP_SDF:
        with Chem.SDWriter(TMP_SDF.name) as w:
            for conformer in mol.GetConformers():
                w.write(mol, confId=conformer.GetId())

        # run the code on the sdf
        process = subprocess.run(
            [
                gnina_path,
                "--score_only",
                "-l",
                TMP_SDF.name,
                "-r",
                receptor,
                "--seed",
                "0",
                "--stripH",
                "False",
            ] + extras
            ,
            capture_output=True,
            check=True,
        )

    output = process.stdout.decode("utf-8")
    CNNaffinities_str = re.findall(r"CNNaffinity: (-?\d+.\d+)", output)

    # convert to floats
    CNNaffinities = list(map(float, CNNaffinities_str))

    return mol, CNNaffinities


def build_molecule(
    scaffolds: Chem.Mol,
    r_group: Union[Chem.Mol, str],
    scaffold_point: Optional[int] = None,
    rgroup_point: Optional[int] = None,
    keep: Optional[int] = None,
):
    """

    :param scaffolds:
    :param r_groups:
    :param scaffold_point: attachement point on the scaffold
    :param keep: When the scaffold is grown from an internal atom that divides the molecules into separate
        submolecules, keep the submolecule with this atom index.
    :return:
    """

    if isinstance(r_group, list) and len(r_group) == 0:
        raise ValueError("Empty list received. Please pass any R-groups or R-linkers. ")

    if isinstance(scaffold_point, list) or isinstance(scaffolds, list):
        raise ValueError("Only one scaffold and rgroup at at time is permitted. ")

    # scaffolds were created earlier, they are most likely templates combined with linkers,
    if isinstance(scaffolds, ChemSpace):
        # fixme - these should become "the cores", it's simple with one mol, and tricky with more of them,
        scaffolds = [mol for idx, mol in scaffolds.dataframe.Mol.items()]

    # convert smiles into a molecule
    if isinstance(r_group, str):
        if '*' not in r_group and rgroup_point is None:
            raise ValueError("The SMILES used for the R-Group has to have an R-group atom. "
                             "That is the character * in Smiles, or you can use the RDKit function .SetAtomicNum(0) ")
        params = Chem.SmilesParserParams()
        params.removeHs = False
        r_group = Chem.MolFromSmiles(r_group, params=params)

        # set the attachement point on the R-group
        if rgroup_point is not None:
            r_group.GetAtomWithIdx(rgroup_point).SetAtomicNum(0)

    built_mols = build_molecules_with_rdkit(
        scaffolds, r_group, scaffold_point, keep
    )

    mol, scaffold, scaffold_no_attachement = built_mols
    rmol = RMol(mol)

    if hasattr(scaffold, 'template') and isinstance(scaffold.template, rdkit.Chem.Mol):
        # save the original scaffold (e.g. before the linker was added)
        # this means that conformer generation will always have to regenerate the previously added R-groups/linkers
        rmol._save_template(scaffold.template)
    else:
        rmol._save_template(scaffold_no_attachement)

    return rmol


def _evaluate_atomic(scaffold,
                     smiles,
                     pdb_filename,
                     h=None,
                     scoring_function=None,
                     num_conf=50,
                     minimum_conf_rms=0.5,
                     use_ani=True,
                     platform="CPU",
                     gnina_gpu=False,
                     skip_optimisation=False,
                     full_evaluation=None
                     ):
    """

    :param scaffold:
    :param h:
    :param smiles: Full Smiles.
    :param scoring_function:
    :param pdb_filename:
    :param gnina_path:
    :return:
    """

    if full_evaluation is not None:
        return full_evaluation(scaffold,
                     h,
                     smiles,
                     pdb_filename,
                     scoring_function=None,
                     num_conf=50,
                     minimum_conf_rms=0.5,
                     use_ani=use_ani,
                     platform="CPU",
                     skip_optimisation=False)

    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    rmol = RMol(Chem.MolFromSmiles(smiles, params=params))

    # remove the h
    # this is to help the rdkit's HasSubstructMatch
    if h is not None:
        scaffold = copy.deepcopy(scaffold)
        scaffold_m = Chem.EditableMol(scaffold)
        scaffold_m.RemoveAtom(int(h))
        scaffold = scaffold_m.GetMol()

    rmol._save_template(scaffold)

    rmol.generate_conformers(num_conf=num_conf, minimum_conf_rms=minimum_conf_rms)
    rmol.remove_clashing_confs(pdb_filename)
    if not skip_optimisation:
        rmol.optimise_in_receptor(
            receptor_file=pdb_filename,
            ligand_force_field="openff",
            use_ani=use_ani,
            sigma_scale_factor=0.8,
            relative_permittivity=4,
            water_model=None,
            platform_name=platform,
        )

        if rmol.GetNumConformers() == 0:
            raise Exception("No Conformers")

        rmol.sort_conformers(energy_range=2)  # kcal/mol

    data = {}
    if scoring_function is None:
        cnnaffinities = rmol.gnina(receptor_file=pdb_filename, gnina_gpu=gnina_gpu)
        data = {"cnnaffinities": [float(affinity) for affinity in cnnaffinities.CNNaffinity]}
        score = data["cnnaffinities"][0]
    else:
        score = scoring_function(rmol, pdb_filename, data)

    data["score"] = score
    return rmol, data
