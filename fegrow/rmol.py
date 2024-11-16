

class RInterface:
    """
    This is a shared interface for a molecule and a list of molecules.

    The main purpose is to allow using the same functions on a single molecule and on a group of them.
    """

    @abc.abstractmethod
    def rep2D(self, **kwargs):
        ...

    @abc.abstractmethod
    def toxicity(self):
        pass

    @abc.abstractmethod
    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        pass

    @abc.abstractmethod
    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        pass


class RMol(RInterface, rdkit.Chem.rdchem.Mol):
    """
    RMol is essentially a wrapper around RDKit Mol with
    tailored functionalities for attaching R groups, etc.

    :param rmol: when provided, energies and additional metadata is preserved.
    :type rmol: RMol
    :param template: Provide the original molecule template
        used for this RMol.
    """

    gnina_dir = None

    def __init__(self, *args, id=None, template=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(args[0], RMol) or isinstance(args[0], rdkit.Chem.Mol):
            self.template = args[0].template if hasattr(args[0], "template") else None
            self.rgroup = args[0].rgroup if hasattr(args[0], "rgroup") else None
            self.opt_energies = (
                args[0].opt_energies if hasattr(args[0], "opt_energies") else None
            )
            self.id = args[0].id if hasattr(args[0], "id") else None
        else:
            self.template = template
            self.rgroup = None
            self.opt_energies = None
            self.id = id

    def _save_template(self, mol):
        self.template = RMol(copy.deepcopy(mol))

    def _save_opt_energies(self, energies):
        self.opt_energies = energies

    def toxicity(self):
        """
        Assessed various ADMET properties, including
         - Lipinksi rule of 5 properties,
         - the presence of unwanted substructures
         - problematic functional groups
         - synthetic accessibility

         :return: a row of a dataframe with the descriptors
         :rtype: dataframe
        """
        df = tox_props(self)
        # add an index column to the front
        df.insert(0, "ID", self.id)
        df.set_index("ID", inplace=True)

        # add a column with smiles
        df = df.assign(Smiles=[Chem.MolToSmiles(self)])

        return df

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        """
        Generate conformers using the RDKIT's ETKDG. The generated conformers
        are embedded into the template structure. In other words,
        any atoms that are common with the template structure,
        should have the same coordinates.

        :param num_conf: fixme
        :param minimum_conf_rms: The minimum acceptable difference in the RMS in any new generated conformer.
            Conformers that are too similar are discarded.
        :type minimum_conf_rms: float
        :param flexible: A list of indices that are common with the template molecule
            that should have new coordinates.
        :type flexible: List[int]
        """
        cons = generate_conformers(self, num_conf, minimum_conf_rms, **kwargs)
        self.RemoveAllConformers()
        [self.AddConformer(con, assignId=True) for con in cons.GetConformers()]

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Enumerate the conformers inside of the receptor by employing
        ANI2x, a hybrid machine learning / molecular mechanics (ML/MM) approach.
        ANI2x is neural nework potential for the ligand energetics
        but works only for the following atoms: H, C, N, O, F, S, Cl.

        Open Force Field Parsley force field is used for intermolecular interactions with the receptor.

        :param sigma_scale_factor: is used to scale the Lennard-Jones radii of the atoms.
        :param relative_permittivity: is used to scale the electrostatic interactions with the protein.
        :param water_model: can be used to set the force field for any water molecules present in the binding site.
        """
        if self.GetNumConformers() == 0:
            print("Warning: no conformers so cannot optimise_in_receptor. Ignoring.")
            return

        opt_mol, energies = optimise_in_receptor(self, *args, **kwargs)
        # replace the conformers with the optimised ones
        self.RemoveAllConformers()
        [
            self.AddConformer(conformer, assignId=True)
            for conformer in opt_mol.GetConformers()
        ]
        # save the energies
        self._save_opt_energies(energies)

        # build a dataframe with the molecules
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(energies),
                "Conformer": conformer_ids,
                "Energy": energies,
            }
        )

        return df

    def sort_conformers(self, energy_range=5):
        """
        For the given molecule and the conformer energies order the energies
         and only keep any conformers with in the energy range of the
         lowest energy conformer.

        :param energy_range: The energy range (kcal/mol),
            above the minimum, for which conformers should be kept.
        """
        if self.GetNumConformers() == 0:
            print("An rmol doesn't have any conformers. Ignoring.")
            return None
        elif self.opt_energies is None:
            raise AttributeError(
                "Please run the optimise_in_receptor in order to generate the energies first. "
            )


        final_mol, final_energies = sort_conformers(
            self, self.opt_energies, energy_range=energy_range
        )
        # overwrite the current confs
        self.RemoveAllConformers()
        [
            self.AddConformer(conformer, assignId=True)
            for conformer in final_mol.GetConformers()
        ]
        self._save_opt_energies(final_energies)

        # build a dataframe with the molecules
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(final_energies),
                "Conformer": conformer_ids,
                "Energy": final_energies,
            }
        )

        return df

    def rep2D(self, idx=-1, rdkit_mol=False, h=True, **kwargs):
        """
        Use RDKit and get a 2D diagram.
        Uses Compute2DCoords and Draw.MolToImage function

        Works with IPython Notebook.

        :param **kwargs: are passed further to Draw.MolToImage function.
        """
        numbered = copy.deepcopy(self)

        if not h:
            numbered = Chem.RemoveHs(numbered)

        numbered.RemoveAllConformers()
        if idx:
            for atom in numbered.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
        Chem.AllChem.Compute2DCoords(numbered)

        if rdkit_mol:
            return numbered
        else:
            return Draw.MolToImage(numbered, **kwargs)

    def rep3D(
        self,
        view=None,
        prody=None,
        template=False,
        confIds: Optional[List[int]] = None,
    ):
        """
        Use py3Dmol to obtain the 3D view of the molecule.

        Works with IPython Notebook.

        :param view: a view to which add the visualisation. Useful if one wants to 3D view
            multiple conformers in one view.
        :type view: py3Dmol view instance (None)
        :param prody: A prody protein around which a view 3D can be created
        :type prody: Prody instance (Default: None)
        :param template: Whether to visualise the original 3D template as well from which the molecule was made.
        :type template: bool (False)
        :param confIds: Select the conformations for display.
        :type confIds: List[int]
        """
        if prody is not None:
            view = prody_package.proteins.functions.view3D(prody)

        if view is None:
            view = py3Dmol.view(width=400, height=400, viewergrid=(1, 1))

        for conf in self.GetConformers():
            # ignore the confIds we've not asked for
            if confIds is not None and conf.GetId() not in confIds:
                continue

            mb = Chem.MolToMolBlock(self, confId=conf.GetId())
            view.addModel(mb, "lig")

            # use reverse indexing to reference the just added conformer
            # http://3dmol.csb.pitt.edu/doc/types.html#AtomSelectionSpec
            # cmap = plt.get_cmap("tab20c")
            # hex = to_hex(cmap.colors[i]).split('#')[-1]
            view.setStyle({"model": -1}, {"stick": {}})

        if template:
            mb = Chem.MolToMolBlock(self.template)
            view.addModel(mb, "template")
            # show as sticks
            view.setStyle({"model": -1}, {"stick": {"color": "0xAF10AB"}})

        # zoom to the last added model
        view.zoomTo({"model": -1})
        return view

    def remove_clashing_confs(self,
                              protein: Union[str, openmm.app.PDBFile], min_dst_allowed=1.0):
        """
        Removing conformations that class with the protein.
        Note that the original conformer should be well docked into the protein,
        ideally with some space between the area of growth and the protein,
        so that any growth on the template doesn't automatically cause
        clashes.

        :param protein: The protein against which the conformers should be tested.
        :type protein: filename or the openmm PDBFile instance or prody instance
        :param min_dst_allowed: If any atom is within this distance in a conformer, the
         conformer will be deleted.
        :type min_dst_allowed: float in Angstroms
        """
        if type(protein) is str:
            protein = openmm.app.PDBFile(protein)

        if type(protein) is openmm.app.PDBFile:
            protein_coords = protein.getPositions(asNumpy=True).in_units_of(openmm.unit.angstrom)._value
        else:
            protein_coords = protein.getCoords()

        rm_counter = 0
        for conf in list(self.GetConformers()):
            # for each atom check how far it is from the protein atoms
            min_dst = 999_999_999  # arbitrary large distance

            for point in conf.GetPositions():
                shortest = np.min(
                    np.sqrt(np.sum((point - protein_coords) ** 2, axis=1))
                )
                min_dst = min(min_dst, shortest)

                if min_dst < min_dst_allowed:
                    self.RemoveConformer(conf.GetId())
                    logger.debug(
                        f"Clash with the protein. Removing conformer id: {conf.GetId()}"
                    )
                    rm_counter += 1
                    break
        print(f"Removed {rm_counter} conformers. ")

        # return self for Dask
        return self


    @staticmethod
    def set_gnina(loc):
        """
        Set the location of the binary file gnina. This could be your own compiled directory,
        or a directory where you'd like it to be downloaded.

        By default, gnina path is to the working directory (~500MB).

        :param loc: path to gnina binary file. E.g. /dir/path/gnina. Note that right now gnina should
         be a binary file with that specific filename "gnina".
        :type loc: str
        """
        # set gnina location
        path = Path(loc)
        if path.is_file():
            assert path.name == "gnina", 'Please ensure gnina binary is named "gnina"'
            RMol.gnina_dir = path.parent
        else:
            raise Exception("The path is not the binary file gnina")
        # extend this with running a binary check

    @staticmethod
    def _check_download_gnina():
        """
        Check if gnina works. Otherwise, download it.
        """
        if RMol.gnina_dir is None:
            # assume it is in the current directory
            RMol.gnina_dir = os.getcwd()

        # check if gnina works
        try:
            subprocess.run(
                ["./gnina", "--help"], capture_output=True, cwd=RMol.gnina_dir
            )
            return
        except FileNotFoundError as E:
            pass

        # gnina is not found, try downloading it
        print(f"Gnina not found or set. Download gnina (~500MB) into {RMol.gnina_dir}")
        gnina = os.path.join(RMol.gnina_dir, "gnina")
        # fixme - currently download to the working directory (Home could be more applicable).
        urllib.request.urlretrieve(
            "https://github.com/gnina/gnina/releases/download/v1.0.1/gnina",
            filename=gnina,
        )
        # make executable (chmod +x)
        mode = os.stat(gnina).st_mode
        os.chmod(gnina, mode | stat.S_IEXEC)

        # check if it works
        subprocess.run(
            ["./gnina", "--help"], capture_output=True, check=True, cwd=RMol.gnina_dir
        )

    def gnina(self, receptor_file, gnina_gpu=False):
        """
        Use GNINA to extract CNNaffinity, which we also recalculate to Kd (nM)

        LIMITATION: The GNINA binary does not support MAC/Windows.

        Please cite GNINA accordingly:
        McNutt, Andrew T., Paul Francoeur, Rishal Aggarwal, Tomohide Masuda, Rocco Meli, Matthew Ragoza,
        Jocelyn Sunseri, and David Ryan Koes. "GNINA 1.0: molecular docking with deep learning."
        Journal of cheminformatics 13, no. 1 (2021): 1-20.

        :param receptor_file: Path to the receptor file.
        :type receptor_file: str
        """
        RMol._check_download_gnina()
        gnina_path = os.path.join(RMol.gnina_dir, 'gnina')

        if not isinstance(receptor_file, str) and not isinstance(receptor_file, Path):
            raise ValueError(f"gnina function requires a file path to the receptor. Instead, was given: {type(receptor_file)}")

        # get the absolute path
        receptor = Path(receptor_file)
        if not receptor.exists():
            raise ValueError(f'Your receptor "{receptor_file}" does not seem to exist.')

        _, CNNaffinities = gnina(self, receptor, gnina_path, gnina_gpu=gnina_gpu)

        return RMol._parse_gnina_cnnaffinities(self, CNNaffinities)

    @staticmethod
    def _ic50(x):
        return 10 ** (-x - -9)

    @staticmethod
    def _parse_gnina_cnnaffinities(mol, CNNAffinities, mol_id=0):
        # generate IC50 from the CNNaffinities
        ic50s = list(map(RMol._ic50, CNNAffinities))

        # add nM units
        ic50s_nM = pandas.Series(ic50s, dtype="pint[nM]")

        # create a dataframe
        conformer_ids = [c.GetId() for c in mol.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [mol_id] * len(CNNAffinities),
                "Conformer": conformer_ids,
                "CNNaffinity": CNNAffinities,
                "Kd": ic50s_nM,
            }
        )

        return df

    def to_file(self, filename: str):
        """
        Write the molecule and all conformers to file.

        Note:
            The file type is worked out from the name extension by splitting on `.`.
        """
        file_type = Path(filename).suffix.lower()

        writers = {
            ".mol": Chem.MolToMolFile,
            ".sdf": Chem.SDWriter,
            ".pdb": functools.partial(Chem.PDBWriter, flavor=1),
            ".xyz": Chem.MolToXYZFile,
        }

        func = writers.get(file_type, None)
        if func is None:
            raise RuntimeError(
                f"The file type {file_type} is not support please chose from {writers.keys()}"
            )

        if file_type in ['.pdb', '.sdf']:
            # multi-frame writers

            with writers[file_type](filename) as WRITER:
                for conformer in self.GetConformers():
                    WRITER.write(self, confId=conformer.GetId())
            return

        writers[file_type](self, filename)


    def df(self):
        """
        Generate a pandas dataframe row for this molecule with SMILES.

        :returns: pandas dataframe row.
        """
        df = pandas.DataFrame(
            {
                "ID": [self.id],
                "Smiles": [Chem.MolToSmiles(self)],
            }
        )
        # attach energies if they're present
        if self.opt_energies:
            df = df.assign(
                Energies=", ".join([str(e) for e in sorted(self.opt_energies)])
            )

        df.set_index(["ID"], inplace=True)
        return df

    def _repr_html_(self):
        # return a dataframe with the rdkit visualisation

        df = self.df()

        # add a visualisation column
        PandasTools.AddMoleculeColumnToFrame(
            df, "Smiles", "Molecule", includeFingerprints=True
        )
        return df._repr_html_()