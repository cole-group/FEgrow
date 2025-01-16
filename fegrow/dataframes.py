

class RGroups(pandas.DataFrame):
    """
    The default R-Group library with visualisation (mols2grid).
    """

    def __init__(self):
        data = RGroups._load_data()
        super().__init__(data)

        self._fegrow_grid = mols2grid.MolGrid(self, removeHs=True, mol_col="Mol", use_coords=False, name="m2")

    @staticmethod
    def _load_data() -> pandas.DataFrame:
        """
        Load the default R-Group library

        The R-groups were largely extracted from (please cite accordingly):
        Takeuchi, Kosuke, Ryo Kunimoto, and Jürgen Bajorath. "R-group replacement database for medicinal chemistry." Future Science OA 7.8 (2021): FSO742.
        """
        molecules = []
        names = []

        builtin_rgroups = Path(__file__).parent / "data" / "rgroups" / "library.sdf"
        for rgroup in Chem.SDMolSupplier(str(builtin_rgroups), removeHs=False):
            molecules.append(rgroup)
            names.append(rgroup.GetProp("SMILES"))

            # highlight the attachment atom
            for atom in rgroup.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    setattr(rgroup, "__sssAtoms", [atom.GetIdx()])

        return {"Mol": molecules, "Name": names}

    def _ipython_display_(self):
        from IPython.display import display_html

        subset = ["img", "Name", "mols2grid-id"]
        display_html(self._fegrow_grid.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        df = self._fegrow_grid.get_selection()
        return list(df["Mol"])


class Linkers(pandas.DataFrame):
    """
    A linker library presented as a grid molecules using mols2grid library.
    """

    def __init__(self):
        # initialise self dataframe
        data = Linkers._load_data()
        super().__init__(data)

        self._fegrow_grid = mols2grid.MolGrid(
            self,
            removeHs=True,
            mol_col="Mol",
            use_coords=False,
            name="m1",
            prerender=False,
        )

    @staticmethod
    def _load_data():
        # note that the linkers are pre-sorted so that:
        #  - [R1]C[R2] is next to [R2]C[R1]
        #  - according to how common they are (See the original publication) as described with SmileIndex
        builtin_rlinkers = Path(__file__).parent / "data" / "linkers" / "library.sdf"

        mols = []
        display_names = []
        smile_indices = []
        for mol in Chem.SDMolSupplier(str(builtin_rlinkers), removeHs=False):
            mols.append(mol)

            # use easier searchable SMILES, e.g. [*:1] was replaced with R1
            display_names.append(mol.GetProp("display_smiles"))

            # extract the index property from the original publication
            smile_indices.append(mol.GetIntProp("SmileIndex"))

        return {"Mol": mols, "Name": display_names, "Common": smile_indices}

    def _ipython_display_(self):
        from IPython.display import display

        subset = ["img", "Name", "mols2grid-id"]
        return display(self._fegrow_grid.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        df = self._fegrow_grid.get_selection()
        return list(df["Mol"])