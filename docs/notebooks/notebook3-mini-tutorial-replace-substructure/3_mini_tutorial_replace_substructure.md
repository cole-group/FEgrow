# 3: Mini - replace substructure

**Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

## Overview

In this mini tutorial, we modify the molecule by replacing an -oxazine ring with our own a methyl group.

If you're ready to use FEgrow as inteded, please proceed to learning about the fegrow.ChemSpace class. 


```python
from rdkit import Chem
import fegrow
```

    <frozen importlib._bootstrap>:241: RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher> already registered; second conversion method ignored.
    <frozen importlib._bootstrap>:241: RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterCatalogEntry> already registered; second conversion method ignored.


# Prepare the ligand scaffold


```python
rdkit_mol = Chem.AddHs(Chem.MolFromSmiles('CN1CC=CN(COC2CNCOC2)C1'))
# get the FEgrow representation of the rdkit Mol
molecule = fegrow.RMol(rdkit_mol)
molecule.rep2D(idx=False, h=False)
```




    
![png](output_4_0.png)
    



2D representation of the core. We want the pyrimidine to stay, and the -oxazine including the oxygen on the chain. 


```python
molecule.rep2D(idx=True, size=(500, 500))
```




    
![png](output_6_0.png)
    



Using the 2D drawing, select an index for the growth vector. In this case, we are selecting the hydrogen atom labelled O:7


```python
# you can also embed the information in your scaffold to avoid passing around the index
molecule.GetAtomWithIdx(7).SetAtomicNum(0)
```


```python
# prepare R-group
R_group_methanol = Chem.AddHs(Chem.MolFromSmiles('*CO'))

# use the second connecting point now implicitly
rmol = fegrow.build_molecule(molecule, R_group_methanol)
```

    /home/dresio/code/fegrow/fegrow/builder.py:238: UserWarning: The linking R atom (*) has two or more attachment points (bonds). The molecule might be modified. 
      warnings.warn(
    The R-Group lacks initial coordinates. Defaulting to Chem.rdDistGeom.EmbedMolecule.
    [11:31:56] UFFTYPER: Unrecognized atom type: *_ (0)



```python
rmol
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Smiles</th>
      <th>Molecule</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>None</th>
      <td>[H]OC([H])([H])C([H])([H])N1C([H])=C([H])C([H]...</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
  </tbody>
</table>
</div>



You can now proceed to the rest of the stages, like the generation of conformers, optimisation, etc. However, please checkout ChemSpace for automatic all of it!

