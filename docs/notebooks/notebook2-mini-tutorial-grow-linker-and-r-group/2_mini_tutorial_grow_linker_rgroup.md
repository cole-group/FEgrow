# 2: Mini - Grow Linker and R-Group

**Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

## Overview

This is a variation of the first tutorial, where in addition to the R-group we also attach a linker. 

If you're ready to move on to the next stage, please proceed to learning about the fegrow.ChemSpace class. 


```python
from rdkit import Chem
import fegrow
```

    The history saving thread hit an unexpected error (OperationalError('attempt to write a readonly database')).History will not be written to the database.


# Prepare the ligand scaffold


```python
rdkit_mol = Chem.AddHs(Chem.MolFromSmiles('CC1C=CN(CN1C)C'))
# get the FEgrow representation of the rdkit Mol
scaffold = fegrow.RMol(rdkit_mol)
scaffold.rep2D(idx=False, h=False)
```




    
![png](output_4_0.png)
    



Show the 2D (with indices) representation of the core. This is used to select the desired growth vector.


```python
scaffold.rep2D(idx=True, size=(500, 500))
```




    
![png](output_6_0.png)
    



Using the 2D drawing, select an index for the growth vector. In this case, we are selecting the hydrogen atom labelled H:9


```python
# you can also embed the information in your scaffold to avoid passing around the index
scaffold.GetAtomWithIdx(9).SetAtomicNum(0)
```

# Create a linker


```python
linker_rcor = Chem.AddHs(Chem.MolFromSmiles('*CO*'))
# note we do not clarify which connecting point * should be used first
linker_rcor
```




    
![png](output_10_0.png)
    




```python
# let us use O as the first connecting point (lower digit)
linker_rcor = Chem.AddHs(Chem.MolFromSmiles('[*:1]CO[*:0]'))
linker_rcor
```




    
![png](output_11_0.png)
    



# Attach the linker


```python
# linker behaves like any other 
# we have to specify where the R-group should be attached using the attachment index
with_linker = fegrow.build_molecule(scaffold, linker_rcor)
```

    The R-Group lacks initial coordinates. Defaulting to Chem.rdDistGeom.EmbedMolecule.
    [11:25:18] UFFTYPER: Unrecognized atom type: *_ (0)
    [11:25:18] UFFTYPER: Unrecognized atom type: *_ (3)



```python
# note that the second connecting point * is left for the future R-group
with_linker
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
      <td>[H]C1=C([H])C([H])(C([H])([H])OC([H])([H])[*:1...</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
  </tbody>
</table>
</div>




```python
# prepare R-group
R_group_cl = Chem.AddHs(Chem.MolFromSmiles('*CCl'))
R_group_cl
```




    
![png](output_15_0.png)
    




```python
# use the second connecting point now implicitly
rmol = fegrow.build_molecule(with_linker, R_group_cl)
```

    The R-Group lacks initial coordinates. Defaulting to Chem.rdDistGeom.EmbedMolecule.
    [11:25:18] UFFTYPER: Unrecognized atom type: *_ (0)



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
      <td>[H]C1=C([H])C([H])(C([H])([H])OC([H])([H])C([H...</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
  </tbody>
</table>
</div>



You can now proceed to the rest of the stages, like the generation of conformers, optimisation, etc. However, please checkout ChemSpace for automatic all of it!

