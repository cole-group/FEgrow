# 8: Active Learning - Details

**Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

## Overview
Configure the Active Learning


```python
import pandas as pd
import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace

from fegrow.testing import core_5R83_path, smiles_5R83_path, rec_5R83_path
```


```python
# create the chemical space
cs = ChemSpace()
# we're not growing the scaffold, we're superimposing bigger molecules on it
cs.add_scaffold(Chem.SDMolSupplier(core_5R83_path)[0])
cs.add_protein(rec_5R83_path)
```

    /home/dresio/code/fegrow/fegrow/package.py:595: UserWarning: ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. Use a Dask cluster with processes as a work around (see the documentation for an example of this workaround) .
      warnings.warn("ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. "


    Dask can be watched on http://192.168.178.20:8989/status


    /home/dresio/code/fegrow/fegrow/package.py:799: UserWarning: The template does not have an attachement (Atoms with index 0, or in case of Smiles the * character. )
      warnings.warn("The template does not have an attachement (Atoms with index 0, "
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator
    [13:43:23] DEPRECATION WARNING: please use MorganGenerator



```python
# switch on the caching
cs.set_dask_caching()
```


```python
# load 50k Smiles
data = pd.read_csv(smiles_5R83_path)

# take only 100
smiles = data.Smiles.to_list()[:200]

# here we add Smiles which should already have been matched
# to the scaffold (rdkit Mol.HasSubstructureMatch)
cs.add_smiles(smiles)
```


```python
# configure manually 5 cases
cs.df.loc[0, ("score", "Training")] = 3.248, True
cs.df.loc[1, ("score", "Training")] = 3.572, True
cs.df.loc[2, ("score", "Training")] = 3.687, True
cs.df.loc[3, ("score", "Training")] = 3.492, True
cs.df.loc[4, ("score", "Training")] = 3.208, True
```

# Active Learning

## Warning! Please change the logger in order to see what is happening inside of ChemSpace.evaluate. There is too much info to output it into the screen .


```python
from fegrow.al import Model, Query
```


```python
# This is the default configuration
cs.model = Model.gaussian_process()
cs.query = Query.Greedy()

cs.active_learning(3)
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
      <th>Mol</th>
      <th>score</th>
      <th>h</th>
      <th>Training</th>
      <th>Success</th>
      <th>enamine_searched</th>
      <th>enamine_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>[H]ON([H])C(=O)N([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5bff40&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>[H]OC(=S)N([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5dd540&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[H]OC([H])([H])C(=O)N([H])c1c([H])nc([H])c([H]...</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5bfe60&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
cs.query = Query.UCB(beta=10)
cs.active_learning(3)
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
      <th>Mol</th>
      <th>score</th>
      <th>h</th>
      <th>Training</th>
      <th>Success</th>
      <th>enamine_searched</th>
      <th>enamine_id</th>
      <th>regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>162</th>
      <td>[H]c1nc([H])c(OC(=O)N([H])OC([H])([H])[H])c([H...</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5dfed0&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.01</td>
    </tr>
    <tr>
      <th>170</th>
      <td>[H]c1nc([H])c(S(=O)(=O)N([H])C(=O)OC([H])([H])...</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5e02e0&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.01</td>
    </tr>
    <tr>
      <th>182</th>
      <td>[H]c1nc([H])c([C@@]([H])(C(=O)N([H])OC([H])([H...</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5e0820&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>1.93</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The query methods available in modAL.acquisition are made available, these include
# Query.greedy(), 
# Query.PI(tradeoff=0) - highest probability of improvement
# Query.EI(tradeoff=0) - highest expected improvement
# Query.UCB(beta=1) - highest upper confidence bound (employes modAL.models.BayesianOptimizer)

# Models include the scikit:
# Model.linear()
# Model.elastic_net()
# Model.random_forest()
# Model.gradient_boosting_regressor()
# Model.mlp_regressor()

# Model.gaussian_process()  # uses a TanimotoKernel by default, meaning that it
#                           # compares the fingerprints of all the training dataset
#                           # with the cases not yet studied, which can be expensive
#                           # computationally

cs.model = Model.linear()
cs.query = Query.Greedy()
cs.active_learning()
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
      <th>Mol</th>
      <th>score</th>
      <th>h</th>
      <th>Training</th>
      <th>Success</th>
      <th>enamine_searched</th>
      <th>enamine_id</th>
      <th>regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>[H]ON([H])C(=O)N([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x76de7d5bff40&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.99</td>
    </tr>
  </tbody>
</table>
</div>



### Search the Enamine database usuing the sw.docking.org (check if online)
Please note that you should check whether you have the permission to use this interface. 
Furthermore, you are going to need the pip package `pydockingorg`


```python
# search only molecules similar to the best molecule score-wise (n_best)
# and return up to 5
new_enamines = cs.add_enamine_molecules(n_best=1, results_per_search=10)
```

    Querying Enamine REAL. Looking up 1 smiles.
    Found 10 in 6.730192184448242
    Enamine returned with 10 rows in 6.7s.
    Dask obabel protonation + scaffold test finished in 0.05s.
    Tested scaffold presence. Kept 10/10.
    Adding:  10


    /home/dresio/code/fegrow/fegrow/package.py:1229: UserWarning: Only one H vector is assumed and used. Picking <NA> hydrogen on the scaffold. 
      warnings.warn(f"Only one H vector is assumed and used. Picking {vl.h[0]} hydrogen on the scaffold. ")



```python
new_enamines
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
      <th>Mol</th>
      <th>score</th>
      <th>h</th>
      <th>Training</th>
      <th>Success</th>
      <th>enamine_searched</th>
      <th>enamine_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>200</th>
      <td>O=C(C(O[H])([H])[H])N(c1c(c(c(nc1[H])[H])[H])[...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002350579485</td>
    </tr>
    <tr>
      <th>201</th>
      <td>C(C(=O)N(c1c(c(c(nc1[H])[H])[H])[H])[H])([H])(...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002362554605</td>
    </tr>
    <tr>
      <th>202</th>
      <td>N(C(=O)N(c1c(c(c(nc1[H])[H])[H])[H])[H])([H])[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002540479822</td>
    </tr>
    <tr>
      <th>203</th>
      <td>C(OC(=O)N(C(c1c(c(c(nc1[H])[H])[H])[H])([H])[H...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002472056239</td>
    </tr>
    <tr>
      <th>204</th>
      <td>O=C([O-])C(N(c1c(nc(c(Br)c1[H])[H])[H])[H])([H...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Z2060314917</td>
    </tr>
    <tr>
      <th>205</th>
      <td>O=C(C(O[H])([H])[H])N(C(c1c(c(c(nc1[H])[H])[H]...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Z1551688424</td>
    </tr>
    <tr>
      <th>206</th>
      <td>C(C(=O)N(c1c(c(c(Br)nc1[H])[H])[H])[H])([H])([...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Z1442921413</td>
    </tr>
    <tr>
      <th>207</th>
      <td>C(C(=O)N(c1c(c(c(C([H])([H])[H])nc1[H])[H])[H]...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002273680800</td>
    </tr>
    <tr>
      <th>208</th>
      <td>C(C(=O)N(c1c(c(c(Cl)nc1[H])[H])[H])[H])([H])([...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002441695625</td>
    </tr>
    <tr>
      <th>209</th>
      <td>C(C(=O)N(c1c(c(c(N([H])[H])nc1[H])[H])[H])[H])...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-003001152073</td>
    </tr>
  </tbody>
</table>
</div>


