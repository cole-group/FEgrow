# 7: Active Learning and Enamine

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

    <frozen importlib._bootstrap>:241: RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher> already registered; second conversion method ignored.
    <frozen importlib._bootstrap>:241: RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterCatalogEntry> already registered; second conversion method ignored.



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


    Generated 7 conformers. 
    Generated 5 conformers. 
    Generated 12 conformers. 
    Removed 0 conformers. 
    Removed 0 conformers. 
    Removed 0 conformers. 


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/parmed/structure.py:1799: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
      coords = np.asanyarray(value, dtype=np.float64)


    Using force field


    Optimising conformer: 100%|███████████████████████| 7/7 [00:01<00:00,  6.48it/s]


    Using force field


    Optimising conformer: 100%|███████████████████████| 5/5 [00:00<00:00, 10.19it/s]


    Using force field


    Optimising conformer: 100%|█████████████████████| 12/12 [00:02<00:00,  5.38it/s]
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator
    [13:17:40] DEPRECATION WARNING: please use MorganGenerator


    Generated 2 conformers. 
    Generated 5 conformers. 
    Generated 5 conformers. 
    Removed 0 conformers. 


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/parmed/structure.py:1799: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
      coords = np.asanyarray(value, dtype=np.float64)


    Removed 0 conformers. 
    Removed 0 conformers. 


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/parmed/structure.py:1799: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
      coords = np.asanyarray(value, dtype=np.float64)


    Using force field


    Optimising conformer:  20%|████▌                  | 1/5 [00:00<00:00,  5.66it/s]

    using ani2x


    Optimising conformer: 100%|███████████████████████| 5/5 [00:01<00:00,  3.49it/s]


    using ani2x


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/aev.py:16: UserWarning: cuaev not installed
      warnings.warn("cuaev not installed")
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/__init__.py:59: UserWarning: Dependency not satisfied, torchani.ase will not be available
      warnings.warn("Dependency not satisfied, torchani.ase will not be available")
    Warning: importing 'simtk.openmm' is deprecated.  Import 'openmm' instead.


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    failed to equip `nnpops` with error: No module named 'NNPOps'
    failed to equip `nnpops` with error: No module named 'NNPOps'


    Optimising conformer:   0%|                               | 0/5 [00:00<?, ?it/s]
    Optimising conformer:  40%|█████████▏             | 2/5 [00:06<00:08,  2.97s/it]
    [Aimising conformer:  50%|███████████▌           | 1/2 [00:04<00:04,  4.56s/it]
    Optimising conformer: 100%|███████████████████████| 2/2 [00:05<00:00,  2.95s/it]
    Optimising conformer: 100%|███████████████████████| 5/5 [00:10<00:00,  2.06s/it]


    Generated 2 conformers. 
    Generated 2 conformers. 
    Generated 2 conformers. 
    Removed 0 conformers. 
    Removed 0 conformers. 
    Removed 0 conformers. 


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/parmed/structure.py:1799: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
      coords = np.asanyarray(value, dtype=np.float64)
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/parmed/structure.py:1799: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
      coords = np.asanyarray(value, dtype=np.float64)


    Using force field


    Optimising conformer: 100%|███████████████████████| 2/2 [00:00<00:00,  6.47it/s]


    Using force field


    Optimising conformer: 100%|███████████████████████| 2/2 [00:00<00:00,  9.18it/s]


    using ani2x
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    failed to equip `nnpops` with error: No module named 'NNPOps'


    Optimising conformer: 100%|███████████████████████| 2/2 [00:02<00:00,  1.06s/it]



```python
# turn on the caching in RAM (optional)
cs.set_dask_caching()
```


```python
# load 50k Smiles
smiles = pd.read_csv(smiles_5R83_path).Smiles.to_list()

# for testing, sort by size and pick small
smiles.sort(key=len)
# take 200 smallest smiles
smiles = smiles[:200]

# here we add Smiles which should already have been matched
# to the scaffold (rdkit Mol.HasSubstructureMatch)
cs.add_smiles(smiles)
```


```python
cs
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
      <th>score</th>
      <th>h</th>
      <th>Training</th>
      <th>Success</th>
      <th>enamine_searched</th>
      <th>enamine_id</th>
      <th>2D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[H]c1nc([H])c(SF)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>1</th>
      <td>[H]c1nc([H])c(SI)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>2</th>
      <td>[H]c1nc([H])c(SCl)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>3</th>
      <td>[H]c1nc([H])c(SBr)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>4</th>
      <td>[H]c1nc([H])c(C#CF)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>[H]c1nc([H])c(-n2nnc(F)c2[H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAIAAAAiOjnJAAAABmJLR0QA/wD/AP+gvaeTAAAXDElEQVR4nO3de1RU19kG8OdwE+UiXhAG8AJGq6ASi0QNivGaVK1tFOoySlyxSUxai8SYEqtfsK0aaKzBtBrHeOnYhqaaNClJjRGNUSBGCmhQSUQRRQYYUSBy1WHm/f44w4CoMAyz55D2/S3WLGczM/sFH85ln33OkYgIjNmag9IFsP9OHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsJgQHCwmBAeLCcHBYkJwsOxErUZcHO7caWl59VVkZytXkGAcLDs5cgRbt2LLlpaWlBQUFipXkGAcLPuZMQMbNqCoSOk67IKDZT/h4Zg/Hy++qHQddsHBsqvERJw8iQMHlK5DPA6WXfn54be/xUsvobbW1PLGGwgJwfLl2LcP58+DSNH6bMdJ6QL+56xYgb17kZRkepqRgfx85Odj504A6N0b4eGIiEBYGB59FP36KVhpl3Cw7M3JCdu34/HH4egIAO++izNnkJODzEycOAGdDkeO4MgR04uDgkwhCwvDI4/AxeWBHxsXh+vXsW8fnJr/S594Am+8gdGjBf88D0LWqq+ntDS6dq2lpbyc0tOt/rz/clFRtGZNy9NnniGA3nuv7cu0WkpNpfh4ioignj0JaPlyc6OICIqNJY2Gzp1r+8aICJIk2rKlpcXJiY4fF/PDWMD6JVZ5OWbOxLhx+Oor0x/fsWOIj8fVq1Z+oNFo1Ol0pc3KyspKSkrKy8tLSkqKiopmzpz5/vvvW12t4iZOhK9vy9OkJNTXo08f6PVwdm5p9/ODnx9+/GMA0Otx5gxOnTJ9XbyIzExkZppe6e+P8eMxfjwmTEB4OADMn4+EBERFYeBAu/1YDySRtZuLRUUICsKgQXjlFaxYAQDvvddxsK5fR3k5SkpQXo7KyvLLl3+n1WrLysq0Wq1OpzMYDPd9l4ODg9FoXLZs2e7du62rVlnbtmHIEMyZ07Z91Sr8+c8YM6ZlfRccDEm6/4fcuoW8PGRmIiMDp06hoqLlW8eP4ze/QXQ0MjJgNOKDDwDA2RlHjyIyUsyP1JGubmNt3IgVK7BgAVQqU4tej4IClJairAxarenRHKbWxzQGDMD162+3/jQfHx9fX9+AgABfX19/f3+VSiU/vv3223v27ElJSYmLixut2FaDlYqKsHo1bt/GuXMIDr7rWzodmpqQk4OcHFOLry/Gj8cjj2DiRIwbBw+Plhd7emLSJEyahPh4EKGgAKdOISsLWVkICzO9ZvNmhITg449NyzwlWb0SvXyZALp5k2bPpoULiYj+/ncaNIgqKu7aMmjz1bcvhYTQrFm0dCmtW2d46623/vnPf3755ZfFxcW3b99up7ulS5cCGDVqVH19vdU1K+LJJwmgp5++/3dv3aL0dEpOpuhoGjCg7a8rKIhiYig5mdLTqd1fD0VEUHIyEdHrr1NgIDU0KLyN1dVV4c2bqKzE6NH46CNUVSE+HleuIDQU/fohIAC+vggIgEpl2nTw84Orq5V/AHV1dWFhYRcuXFi5cmVycrKVn2J3n3+O6dPh7o4LF+Dn1/HrS0tNe4gZGcjJQWNjy7fc3PDww6Y1ZmQkhgy5642TJiE6GitX4s4dPPwwlixBQoKSq0IbLLGIKCGBRoygffto0CBbJd6kqampsLBQ/nd2draLi4skSampqTbuRoymJho9mgBKTLTm7Y2NdPIkJSfTU09RUFDbhdmgQRQdTX/8I5WVEbVaYhHR0aPk6UkODkousWwTrIYGeughmjTJxsG6efPmlClT/P39Kyoq5JakpCQA3t7epaWltuxJjLfeIoCGDqXGRht8WnU1paVRQgLNnUv9+rUkLD+f6O5gEdGiRQR8/4NFRJ99ZvozsiGDwfDYY48BmD17ttFolFtmzJgBYNasWQaDwZad2drNm6b/fhGLV4OBzp+nvXtpxQqSfw3bt1NGRssLSkspPp4uX7Z91xayWbCIaP58268Kr1271rdvXwDbt2+XW8rLy318fABs3rzZxp3Z1AsvEEAzZti736++ou3b6Ztv7N1vG9YHq7qaSkpsWMkDyeOirq6uX3/9tdxy8OBBSZKcnZ1PnTpljwo679w5cnIiJyc6e9beXS9fTgBt22bvftuwPlgbNpCLy13HEMRZtmwZgJCQEPNYw4oVKwA89NBDt27dskcFnfTYYwTQSy8p0PX69QTQunUKdN2alcFqbCSVigA6csTU8u23ZDTarKw2amtrR4wYAWDFihXNBTSGhoYC+PnPfy6qV2v94x8E0IABVFWlQO87dxJAy5Yp0HVrVgZr924CaPRoU5iqqsjDg0JCSNziIycnRx5r+Oijj+SW8+fP9+rVC0BKSoqoXjuvvp6GDCGA3nlHmQI++YQAeuIJZXo3szJYY8YQQPv2mZ4mJdljQ3Xz5s0A+vfvr9Vq5ZZt27YB8PLyKioqEtu3xRISCKCxY6mpSZkCcnMJoDFjlOndzJpgffopAeTnZzrIcOcODRxIAH36qY2La8NoNM6ZMwfAlClTmpr/337yk58AiIiIaFLqf7KV4mJycyNJUnIAqbycAOrfX7ECZNYEa8aMu0aT//pXAmjECIHbWGY6nc7X1xdAUlKS3FJZWTlo0CAA69evF959R6KjCaCnnlKyBoOBnJ1JkqihQckyOh2svDySJHJzoxs3TC1jxxJAu3fbuLIH+fTTTyVJcnJy+uqrr+SW48ePOzo6Ojg4HDt2zE5F3E96OkkS9exJV68qWAURmVYgym4ddDpYTz9NAMXGmp4ePWraA7Ln38fKlSsBDB069LvvvpNbXn31VQABAQE3W4/Y2lFTE4WGEkC//70i/d9l/HgCKDNTyRo6FyytllxcyNGRLl0ytcyercBvs7Gx8eGHHwawdOlSuUWv10+cOBHAggUL7FpKM4OB3nmHHnmEusOkHnmizoEDStbQuWC9+ioBFBVlevrtt+TgQL16UfMxYvspKChwd3cH8Le//U1uKSws9PT0BLBb/Fq5vp7UasrObmm5coU0GtHdWuqXvyTgrmPS9teJYNXVmY6qfvmlqeXZZwmgF18UUlmH1Go1gN69e19uPta6b98+AG5ubt8IPlSm0xFAAwdSTY2p5V//Un5HzGzDBgIoPl7JGjoRrF27dA4O9Oijpqc6HfXsSZKk5PHOhQsXAggPD79z547csmTJEgA//OEP25+P2kVysEJCaPVqU0u3CtbevSRJxmefVfJgl6XBampqGjp06JAhsz7+2DQ4uWXLFw891DhvnrDSLFBVVTV48GAA65qPjdXU1AwbNgzAyy+/LK5fOVipqdSjB8lHxrtVsA4f/tzZ2Xn69OkK1mBpsOQpBoGBgfI4ZH19vbe3tyQ5ZGTcc4abfZ04cUIeazh69KjckpWV5ezsLEnSJ598Ytu+btygs2cpJ8cUrOJievFFmjiRDIbuFay8vDwAI0eOVLAGS4Ml73P96U9/kp/u2LEDwLhx44QV1gnr1q2TxxpuNI+tbdy4EcCAAQPK5Hm7Fquvp8JCSk+n/fspOZni4ykmhmbMoOBg8vAwzdgMDW0JVmUleXvTzp3dK1g3btyQtz4VrMGiYGVlZQHo06dPTU0NERmNxpEjRwJ4794zeZWg1+sfffRRAE8++aTcYjAYpk2bBuCJJ54w3n1AoK6urqCg4MSJEykpKVu2bFm1atXixYsjIyOnTn2+V6/2zi8CqHdvCg6mqKiWYBHR3r3k7U0aTTcKFhH17NkTQG1trVIFWBSsBQsWAFjTfIp4amoqgMGDB+v1epG1dUJxcXGfPn0AqNVqueXatWv9+vUDMGPGjJiYmOnTpwcHB/fu3ftBJ5X06TMUIFdXUqkoLIyioyk2lhITSaOhtDQ6d46qq1u6ax0so5EmTaKQkO4VrMDAQAAFBQVKFdBxsC5fvuzo6Ojs7Hyt+ToNU6ZMAbDFPnP8LJaSkgJg2LBh5j1EtVrt0fqMTwBAjx49AgMDIyIioqOjV65cmZSUtG/fvqNHj+bn5zcP43esdbCI6OxZcnbuXsGKiIgA8MUXXyhVQMdnQr/55psGg2HJkiUBAQEAcnJyjh8/7unpKc/q7D4WLVpUWVm5cOFC5+ZrIZw+fbqmpiY8PPwXv/iF+QTr/v37d70vR0cEBbVc12XUKPz61zh0CABKSy06f1A0Pz8/AKWlpYpV0H7uKisr5QHu06dPyy2LFi0CsNo8gNNd3bx5083NTZKk8+fP263Tkyepf3/atMluHT6QfDhVwfNNOriin1qtrq2tnTlzpnxsrqSk5P3333dycvrVr34lPvNdsm3btrq6utmzZwe3uV6CSNevo6oKa9dizx679Xl/KpUKQFlZmWIVtBO6O3fuyKu/Q4cOyS0vv/wygKeUnXBkgcbGRnna1ueff27nrnfsIICcnengQbv2m5eXl5aWZn4qH91atGiRXYtopb1gaTQaAKNGjZL32G/duiXvVXXbk67Mdu3aBWDMmDFGO0w+vMeaNQRQr14tB1XFMRqNaWlpc+fOlSRpyJAh5mm0R44cATBlyhThFTxAe8Hy9/cHsGPHDvlpdXX1a6+9FmWe29BdGY3GkJAQtJr4YPcCTBfs69+fvv1WVC91dXVvv/32D37wA3nN4+HhERsbaz4ZLj8/H8CwYcNEdd+RjoM1depUu1VjEwcPHgTg7+8v9Dh0++7coR/9yHQdok4O/nesrKwsISGhX/OFb1UqVUJCQpsZjkVFRQBcXV3r6ups3L1l2gvW1q1bJUlSfMpvZ02fPh3AH/7wB2XLqKujiRMJoLCwltk1XZSbmxsTE2MeTwkLC9NoNG2GqS9evBgbG+vm5ubg4NCzZ09/f3+NRmP/TYIOhhvi4+MBDBw4UKkpv52Vl5cnSZKHh0eVImeL3q2igoYPJ4CmTevgsmntMxgMqamp8tVQADg4OMydOzej9TVAiIgoLS1tzpw5Dg4OACRJmjx5srxJACAyMtI8YGQfHQRLr9dPmDAByk357ayYmBgAcXFxShdiUlhIPj6mU3esWGrcunVLrVabN6Q8PT1jY2Ov3n22xu3btzUazZgxY+TX9OjRIyYm5ty5c0RkNBo1Go18DRUHB4eYmBidTmerH619HR/SMU/53bNnjx0K6gqtVuvi4uLo6Gi+Vlt38J//kLs7+fgYNm7sxOTly5cvx8fHywdAAQwdOjQ5ObnNQWWdTpeYmChvCgPw8fFJSEiouGeeeG1tbUJCQo8ePQB4eXklJibaYevTooPQ8riDHab8dpG84v7Zz36mdCFtffaZPjBwGlrNO2pHdnZ2TEyMU/MBo4iIiP3797c5HffChQuxsbHyFAYAY8eOVavVDe2eKVVQUDB37lz59cOHD//3v//d1Z+qXZbOx1q8eDHET/ntitraWnlH6eTJk0rXch/vvvuuvCd04AFnzzQ1NaWmpsrz3gC4uLhER0e3GTJsPWpl3thqPS7aobS0NPOhiLlz514yn25la5YGyzzlt9seJdy6dSuASZMmKV3IA8nTD11cXI6Yr9HT7M033zSv0QYMGPDaa6+Vl5e3fkFjY6NGozFnwt3d/fnnn7duBXL79u2kpCR53kePHj02/+UvdQKujdiJkynETfntOnlKPoAPP/xQ6VraExsbC6B3795nzpxp3f7ss8/Ka6jk5OQ2I09tRq38/PwSEhIqKyu7WElFRUVsbKyjk1N0VtbjX3/9QUWFbcPVufMKN2zYAKum/Ip24MABAEFBQd3h0iDtMBgMUVFR8vht6527goKCw4cPtxltsmTUqovOXLq09JtvwrKzw7Kzn/nmm/O2G03tXLDamfKrLHnTZJviF0i0QENDw+TJkwEEBwffd8Fz31GrTGEnzBuJ0ior5+TlhWVnj8vO/r+iohvNMyW7otPXbigpKZEXy1u3bu169zaRmZkJoG/fvgpO8e6U6upqedgpMjKy9a6cJaNWgtQbDGqtdmJublh29qTcXLVWe7trCw5rLmP04Ycfytt9dh7MfZD58+cDWLt2rdKFdEJJSYl89aV58+Y1NTVZMmplB8WNjfGFhfKa8cmzZ0+0nuffSVbe8uSFF15Qq9UjR47Mzs6Wr9eolKKiomHDhjk5ORUVFanMd4r6Pjh79mxkZGR1dbW7u3t9fb3RaAQwbdq0uLg485EZRWTV1Pzx2rXChgYAj3h6rg4ICOrZ84OKiv2tbzgG/KP9GZTW5bGhoUFemC9fvtzqUNuEfPnkZ555RtkyrHP8+HEXFxcAkiRFRUV1n4lueqPx3fLyx06fDsvOnpCToy4tVZeWLs7PP11TY/5q/xOsvxz3uXPn5JFfBc8uNE/Jb7P3/j2SkZGxZs0a8yXsu5Xv9Po3iovDc3I0ZWXq0tLlFy5Y/l7rg0XNY5JeXl5XrlzpyudYbdOmTQAef/xxRXr/H3Ghru620djZYFl/Wzl5NfrTn/40NTV18uTJx44dc5Tv4CtAU1OTTqfTarXBwcHyIgqAXq8PCgoqKSk5fPjwzJkzBXXNZDvLynaVljo3b/nF+Pi80O5pbl26w6okSbt27QoNDU1PT09MTFy7dm1XPq2qqkq+FfS9j8XFxU1NTQBOnDghDwIBSElJKSkpGTVqlHnIhwk1yt09MTBQ/rdbRwuRrt6619vbOyUlZfr06evXr586dap8DYUHuXHjhnzzcPlRp9Ndu3ZNfrx+/bper3/QGx0cHFQqVZudPvl2mKtXr5YedBdlZlMukjTAxcXCF3dpVWj2yiuvbN68OTAw8PTp062vj7Bnz55Dhw6Z70rf2PqGoffw9vY2n6/c+tHf39/Hx8c8jUSWlpY2a9YsHx+fK1euuFp921ZmsZ1lZbk1NTuGD7fw9V1dYsk2bdqUnp5+6tSp5557bv/+/eb23Nxc+SiezNXV1c/PT6VS3fs4ePBg88aTJVatWgVg5cqVnKruyTZLLACFhYVjx46tqanRaDRPP/203JiVlVVYWGhe8Fg9lKrT6eQ1Znl5+aVLlw4cOFBYWChJklar/X4Nin5/5dfVVej1U7y8LHy9zYIFYO/evcuWLXNzc8vNzR1u8TJT1tDQcN/N9tLS0qtXr9bV1d37lpkzZx4+fNhGtTMbs2WwACxevDglJSU0NDQzM9PNza31t2pra82b7eXl5VqttqyszPzY/uZXv379VCqVv7+/vOF18eLFCRMmxMXF8WZ7t2XjYH333XcjRoyQ764bHh7u5eVl3u+771LHzMPDw7zGlDPk10ylUvGG1PeOjYMFYNeuXc8999y97a6urn369Gmz2R4UFCTHyMvilTf7XrB9sAAcPHjw9ddf9/HxmTdvnnnBw9H5nyIkWIwpNumH/XfjYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMiP8Hb4efnJc8fMUAAAGSelRYdHJka2l0UEtMIHJka2l0IDIwMjQuMDkuNgAAeJx7v2/tPQYgEABiJgYI4AFiXiBuYGRzsADSzIws7A4aIAYzXIANIgCXQKdh8pwMCkCaEUkfOoObgTGDiZEpgYk5g4mZRYGFNYGVLYOJjT2BnYOBnRMoxpXAxZ3BxM2QwMmS4ARyIxsDNxczEyMrKwsnO5v4QpD5DDCnP80rOVD6qGEfiOMq7nvA61WfPYi9X+v2ft3wRWB25prQ/VyqIbYgtk60hn1kQYAdiM28d4v9acYDYDXiJhYOOSXTwezOV44OFbbzwGaaFHU5nJbr2A9i+2tfsS+VzwKzS1t37t/fuALMnu9ic2DRu91gtj0kUIVSQOYz8APJ+Pjk/NyC0pLUlICi/IJiNlioswNxXmmuY1F+LlhZcElqUWp+ckZqrkt+XiqSLCNIBxYljKBQELa6wASKSWw2wUKJAyTn7BkQlJiXjcxmYmZgIFsvOwV6WSjQy0SBXk4K9HJRoJeDAr1sFOjlpkAvKwV6GSjQC1YgLAYAJxTlrZyGvmcAAAFpelRYdE1PTCByZGtpdCAyMDI0LjA5LjYAAHicfVNLagMxDN3PKXSBGEmWf8tMJi2lZAJt2jt03/vTp4TEEzD1jECWn35P9kS+Ppb3n196LF2miYj/+Vtr9B2ZeTqRKzQfX99WOlz2891yOH+tl08SJYnwwfeM3V/Op7tF6EC7GKo0zZV2HDRJ0Uoc+Lq6r9LqSGVLlkiCSE6aBsDoISUUUTWUEBBY6whoDuRQo+Wsfo7ULQ6ACak55FSyCRTlmJINcBk4CSbKUqEkHKZR4gKchloq8nknNarxAFdRYAysreEYlUZunEcFNnohC1GzoUD0DpQNIwqmg4xoJRX1kE1A5yikyI1GsxbBHrQo2fIwqDpUQzPJtV6dQLmOoMd1eRr/7ULM53XpF8Ig2qdukNhnay59gg5PfU7YUO7jMEjprBukdnIF0jqF5qYtU+bxZUOIb9Htpm+73vKHwcuDY9q2u23O9/eXAn36Awz/p7kapfG5AAAAtHpUWHRTTUlMRVMgcmRraXQgMjAyNC4wOS42AAB4nB3OS44DMQgE0KvMMpEcRBUfG/U+F/G+T5DDD+4deqgKvhv7fn0297335vu+8fd7uRjTfXwgmtbDZaKsalExrcZxQTJmzEMFD2/SQ+kYKlSL6GBvl3km2zQwy9AImSB9QJiL1WKd8IgWIIOPLBSffnaQq41SjqzzmHeQT5V72UOG9DyPOajsqlDVGBdlzYV1upf11ffvH72cMyl6lfo6AAAAAElFTkSuQmCC" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>196</th>
      <td>[H]c1nc([H])c(-c2nnn(F)c2[H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>197</th>
      <td>[H]c1nc([H])c(C([H])([H])C#N)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>198</th>
      <td>[H]c1nc([H])c(C(=O)N([H])C#N)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>199</th>
      <td>[H]c1nc([H])c(N([H])C(=O)C#N)c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAIAAAAiOjnJAAAABmJLR0QA/wD/AP+gvaeTAAAVKElEQVR4nO3deVSU9f7A8c8wMMAgoCgoqKmUgVu/FC8WeIhloI5CaR3O1auZegRT1LIUrmaaWl6se+8p09ulkwZWpkCiArLNSC5EJKm5HPUqLmGishmbDLN8fn98h4dhZ2C+M6N+Xqc/ep4zMN+R9zzz7CNCRCDE2KzMPQDyaKKwCBcUFuGCwiJcUFiECwqLcEFhES4oLMIFhUW4oLAIFxQW4YLCIlxQWIQLCotwQWERLigswgWFRbigsAgXFBbhgsIiXFBYhAsKi3BBYREuKCzCBYVFuKCwCBcUFuGCwiJcUFiECwqLcEFhES4oLMIFhUW4oLAIFxQW4YLCIlxQWIQLCotwQWERLigswgWFRbigsAgXFBbhgsIiXFBYhAsKi3BBYREuKCzCBYVFuKCwCBcUFuGCwiJcUFiECwqLcEFhES4oLMIFhUW4oLAIFxQW4YLCIlxQWIQLiwursLDw7t275h4F6SuLCKuioiIlJWXx4sXjxo3z8/Nzd3ePiooy96BaUavhyBFoaGiZU1MD+fnmG5DlQzOpr6/Pzs5etWrVxIkTraxa+ra3twcAkUi0e/duc42tvaoqBMCLF1vmFBSg+f7xHgLWpoxYo9GcOXNGLpfL5fITJ040NjYKMfn7+8tkMplMNnHixKioqF27di1YsAAAXn/9dVOOkBiLKcK6cAEUCjh16kZa2v/V1NSwmWKxeMqUKTKZLCQkxM/Pz9bWVnj8zp07x4wZs3r16gULFohEorlz55pgkMS4DAirrg4WLoSICBAWItnZcOIEfPhhBw++cweOHwe5HLKyoLQUAEAkGuHiYuvp6SlrNmDAgM6ea9WqVYgYGxs7f/58ALCQtvbsgcGDdf9//bpZh2LxDAirqQlSUiA7G4KCYNgwAICrV+HIkZYHVFfDjz+CXA4KBVy+3DLf3R1CQiAkRBQWdsXDw7nLp2hKTEyMiooSiUSrV69GxLi4uPnz54tEojlz5hj40ozv8mW4c0f3/7Tl2o2er45VViIAvvIKzpypm/P55/j881hSgmvWoK8visUIoPvPyQkjIvCzz/D8eQPW+CIjIwEgJiZGq9WyORs2bAAAsVj83XffGfCLjI1W3g1lcFjnzuHAgXjwIGJzWD/9pIvJ2hp9fDAuDvPyUKnszWhyc3Pt7OwAIDo6un1be/bs6c0vNQYKy1AGh/XHH7htG44YgXV1urBUKoyLw6wsrKszwoBycnJYW2+99ZbQ1vr1683bFoVlqN6EpVbjxIm4Zo0uLKMT2nr77beFme+//z5r6/vvvzf+U3anpgb9/PDmzZY5586hv7/pB/LQ6E1YiFhYiPb2+O67XMJCxOzs7PZtrVu3DgBsbGz279/P5Vm79MMPePp0y+StW5iRYfpRPDR6GRYiLlqEUimvsBAxKyuLtbVy5Uph5nvvvcfaSktL4/XEnZBKcfBgrKrSTaal4fDhJh7Cw6T3YVVW4qBBHMNCxKysLLbj9J133hFmCm0dOHCA43O3I5WitzcuXqybpLC6ZkBYjY2YkID19S1zTpxA3n/cw4cPt29r7dq1ACCRSEzZllSKe/eigwP+9BMihdUdwzZsTp3CWbPQxGs4mZmZrK13331XmLlmzRrW1kG254M/qRR//hnXr8dnnkGVisLqhmFhbd6MALh0qW4yPR3T042zl6FrQlvsUA/z97//nbV16NAhTs/b0IB5efif/yA2h9XQgJ6e+K9/UVjdMCysgAAEQGG92ccHATA31/jDau+HH36wsbEBAHaoh4mLizN6WxoNFhdjfDzKZGhnhwAokWBtrS4sRMzIQGdnTEigsLpiQFh1dSiRoLU13r+PiFhRgVZWaGeHDQ28BteG0FZsbKwwMzY2lrWVnp7el19+6RLu2IGvvooDBrQcmLKywsmTMS4OKypawkLEGTNw1KhHNqx793DvXoyKwoULdXO8vPDJJ1v+0MXF3e8cNiCs9HQEQD8/3eT33yMAhoYaNug+Sk1NbdOWVqtdsWIFayvDwD1LZWXKb7/F+fNx+PCWmABw9GhcsgRTU7GysuXB+mH9/js6ODxSYdXXY14exsWhjw9aWen+Hezs8MEDREQvL3R0xHXrdA82clgrViAAbtigm1y4EAHw448NfQl9lZKSYm1tDQAffPABm6PVapcvXw4Atra23bZVV1eXl5cXFxfn4+MzefJ7QkyurhgZiQkJeP16xz84fXqrQzr//S/Ono3V1Rga2hLcw0WlwoIC3LgRAwJQIml5X9nbY2goxsfjyZOo0SAiennhBx+gVKr7FzByWN7eCIAFBbrJJ55AADxzpjcvqY+EtjZu3MjmaLXaZcuWsbYyMzPbPL6pqenYsWPr16/39/dnP8i4uY0MD9f++9949iw2H5Y0zPvvIwA6O2NRUR9fk+mcP39+586rERHo6NgSk1iMvr64di0qFLqllD4vL0xNxZgYDAxErdaoYZWWIgA6OmJTEyLixYsIgIMH9/Lv0XfJyckskU2bNrE5Wq02JiYGAOzt7fPy8hCxpKQkISEhMjLS2bnlJDCxWOzj4xMXF5eXl9fY2NjHYajVOGfOQ9DWnTt3kpOTo6Ojhw8fDgBTpmxhPXl6YnQ0Jie3+tAXaLV4+TJic1hVVejmhklJRg1r504EwBkzdJPbtiEAzpnT0xfGw759+1hbmzdvZnO0Wu3ixYvZ+tbAgQOFmEQi0YQJE1auXJmRkVFbW2vcYajV+Le/WWJb1dXVaWlpMTEx3t7e+mfgeXh4REevTkzEW7c6/sHbtzE5GaOjcehQFIuxqkoXFiJ+/TW6uaFCYbywZs1CANyxQzcZEYEAmJjYw5/mpcO2AgMDhw4dCgBDhgyJjIxMSEgoLS3lOgy1GmfPRgDs3x9/+YXrU3VDpVIVFxfHx8fLZDK2lcM4ODjIZLL4+Pji4mJtR58y5eW6mJ58stV2zIgReOpUS1haLU6dimFhRgpLo0FXVwTA//2PjR6dnBAAOf+9emT37t1isVj/QPW0adMA4GPTblao1br3nnnbio6OFmKytbUNDAzcvHlzYWGhWq1u/2C2+7fNliBb4ZHJMD4ei4t1jxTCQsTz59HGxkhhsc/UESN0k0ePIgCOH9+THzWFkydPCv+vVCr79esnEolu375t4mHot6U3IpPat2/fxIkTV69enZ2dXa9/WLeZWq0uLCz88MMP58z5p61tB1uCxcW6LUF9+mEh4sqVRgrr00//FIsxKko3uXXrXakU9U6UsiD5+fkA8Mwzz5jl2ZuacMYMXVvC290SCNsxwpVREkm/fv20wqnk7bcEEbGpCY8fxw0bWk5p6bkehRUcHDxgwJMHDujehlOmTLG1dcrJMdO7skvsxAf9w9UmJrQ1YICZ2/rjjz+SkpLmzZvH1jgFXl5eMTEx+/fvr6rqeKO4pAQTEjAyEp2ddcuzXqxMdx9WfX29ra2tWCyurKxExOrqarFYLJFI6kxw8NlwkydPBoDs7GwzjkGpxIgIdHCoDQiYefbsWVM+9c2bN9evX79gwYIxY8box+Tu7j537tzExMTOtmNu3MCdO3H2bBw8uNXK+/jx+PbbeOqUwSPpPqzDhw8DgK+vL5tMSUkBgODgYIOfir+KigorKys7O7sOVy9MSanERYvWAoCrq+u5c+e4Ppf+lqBIJOr5lmBFxf2UFHzzTRw9ulVMw4fjggX47bdYVtb7UXV/wWpeXh4AhIWF6U+GhoZ2+4Omp1AotFrt1KlTpVKpeUcikcCOHRvu3Tt/6NCh4OBghUIxYcIEI/5+rVb722+/yeVyhUJx/Pjxhub74FhZWfXv3z8sLGzFihW+vr76hxmYBw8eFBQUsNtn/PbbOSenB1VVVgDQrx889xzIZCCTwaRJoNdnb3Wb3vjx4wHg6NGjbNLT0xMAii1q1bTZokWLAGDr1q3mHoiOUqmMiIgA4y23SkpKkpKSoqOjPTw89P+Inp6e0dHRycnJ99mZJ60JW4JBQUH698iws7NbuLDkH//AX37BjnZH9Ek3YZWVlYlEIgcHB3b048qVKwAwcOBATftNUgswcuRIADjVizUCbpRKZXh4OAC4ubmdN+iq8Gbl5eXsaMyoUaP0Y3J3d2e7f291sge9/ZYgW6T5+PisWLEiOTm5pqamby+uK92ElZiYCADh4eFscseOHQAwa9YsfgPqtUuXLgHAoEGDLC36XrRVX18vnIKhf/OwQYMGRUZGfvrpp11/Yly9erXNIm306NFLlixJTU2t7PCgIAfdrGO1WaOy5BUsYV1Q/y9hCSQSSWpq6muvvZaZmRkcHJyfnz927Nj2D9O/edjx48eVSiWb3+bmYT15dSNGjKirq3N1dQ0MDJTJZGFhYWxZblJdRKfVat3d3QHg4sWLiKhSqdhpAjf1rwi2GC+//DIAfP311+YeSMcaGxvZsSY3N7cLFy4I84UPrP79+wt/FP1TMB50uO+yOzdv3uxwS9BkugrrzJkzADBs2DA2WVBQAADe3t4mGZhhVCqVk5MTAPA+3twX+c03LXV1dX311VenTZs2ZMgQ/Tf52LFjly9ffvDgwT///NPcg+2rrj4Kc3Nz4SHZ0VBYWFhTUzNu3Lhh7M5dFuno0aMAMGzYsLKysv3797OZzs7OYWFhMpnspZdeeuKJJ8w6QGPqKqxDhw6BXklxcXF+fn5t1gothCVHL2CD3Lp168GDB4uKimpra6uqqtLS0oKCgsw9NA66WJo5OjqKRCL9M1Is1nPPPQcA7U9Kthw1NTU2NjbW1tbsY+7OnTsikUgqlfb9LFbL1NUmhp+fHyJ+/vnn7KiOxbp///7JkyclEklAQIC5x9IphUKhUqmef/55ti7ITp5+4YUX9PdYPkq6Cis7O3vu3LlqtXrmzJnp6ekmG5OhFAqFRqPx9/fv16+fucfSqYdox41xdLtMM9YVofyw89y3bNli7oF0ZfTo0QDwc/OVYmwjo3f74h8K3YclXLXXiytCTYMdvjxprrM2e+DGjRsA0L9/f3aK8Llz5wDAw8PDvLuauOp+N65IJPrss8+WL1/e1NTEdh/zW3z2wtWrV69duzZw4MBJkyaZeyydysnJAYCQkBCxWAzN+3FCQ0NFRjiLwEL16OgHa2vZsmVKpTIyMlIul/MeVs+xlRWZTGZpR3L0PXYrWGDIjX/1rwiVy+X8lqIGmTlzJgB89dVX5h5Ip9RqtYuLCwCUlJQgolKpdHBwEIlEZX05j87iGXYbI61Wu3TpUgCQSqUKhYLTmHpOrVazc0Kud3bHBQtQVFQEAE899RSbVCgUAPDss8+ad1S8GfbxIRKJtm/fvnTp0oaGhoiIiCP6X3hiDkVFRdXV1V5eXmY4et9jZ44ehcftc7AXX4TJ2lqyZAlrK99U3wZZXl6+b9++qKgodmicaXPatGWKzsxUe3ismz6dTQpr7mYdFH+9W9Bptdo333wTAKRS6ZEjR4y7FBV0eL7bRx99JDzA398fAPjdKtII6urQ1lZ3CwRELC+vHzv2mEzWYLLb1ZlJ77+1Q7gDh1Qqzc/PN9aAVCpVQUHBpk2bAgICJBKJ8AaQSqUvvvjiJ598cunSJfbINkffLFSHt6sLCzPrmEyh91+EKRKJvvjiC0T88ssvw8PDMzIyAgMDe/3brl27xk6ezMvLu3//PpvJzndjJ09OnTqVfZ+AYPv27SqVatKkSezom4XKywMAED742kw+wvoYpkajeeONNwDAwcHhxx9/NOhny8rK9G/aJBCuOakSvgdCj1arPX369NKlS9nyzDKvcGxhOberMy0jfIGVRqOZN28ea0u4SqwztbW1wmqT/n7nwYMHs2tOOjvv+fbt26xC/QvGpVLpiRMn+v4SeGG3q3NyspTb1ZmQEb4T2srKateuXYj4zTffRERE5OTksLOjBI2NjQUFBQqFQqFQ/PrrrxqNhs13dnYOCgoKCQkJCQlpc0k4U1FRceTIEYVCIZfLr127JswfOXJkYGCgo6PjsmXLnn766b6/BF5ycgAAgoOB3aoqNxcAIDTUGNeDWjxjFapWq9k3Nzs5Of3c+m6vxcXFwtNZW1sLlwk0sfdxaw0NDR1e+eTo6ChcMG6sMXP317+2ul1deDgCYFKSWcdkIsb8LkehLWdnZ/22NBpNUFBQbGxsTk5OZzdtEu4+oL+Gbm9vL8RkaVcLdk+4Xd2VK4iITU0WdLs6/oz8JaFqtZp9K7izs3NRd3fk7PBS3b5f+WQp2O3qRo7UTVra7eo4M8I6lj6xWJyUlISIe/bsCQsLy83N9fX11X/A3bt3jx07JpfLs7Ozf//9d2G+p6enrJl+Zw8xtkYlHBV4fHY0MDxqVavVs2fPBgAnJ6fdu3dXVFR0uCXo5ubGtgRv3LjBYxhmFhSEAJiSopv09UUAPHzYrGMyHREi8uhVpVK98sorWVlZbeY7OTkFBgaGhITIZLIOrzR/RDQ0gIsLqNVw7x64uEB1Nbi6glgMVVXg4GDuwZmCkT8KBTY2Nqmpqd7e3qWlpQAwfvz46dOny2SyNgdqHlm2tpCTA+fPg4sLAIBCARoNvPDCY1IVAPBaYglOnz49dOhQNzc3rs9i6YqL4csv4S9/gagocw/FRLiH9ThSKmHjRti7F27dAldXCA+HLVtA75syHgeWe574Q2z2bNi/H776CsrK4MABuHABZDJovi3RY4KWWMZWVAR+fnD2LIwbp5tTVQWjRsG2bfDGG2YdmUnREsvYfv0VvL1bqgIAFxcICQFTnWprISgsYysthdZ3vQIA8PCA8nJzjMZsKCxjc3ODqqq2MysrwZLPRuSAwjK2MWPg8uVWyyeVCgoLofWhrUcehWVsMhmMGgVLlgC7qb9GA2vWQH09zJtn7pGZFK89748va2vIyIBZs8DDA7y84Pp1GDAAMjMft/1YtLuBm6tX4eZNGDIExo59LE4ZbY3CIlzQOhbhgsIiXFBYhAsKi3BBYREuKCzCBYVFuKCwCBcUFuGCwiJcUFiECwqLcEFhES4oLMIFhUW4oLAIFxQW4YLCIlxQWIQLCotwQWERLigswgWFRbigsAgXFBbhgsIiXFBYhAsKi3BBYREuKCzCBYVFuKCwCBcUFuGCwiJcUFiECwqLcEFhES4oLMIFhUW4oLAIFxQW4YLCIlxQWIQLCotwQWERLigswgWFRbigsAgXFBbhgsIiXFBYhAsKi3BBYREuKCzCBYVFuKCwCBcUFuGCwiJcUFiECwqLcEFhES4oLMLF/wPwgeMOuREV4gAAAYJ6VFh0cmRraXRQS0wgcmRraXQgMjAyNC4wOS42AAB4nHu/b+09BiAQAGImBgjghuIGRjYHCyDNzMjC7qABYjDDBdggAizsDBABZjYGiAAHhGaC8JmACsA0sk4og5uBMYOJkSmBiTmDiZlFgYVVgZVNg4mVXYGdQ4OZmTOBkyuDiYshwYkRqJyNgYuTmYlRfCqQzcgAc+ea8wkHau4v2gfiSK50PxAyeYI9iN25+er+fysOgtncR2T2Xz771A7E9qgMs7cVCgWLa/R9sg9NzgXrZXI8YD/v+7X9IHamWJjDL09psPr7RTMdLv3lAKvXXt+wv3Z9PVjNKzXZAxlvN4DZ9pBQE0oBeYqBH0jGxyfn5xaUlqSmBBTlFxSzwYKVHYjzSnMdi/JzwcqCS1KLUvOTM1JzXfLzUpFkGUE+xKIELC5sNYUJFFXYbIKFDAdIztkzICgxLxuZzcQM9i15ejko0MtCgV4mCvSyU6CXjQK9XBToZaVALycFehko0AtWICwGAGBx1x4JbOtxAAABTnpUWHRNT0wgcmRraXQgMjAyNC4wOS42AAB4nH1SW24DIQz831P4AkH4BfgzyUZVVWVXatPeof+9v2pvlbJRUQFLBg3DeMwEMV7nl88v+B00TxNA/meZGXxwznm6QiRwujw9L3C+HU/3k/P6vtzeADFWjvmIPd7W6/0E4QwHTopUqsIhJ8aGcS3lbfS7BEsgEa1YAUxIJp78BXJQYirFVMiBzm01D4ASwJwKEpcgEmmNR4zqT+fUuNbWtoQYR4TFCTG10MXB7LBGMgBWWENYRmuyaa0mTANgc0ZOLIoYL6NiERvgzBVKqqZUg0bZi+YBzmVv5uSsLCU0mrVaR37j1hpKoqp+7pe4EYkOoJdlfmjqT5tP6zL3NosH9V6KB/eOSUTvS8C1u48epZtMHrVbib5t3TD2sO6LBN2+/th6cbsyZfuq+2L20mN//92eT9/qCZocy2q1lQAAALB6VFh0U01JTEVTIHJka2l0IDIwMjQuMDkuNgAAeJwdjjsOw0AIRK8SKY0tEbT8QVYq984hts8JfPiw6YbHMMP1PM/t/dmvSfM756THvSlGGQcMNDFigUNQtFUTMnItOAiztBxeA1koWReyQZXwIvQo7bOBKRGZsETb4Gi7d6J7I9VM8Wbt9zKFDiCuaCJI1IKwZ/vPvfFcbV1GtJIY1cxWmySz/XPG6I+XqyojCPb7B+BkLh80rhORAAAAAElFTkSuQmCC" alt="Mol"/></div></td>
    </tr>
  </tbody>
</table>
<p>200 rows × 8 columns</p>
</div>


# Active Learning


```python
# There is nothing to train the model on, so initially "first_random" is used by default
random1 = cs.active_learning(3, first_random=True)
random2 = cs.active_learning(3, first_random=True)

# note the different indices selected (unless you're lucky!)
print(random1.index.to_list(), random2.index.to_list())
```

    [149, 49, 151] [160, 112, 153]


    /home/dresio/code/fegrow/fegrow/package.py:1284: UserWarning: Selecting randomly the first samples to be studied (no score data yet). 
      warnings.warn("Selecting randomly the first samples to be studied (no score data yet). ")



```python
# now evaluate the first selection
random1_results = cs.evaluate(random1, ani=False)
```


```python
# check the scores, note that they were updated in the master dataframe too
random1_results
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
      <th>149</th>
      <td>[H]c1nc([H])c(OC([H])([H])F)c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x72731d573f10&gt;</td>
      <td>3.283</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49</th>
      <td>[H]c1nc([H])c(OC(F)(F)F)c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x72731d573600&gt;</td>
      <td>3.291</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>151</th>
      <td>[H]c1nc([H])c(C([H])([H])SF)c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x72731d54ee80&gt;</td>
      <td>3.856</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# by default Gaussian Process with Greedy approach is used
# note that this time 
greedy1 = cs.active_learning(3)
greedy2 = cs.active_learning(3)
print(greedy1.index.to_list(), greedy2.index.to_list())
```

    [113, 168, 191] [113, 168, 191]



```python
# learn in cycles
for cycle in range(2):
    greedy = cs.active_learning(3)
    greedy_results = cs.evaluate(greedy)
    
    # save the new results
    greedy_results.to_csv(f'notebook6_iteration{cycle}_results.csv')

# save the entire chemical space with all the results
cs.to_sdf('notebook6_chemspace.sdf')
```


```python
computed = cs.df[~cs.df.score.isna()]
print('Computed cases in total: ', len(computed))
```

    Computed cases in total:  9



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
      <th>regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>[H]c1nc([H])c(OC([H])([H])I)c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731c1db4c0&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.718</td>
    </tr>
    <tr>
      <th>197</th>
      <td>[H]c1nc([H])c(C([H])([H])C#N)c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731d552c70&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.633</td>
    </tr>
    <tr>
      <th>189</th>
      <td>[H]c1nc([H])c(OC([H])([H])Cl)c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731c1dbed0&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.704</td>
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
      <th>33</th>
      <td>[H]C#CSc1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731d687920&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.137</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[H]c1nc([H])c(SC#N)c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731d687a00&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.087</td>
    </tr>
    <tr>
      <th>54</th>
      <td>[H]C(=O)Sc1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731c1d87b0&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.087</td>
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
      <th>177</th>
      <td>[H]c1nc([H])c(Sc2nnn([H])n2)c([H])c1[H]</td>
      <td>&lt;rdkit.Chem.rdchem.Mol object at 0x72731c1db990&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>1.8</td>
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
    Found 10 in 6.407189130783081
    Enamine returned with 10 rows in 6.4s.
    Dask obabel protonation + scaffold test finished in 0.06s.
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
      <td>C(SC(c1c(c(c(nc1[H])[H])[H])[H])([H])[H])([H])...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002558062946</td>
    </tr>
    <tr>
      <th>201</th>
      <td>C(SC(c1c(c(c(Br)nc1[H])[H])[H])([H])[H])([H])(...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Z3340872668</td>
    </tr>
    <tr>
      <th>202</th>
      <td>C(SC(c1c(c(c(C([H])([H])[H])nc1[H])[H])[H])([H...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002903174203</td>
    </tr>
    <tr>
      <th>203</th>
      <td>C(SC(c1c(c(c(Cl)nc1[H])[H])[H])([H])[H])([H])(...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-004253211555</td>
    </tr>
    <tr>
      <th>204</th>
      <td>C(SC(c1c(c(c(F)nc1[H])[H])[H])([H])[H])([H])([...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-005723429185</td>
    </tr>
    <tr>
      <th>205</th>
      <td>C(SC(c1c(c(c(nc1Br)[H])[H])[H])([H])[H])([H])(...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Z2832853555</td>
    </tr>
    <tr>
      <th>206</th>
      <td>C(SC(c1c(c(c(nc1C([H])([H])[H])[H])[H])[H])([H...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-003024225282</td>
    </tr>
    <tr>
      <th>207</th>
      <td>C(SC(c1c(c(c(nc1Cl)[H])[H])[H])([H])[H])([H])(...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-004696925594</td>
    </tr>
    <tr>
      <th>208</th>
      <td>C(SC(c1c(c(c(nc1F)[H])[H])[H])([H])[H])([H])([...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-005922678029</td>
    </tr>
    <tr>
      <th>209</th>
      <td>C(SC(c1c(nc(c(c1Cl)[H])[H])[H])([H])[H])([H])(...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>PV-002978169168</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we marked the molecules to avoid searching for them again
# for that we use the column "enamine_searched"
cs.df[cs.df.enamine_searched == True]
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
      <th>168</th>
      <td>[H]c1nc([H])c(C([H])([H])SI)c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7272ea02c720&gt;</td>
      <td>3.858</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>3.858</td>
    </tr>
  </tbody>
</table>
</div>


