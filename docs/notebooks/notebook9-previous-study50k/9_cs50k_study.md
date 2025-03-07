# 9: Pre-evaluated CS50K with Active Learning

**Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

## Overview
An AL study using precomputed Gnina scores. 


```python
import pandas as pd
import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace

from fegrow.testing import core_5R83_path, smiles_5R83_path
```


```python
# create the chemical space
cs = ChemSpace()
# we're not growing the scaffold, we're superimposing bigger molecules on it
cs.add_scaffold(Chem.SDMolSupplier(core_5R83_path)[0])
# we can ignore the protein as the values have been pre-computed
cs.add_protein(None)
```

    /home/dresio/code/fegrow/fegrow/package.py:597: UserWarning: ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. Use a Dask cluster with processes as a work around (see the documentation for an example of this workaround) .
      warnings.warn("ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. "


    Dask can be watched on http://192.168.178.20:8989/status


    /home/dresio/code/fegrow/fegrow/package.py:801: UserWarning: The template does not have an attachement (Atoms with index 0, or in case of Smiles the * character. )
      warnings.warn("The template does not have an attachement (Atoms with index 0, "



```python
# switch on the caching
# I set it here to 6GB of RAM
cs.set_dask_caching(6e9)
```


```python
# load 50k Smiles
oracle = pd.read_csv(smiles_5R83_path)

# remove .score == 0, which was used to signal structures that were too big
oracle = oracle[oracle.cnnaffinity!=0]

# here we add Smiles which should already have been matched
# to the scaffold (rdkit Mol.HasSubstructureMatch)
smiles = oracle.Smiles.to_list()
cs.add_smiles(smiles)
```

# Active Learning

## Warning! Please change the logger in order to see what is happening inside of ChemSpace.evaluate. There is too much info to output it into the screen .

```python
import logging
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
```


```python
from fegrow.al import Model, Query
```


```python
# This is the default configuration
# cs.model = Model.gaussian_process()
cs.model = Model.linear()
cs.query = Query.Greedy()
```


```python
# we will use the preivously computed scores for this AL study
# we're going to look up the values instead
def oracle_look_up(scaffold, h, smiles, *args, **kwargs):
    # mol, data
    return None, {"score": oracle[oracle.Smiles == smiles].iloc[0].cnnaffinity}
```


```python
# the first cycle will take more time
for cycle in range(20):
    # select 2 hundred
    selections = cs.active_learning(200)
    res = cs.evaluate(selections, full_evaluation=oracle_look_up)
    
    print(f"AL{cycle:2d}. "
      f"Mean: {res.score.mean():.2f}, "
      f"Max: {res.score.max():.2f}, "
      f">4.8: {sum(res.score > 4.8):3d}, "
      f">5.0: {sum(res.score > 5.0):3d}, "
      f">5.2: {sum(res.score > 5.2):3d}, "
      f">5.4: {sum(res.score > 5.4):3d}, "
      )
```

    /home/dresio/code/fegrow/fegrow/package.py:1287: UserWarning: Selecting randomly the first samples to be studied (no score data yet). 
      warnings.warn("Selecting randomly the first samples to be studied (no score data yet). ")


    AL 0. Mean: 4.50, Max: 5.50, >4.8:  46, >5.0:  23, >5.2:   7, >5.4:   1, 
    AL 1. Mean: 5.17, Max: 6.11, >4.8: 187, >5.0: 151, >5.2:  90, >5.4:  33, 
    AL 2. Mean: 5.16, Max: 5.73, >4.8: 177, >5.0: 146, >5.2:  90, >5.4:  36, 
    AL 3. Mean: 4.93, Max: 5.73, >4.8: 132, >5.0:  85, >5.2:  42, >5.4:  20, 
    AL 4. Mean: 4.95, Max: 6.16, >4.8: 130, >5.0:  95, >5.2:  54, >5.4:  19, 
    AL 5. Mean: 4.93, Max: 5.89, >4.8: 128, >5.0:  75, >5.2:  37, >5.4:  21, 
    AL 6. Mean: 4.85, Max: 5.69, >4.8: 114, >5.0:  75, >5.2:  38, >5.4:  14, 
    AL 7. Mean: 4.76, Max: 5.59, >4.8: 101, >5.0:  60, >5.2:  20, >5.4:   2, 
    AL 8. Mean: 4.77, Max: 5.77, >4.8: 100, >5.0:  57, >5.2:  30, >5.4:  11, 
    AL 9. Mean: 4.67, Max: 5.65, >4.8:  76, >5.0:  39, >5.2:  16, >5.4:   7, 
    AL10. Mean: 4.59, Max: 5.62, >4.8:  63, >5.0:  33, >5.2:  18, >5.4:   7, 
    AL11. Mean: 4.60, Max: 6.06, >4.8:  63, >5.0:  36, >5.2:  10, >5.4:   2, 
    AL12. Mean: 4.92, Max: 5.78, >4.8: 138, >5.0:  89, >5.2:  45, >5.4:  15, 
    AL13. Mean: 5.03, Max: 5.88, >4.8: 155, >5.0: 110, >5.2:  61, >5.4:  26, 
    AL14. Mean: 5.12, Max: 6.24, >4.8: 174, >5.0: 125, >5.2:  77, >5.4:  32, 
    AL15. Mean: 5.10, Max: 6.20, >4.8: 165, >5.0: 126, >5.2:  78, >5.4:  38, 
    AL16. Mean: 5.12, Max: 5.98, >4.8: 177, >5.0: 144, >5.2:  75, >5.4:  31, 
    AL17. Mean: 5.10, Max: 5.96, >4.8: 169, >5.0: 130, >5.2:  71, >5.4:  25, 
    AL18. Mean: 5.09, Max: 5.83, >4.8: 176, >5.0: 136, >5.2:  67, >5.4:  20, 
    AL19. Mean: 5.08, Max: 6.02, >4.8: 173, >5.0: 129, >5.2:  64, >5.4:  22, 



```python

```
