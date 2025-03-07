# 5: Chemspace streamlined

**Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

## Overview

Building and scoring molecules can be further streamlined by employing our established protocol. Here we show how to quickly build a library and score the entire library. 


```python
import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace, RGroups, Linkers

rgroups = RGroups()
linkers = Linkers()
```

    <frozen importlib._bootstrap>:241: RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher> already registered; second conversion method ignored.
    <frozen importlib._bootstrap>:241: RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterCatalogEntry> already registered; second conversion method ignored.



    MolGridWidget(grid_id='m2')



    MolGridWidget(grid_id='m1')


# Prepare the ligand template

The provided core structure `lig.pdb` has been extracted from a crystal structure of Mpro in complex with compound **4** from the Jorgensen study (PDB: 7L10), and a Cl atom has been removed to allow growth into the S3/S4 pocket. The template structure of the ligand is protonated with [Open Babel](http://openbabel.org/wiki/Main_Page):


```python
init_mol = Chem.SDMolSupplier('sarscov2/mini.sdf', removeHs=False)[0]

# get the FEgrow representation of the rdkit Mol
scaffold = fegrow.RMol(init_mol)
```


```python
# Show the 2D (with indices) representation of the core. This is used to select the desired growth vector.
scaffold.rep2D(idx=True, size=(500, 500))
```




    
![png](output_6_0.png)
    



Using the 2D drawing, select an index for the growth vector. Note that it is currently only possible to grow from hydrogen atom positions. In this case, we are selecting the hydrogen atom labelled H:40 to enable growth into the S3/S4 pocket of Mpro.


```python
# specify the connecting point
scaffold.GetAtomWithIdx(8).SetAtomicNum(0)
```


```python
# create the chemical space
cs = ChemSpace()
```

    /home/dresio/code/fegrow/fegrow/package.py:595: UserWarning: ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. Use a Dask cluster with processes as a work around (see the documentation for an example of this workaround) .
      warnings.warn("ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. "


    Dask can be watched on http://192.168.178.20:8989/status
    Generated 2 conformers. 
    Generated 1 conformers. 
    Generated 1 conformers. 
    Generated 8 conformers. 
    Generated 2 conformers. 
    Generated 4 conformers. 
    Removed 0 conformers. 
    Removed 0 conformers. 
    Removed 0 conformers. 
    Removed 1 conformers. 
    Removed 1 conformers. 
    Removed 3 conformers. 


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/parmed/structure.py:1799: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
      coords = np.asanyarray(value, dtype=np.float64)


    using ani2x
    using ani2x
    using ani2x
    using ani2x


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/aev.py:16: UserWarning: cuaev not installed
      warnings.warn("cuaev not installed")


    using ani2x
    using ani2x


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/__init__.py:59: UserWarning: Dependency not satisfied, torchani.ase will not be available
      warnings.warn("Dependency not satisfied, torchani.ase will not be available")
    Warning: importing 'simtk.openmm' is deprecated.  Import 'openmm' instead.


    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/torchani/resources/
    failed to equip `nnpops` with error: No module named 'NNPOps'


    Optimising conformer:   0%|                               | 0/1 [00:00<?, ?it/s]

    failed to equip `nnpops` with error: No module named 'NNPOps'
    failed to equip `nnpops` with error: No module named 'NNPOps'
    failed to equip `nnpops` with error: No module named 'NNPOps'
    failed to equip `nnpops` with error: No module named 'NNPOps'
    failed to equip `nnpops` with error: No module named 'NNPOps'


    Optimising conformer: 100%|███████████████████████| 1/1 [00:01<00:00,  1.66s/it]
    Optimising conformer:   0%|                               | 0/5 [00:00<?, ?it/s]
    [Aimising conformer:   0%|                               | 0/2 [00:00<?, ?it/s]
    
    
    [A[A[Ag conformer:   0%|                               | 0/1 [00:00<?, ?it/s]
    
    [A[Asing conformer:   0%|                               | 0/3 [00:00<?, ?it/s]
    
    
    
    Optimising conformer:  20%|████▌                  | 1/5 [00:05<00:23,  5.84s/it]
    [Aimising conformer:  50%|███████████▌           | 1/2 [00:05<00:05,  5.61s/it]
    Optimising conformer: 100%|███████████████████████| 2/2 [00:07<00:00,  3.62s/it]
    Optimising conformer:  40%|█████████▏             | 2/5 [00:08<00:12,  4.18s/it]
    
    
    Optimising conformer: 100%|███████████████████████| 1/1 [00:08<00:00,  8.07s/it]
    
    
    Optimising conformer:  60%|█████████████▊         | 3/5 [00:12<00:08,  4.13s/it]
    
    
    
    Optimising conformer: 100%|███████████████████████| 1/1 [00:07<00:00,  7.59s/it]
    Optimising conformer:  80%|██████████████████▍    | 4/5 [00:14<00:03,  3.31s/it]
    
    [A[Asing conformer:  67%|███████████████▎       | 2/3 [00:11<00:04,  4.93s/it]
    
    Optimising conformer: 100%|███████████████████████| 3/3 [00:12<00:00,  4.18s/it]
    Optimising conformer: 100%|███████████████████████| 5/5 [00:16<00:00,  3.37s/it]



```python
cs.add_scaffold(scaffold)
```

# Build a quick library


```python
# building molecules by attaching the most frequently used 5 R-groups
cs.add_rgroups(rgroups.Mol[:3].to_list())

# build more molecules by combining the linkers and R-groups
cs.add_rgroups(linkers.Mol[:3].to_list(), 
               rgroups.Mol[:3].to_list())
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
      <td>[H]Oc1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>8</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>1</th>
      <td>[H]c1nc([H])c(OC([H])([H])[H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>8</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAIAAAAiOjnJAAAABmJLR0QA/wD/AP+gvaeTAAAWKUlEQVR4nO3de1xT5/0H8G8S7oIKIohMq4iXopVioFMCCIplWAWh0LlZprKNbq3SV311L9tNpbpL7U1xdWutU5o6dV7qAAcoqLgShHHVlYpXNpxQQES5KiTk+/vj+EtDuIXkPElIvu+Xf/R1nuPJV/rhnJPnOc9zBIgIhPBNaOwCiHmiYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmKFiECQoWYYKCRZigYBEmzDBYjx49KigoePnll2NjY3Nzc41djoUSmMc7oevr68vLywsLC2UyWVlZWXd3N7ddKBRmZma+8MILxi3PAo3WYMnl8srKyqKioqKiokuXLv3vf/9TNYlEonnz5s2ZM+fMmTOtra22trbXrl2bNm2a8Yq1RKMpWK2traWlpTKZrLCw8NKlS11dXaomJyen+fPnBwUFSSSSoKAgZ2dnAGhpaXnmmWfq6+u9vb1lMpm7u7vxarc4Jh2s3t7ea9euqa5x1dXV6tV6eXlJJBKxWBwUFOTn5ycUDnC/2NbWFhYWVlFR4evre/HixfHjxxuwfIume7DS0uDIERg3Dg4fBlvbPk07dkBBAezdC7Nnj/iwra1QVPTkD8AneXmvqpocHR0DAgIkEsnChQsXLVrk4uKizQHv3bsXHBx8/fr1sLCw7OxsOzu7EddEdIC6+s1vEAABcPt2zaa4OATAkhJtD1VXh8ePY3IyisUoFD45LADOnl05ffr0NWvW7N27t6KiQi6X61bqnTt3pkyZAgBRUVE6H4SMiL7BsrFBOzu8caNP07DB6uzEggJMTcX4eJw48bskAaCVFYrFmJyMUin+9786V6epqqqKO8MlJCQolUrejksGYaXnCe/VVyE1FZKS4MIFEAiG2rO+HsrLobAQZDIoK4P/7xAAAJg0Cfz9QSyGoCAICgL9L1ZKpVLjlmvu3LnZ2dnh4eGHDh1ydnbes2ePvp9BhqZzJLkz1qlTGB2NAHj48HdNqjNWXR3u2oVxcTh5suZpacEC3LABDx/G//xH/1+PPr799luxWPyPf/yjf1NeXp6trS0AvPfeezx/KumLh2Ddvo12dujuji0tT5pUwSor+y5MY8dieDimpGBmJj54wE/1A3rvvfcAwMHBQSaT9W/98ssvRSKRQCD47LPPGBZh8XgIluq/f/GLJ02qYMnl+MorePAgVlejwW5slEplUlISAIwbN66ioqL/Dp988gkAiESi48ePG6gmy8NPsLq6cPp0FAqxuBhx5N8Kedfb2/vSSy8BgJub2/Xr1/vvsH37dgCwsbE5c+aM4cuzBPwECxGzsxEAFyzA3l7jBwsRu7u7IyIiAMDLy6u+vr7/Dps2beKumIWFhYYvz+zx9nRDZCRERUFFBRw6xNch9WJjY3Pq1KnAwMCamprnn3++paVFY4cPP/xw/fr1XV1d0dHR1dXVRinSnOkcSY0zFiLW1uKYMTh5MkZGGv+MxWlubvbx8QGA73//+x0dHRqtCoUiNjYWADw9Pf/LY6cZ4fGMBQBTp8Kvfw319WA6D0FNmDAhNzd32rRp//rXv1atWtWt3nsGIBKJ/vrXv4aEhNTV1S1btqyxsdFYdZofnh/0e/NNmDMHenv5PapePD098/Ly3N3dz507t379eqVSqd5qb29/+vTpBQsW3Lx5MyIi4uHDh8aq08zwHCwbG/j4Y34PyQNvb++zZ8+OHz/+6NGjGzZs0GgdO3bsmTNnZs+efeXKldjY2MePHxulSHOj80W0pAT37Ru43/zYMdy3D5uadD42E/n5+dyjDSkpKf1bzWeguqUFGxpQoTBuFboHi1NTw0sZBpKRkWFlZQUAu3bt6t/69ddfj+KB6ooKjIvDsWOfDHRYW+PixXjypOE6pvvSK1j37qFIhHPnGv3XYwS++OILgUAgEAjS0tL6txYXFzs6OgJAcnKywUvTQ1oaWlmhQIChobh5M27dinFxaGeHAJiYiL29hq9Ir2AdPIgAuHy55nYT/23/4x//CADW1tZmMlBdUoLW1ujkhLm5fbbfuoUzZyIAfvSR4YvSK1jccw379mluF4vxxRdN7h5L3VtvvQUA9vb2X331Vf/WUTZQvXw5AuCnnw7QdOUKCoXo4oKdnQYuSvdgdXWhgwMKhagxXnLjBgLghAloynfASqXy5z//OZjBQHVrKwqF6OSEjx4NvMPSpQiAGRmGLUuPYP397wiAixZpbn/3XQTAdev0KssAFAqFOQxUX7yIABgcPOgOW7ciAG7bZsCaEPXpec/IAACIjtZ2u6kRiUSHDh2KiIhoamqKjIz89ttvNXbYtm3bpk2benp6YmNjL126ZJQih3fvHgCAh8egO0ye/N1uhqRbHhUKdHVFALx2rc/2hgYUCtHeHvuNy5mozs7OwMBAAJg3b979+/c1WpVK5fr16wHA1dX16tWrRqlwUI8fIyIeP44A+MMfDrrbvn0IgK+8YrC6ODqesWQyaG6GmTM1J3ilp4NSCc8/D2PG6B15g3BwcMjMzPTx8amqqlq+fHlnZ6d6q0Ag2L9/f2xsbHNz87Jly2pra41V5xP19XDiBLz+Ovj7w/TpAADOzgAA9+8P+le4c5Wrq0HqU6NbHt94AwHwrbc0t3PPNRw8qG/eDezu3bvcHPzw8PDH3JlATVdXV0hICADMnDmzoaHBoJV1duI//4nvvotRUZrzmWxssK4OGxsRACdOHLSzKjYWAdDgX0F0DJaXFwJgUVGfje3taGeHIpFJdzQM5ubNm9wc/B/96Ee9/f4ntba2LliwAAB8fX0fMH1iHxHr6jAzEzdvRokEbW37hGnSJFyxAlNSMC8Pu7qe7B8QgACYlTXAoRob0d4e7e2xuZltzf3oEqzLlxEA3d01f0mOHUMADAnhpzLDu3z5MjcH/9VXX+3f2tTUNHv2bAAICwt7NNh3e93I5VhVhfv2YUICTp/eJ0kiEfr4YEIC7tuHVVUDdz2fPIkAOGOGZsdPTw9GRSEAGmMUQZdgbd8+8O3gj39srG5e3hhsoLqhoSE9Pb18504MCkJ7+z5hcnbG5cvxt7/F8+exvV2rw73yCgKgmxv+/vd44QLKZPjZZzh/PgKgv79RvknpEqwFCxAAs7P7bOzpQWdnBMDbt/mpzFi0HKj+yU9+MqKBaoVCUVVVJZVKk5KSfHx8BAIBAPxaLH4SJi8vTEjA1FQsK9NlaE+pxD17NG/CbG1x40Zto8m3EQerthYFAnR01Ozpzc1FAJw/n7fKjIivgeoHDx5kZ2dv27YtPDzcyclJ/TuTo6PjkiVLdu/YgVlZ303I1JNcjkVFePgwSqWYm2vcLp8RB2vPHgTAl17S3P7aawiAW7fyU5bR6TxQffv2balUmpycLBaLNab5e3h4xMfHp6amFhQUdHd3G+Tf0ZdUil9+aZiPGnGwlizRnFCPiEolTpmCAFhWxltlRqflQPXHH39cUFCQmpoaHx8/ceJE9SRZWVmJxeLk5GSpVGr8yRrFxSgQoK0tnjtngE8bWbAePEBra7S21jx5l5YiAHp6mvoDMyMy7ED1n/70J+6iph6myZMnx8XF7dq1q6ioyDinpSFs2oQA6OCA7KdSjixYX3yBALhsmeb2LVsQADds4K0sE6FQKOLj42HwgeoZM2bY2dnNmTNn48aNR44cMf5paWhKJa5fjwDo6orffMP0o0YWrBdfRADcu1dz+7x5CIB5ebyVZToeP368dOlSgUBw4MABjaaGhgahUGhvb99/xqLpUiie9MV7evK/0I+aEQTr0aNHISEfSCQ1d+702X77du3ixWVz5z7u6eG5OBPR1taWnp7ef/unn34KANHR0YYvSS9dXRgSggDo7Y3MRqhGEKzTp08DQEBAgMb2Dz/8EABefvllXgsbBSIjIwHg4KgbGUXE1tYnvZG+vozWlBrB0w0ZGRncL6iW281bR0dHfn6+SCRasWKFsWsZubFj4cwZmD0brlyBmBhgMZVSywD29vZOmjQJAL7++mv17c3NzSKRyNbWtq2tjUHuTdexY8cAIGT0jowi4p07T3qJoqJ4f5Bc2zNWUVFRQ0PDjBkz5s2bp749IyOjt7e3f8+y2TOH8/SUKZCTAy4ukJkJiYnA64r/2gaL+zmuWrVqwO2j++c7cnK5PCcnBwBWrlxp7Fr0M3cuZGeDoyMcOgSvv87nkbU8s82aNQsANPqgOzs7HRwchELhgCubmTHupWLPPPOMsQvhyblzTx782rmTr0Nqdca6evXqjRs3XF1ducfDVc6ePdvV1bVw4UKPIR7mN0eDnb9Hq6VL4ehREIng7bdh/35eDqlVsNLT0wEgKipKJBKpb7fM6yAiZmZmgpn9w2NiYO9eQIRf/hJOnODhgNqc1p577jkAyOg76VGhULi6ugLANY2ZOuautLQUADw9PUffwiHD2rFDKRBsXrr0nN4D1cMHq66uTiAQODg4dPadpp2fnw8ATz/9tJ4VjDpbtmwBgA3mNzKKiIhHduwAACcnpxL9lvoc/lLInagiIiIcHBw0toM53WdojbsxMKvroJrVW7YkJia2t7dHRkZevXpV9wMNGz1uUevPP/9cY/v06dMBoJhb2d1i3Lp1CwDGjx9vco/E8Ed9zd//6DpQPUyw2trabG1tRSLRvXv31LdfvnwZANzd3fvPlDJv3MDomjVrjF0IW6qplN7e3rpNpRzmUpiVldXd3R0cHOzadyotdzmIiYkZ8L2mZsxCvgir1vy9deuWjmv+Dp271atXA8Du3bs1tvv5+QFATk6ODlkevZqbm62srCxnYFQ1lTI0NHSkUymHClZPTw83gbOm70qjtbW1AoHA0dGR53mbJu/AgQMAsLz/EobmS+eplENdyC5cuPDw4UNfX1/uPl2lsLCQ+/la2vuVLeQ6qG7KlCk5OTkuLi6ZmZmJiYmo/UD1EKFbs2YNDDIn+O7duzc03tdr7ix2YBTVplJu3LhRy78yVLDs7e0BYMBJmxbo1KlTABAYGGjsQozj3Llz3FTKndoNVA8VLO6JhjFjxhh67R6TtHbtWhhFSykzcOrUKe3X/B0qWP/+97+tra0BYOHChZZ2n67BYgdGNWi/5u8w3Q0XL150c3MDgOjo6NH9IhD9WOzAaH87duwALdb8HaZ7c/HixefPn3dxccnIyPjZz36GvD69OopY4PfBwWzdujU5Odna2nqYMGgT0qKiojFjxgDAW/0Xh7QMXIdLkcYShpZKqVRWV1cPvY+2jyar1ld5//339S5slLHYgVF9aDvSFx4enpaWJhQKN2/ezHVAWw6LHRjVy4hiyK2vIhKJTpw4wSjpJsgyB0b1NOL1sVJSUgDAxsbm7NmzLAoyNRY7MKqnEZ/b33nnnTfeeKOnpycuLq68vJz3M6ip4ZYDeeGFFyxtYFRPutw0fPTRR+vWrWtvb//BD35QXV3Ne00mhToadCNAnbqm5HJ5TExMVlbW9773PZlM9tRTT/FemSl4+PAh1z/c2NjozL1chGhHx6851tbWJ06cCA4Ovnv37rJly5qamvgty0ScPn1aLpeHhYVRqkZK9+/P3NOrfn5+N2/eXLFiRXt7O49lmQi6DupMx0uhSlNTU3Bw8I0bN5YsWZKVlWVOd7jd3d0TJ07s6Oiora3lnqIk2tO3x8/NzS0nJ8fDw+PChQurV69WKBS8lGUK8vLy2tvb/f39KVU64KEr2cvLKzc31/wGquk6qBe+OsTMbKBatYJhVVWVsWsZlXgLFiLm5uaOooFqhUJRWVm5d+/eAZ9Zk8lkADBjxgzDF2Ye+AwWIh45ckQoFAoEgr/85S/8HpkXbW1teXl5KSkpK1asUPUghIaG9t/zV7/6FQC8+eabhi/SPPAcLDSxgWqlUnn16tUDBw4kJiY+/fTT3MvcVLy9vRMSEgacLcI9719QUGDwks0E/8FCYw9Ud3R0qN6apLEygLW1NffWpOPHjzc2Ng52hG+++QYA3NzcFAqFISs3J1YsvhC88847bW1tu3fvjouLy8/PF4vFLD5FXX19fWFhoUwmKy8vLy0t7enpUTV5eHiIxeKgoCCJROLv769NTxv3ANbKlSs1VjAkI8AosEqlct26dQDg6uo67GOsOpDL5WVlZampqQkJCRojlVZWVj4+PklJSVKpVLfvdOPGjQMAqVTKe9mWQ9+e9yHI5fJVq1ZlZ2fzNVDd0NBQWlqqOjM9Vnufwrhx4wICAiQSSVBQUGBgoMYacSOSnp4eExMDAM3NzRMmTNCzZovFMFgA8OjRo4iIiIKCgpkzZ8pkMu5JAe319vZeu3ZNlSSNBea8vLy4JEkkEtVblnXAfUp5eblMJpNKpdxldOrUqbW1tbodkADrYAFAa2trWFhYZWVlQEDA+fPnh32BRVtbW0lJCZekgoKC1tZWVZOjo6Ovry+XJIlEwr30WzcPHjwoKioqLi4uLCwsKSnp6OhQNQkEgqeeeio/P3/atGk6H58wDxZoPVC9c+fOzz///Pr16+obZ82atWjRosDAwMDAQB8fH32mM9TU1HB5LSwsrKysVCqVqiYPDw8ur9OmTYuIiDCnoXRjYfKtUAM3UB0UFMQNVJ88edLKaoDPvX///vXr1x0cHPz8/LjvcaGhoRpvWR6Rzs7OyspKLkn5+fnNzc2qJmtraz8/P4lEIhaLQ0NDp06dqvOnkAEZ4ozFqaqqWrx4cUtLy9q1a9PS0vrfEt26devhw4fPPvvsgLHTEr/9DkRnhgsWABQXF4eHh3d2dr799tt/+MMfeDmmQqG4cuUKl6SvvvpK/Y7byspq1qxZXJLEYvHcuXN5+USiDYMGCwDy8vJWrlzZ3d39/vvvc+NxOuD6HbhrnEwmU+93GDt27HPPPcdLvwPRh6GDBQBHjx7l3vO7f//+n/70p9r8FcP0OxAeGSFYAPDnP//5tddeE4lEf/vb3+Li4gbcR73fQSaTqa8IzWO/A2HFWF3+3EC1SCT64IMPVBtv374tlUqTkpL69yx4eHjEx8enpqaWlZXR4hymzzhnLE50dHRmZqZAIHj22Wfv3r3b3d3d1tamanVwcPD39+d6sBYuXKhPvwMxPGMGq7e3d9asWTU1Naot6j0CAQEB3POoZDQyZrAAoKur63e/+11ubm5ISMjatWt9fX2NWAzhkZGDRcwVrSRGmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESYoWIQJChZhgoJFmKBgESb+D8D/Sy063gZ1AAABR3pUWHRyZGtpdFBLTCByZGtpdCAyMDI0LjA5LjYAAHice79v7T0GIBAAYiYGCOCA4gZGNgcLIM3MyMLuoAFiMMMF2CACLBwMYJqJjcEAzEdSAGVwMzBmMDEyJTAxZzAxsyiwsDIwsyWwsWcwsTMkODEC1bAxsLMxMzGKFwLZjAwwNyyYI3Lg3b1t+0Ccv2of9i+a1WkPYj/Ivrzv27XNYPbUvVL2a5hn24HYGaEsDrW/FMHinkWuDkbJH8B6rVOU7IonTtgPYtf9X7b/2NldYLY9xLdCKSBXMvADyfj45PzcgtKS1JSAovyCYjZYcLADcV5prmNRfi5YWXBJalFqfnJGaq5Lfl4qkiwjyPVYlIDFha0eMIKCGJtNMF+DAj3e2TMgKDEvG5nNBHYKmXrZKNDLTIFeVgr0slOgl4UCvYwU6AXLCosBAKo7py929/5DAAABGHpUWHRNT0wgcmRraXQgMjAyNC4wOS42AAB4nH1SW2rEMAz8zyl0gRi9bNmfm2QppWwCbdo79H/vT+WU1LtgKkUgOyNlZsgANd6Xt+87/AUvwwCA/zylFPgSRBxuUBuYri+vK8z7ZTpv5u1z3T8ge2LNZ+Rl327nDcEMIwdhxoQwYpBoZD4S8Ig2y7DCSCFbouJzAS2icAcodSUGJS7+noKilpI7QHUghoSqMXojyJZ6CyNswAGTGUkdEDbu7Uu+TwLmSHhI0VQiWgdox4cpZs+qyZmqlA4wVynkzDQZ105jUYwd5HVdnnz9dXra1qU5rV7c7FQvaaZprWZNhcfmAHmlJlQ9rcnxA+RGWr3okdkjj3o+/xXvhx98V3lspY235AAAAIx6VFh0U01JTEVTIHJka2l0IDIwMjQuMDkuNgAAeJwdjLsNQzEMA1dJmQCKQP1tuHwDZAj3meANHzlgQx6Iuz5b9nfvLY/7aYwRQm+w5wwULWVklRA4TUsHrW5wj2hk0Eqldf6i05SEHdFAeFTK7I0KWBNlU0Uet0VJ/U8Gz6IuHtOPWmJ0DmmfO73uH8xQISUxiHCYAAAAAElFTkSuQmCC" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>2</th>
      <td>[H]c1nc([H])c(N([H])[H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>8</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>3</th>
      <td>[H]OC([H])([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>8</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>4</th>
      <td>[H]c1nc([H])c(C(=O)N([H])OC([H])([H])[H])c([H]...</td>
      <td>&lt;NA&gt;</td>
      <td>8</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>5</th>
      <td>[H]c1nc([H])c(N([H])C(=O)N([H])[H])c([H])c1[H]</td>
      <td>&lt;NA&gt;</td>
      <td>8</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
  </tbody>
</table>
</div>


### Prepare the protein

The protein-ligand complex structure is downloaded, and [PDBFixer](https://github.com/openmm/pdbfixer) is used to protonate the protein, and perform other simple repair:


```python
# get the protein-ligand complex structure
!wget -nc https://files.rcsb.org/download/7L10.pdb

# load the complex with the ligand
sys = prody.parsePDB('7L10.pdb')

# remove any unwanted molecules
rec = sys.select('not (nucleic or hetatm or water)')

# save the processed protein
prody.writePDB('rec.pdb', rec)

# fix the receptor file (missing residues, protonation, etc)
fegrow.fix_receptor("rec.pdb", "rec_final.pdb")

# load back into prody
rec_final = prody.parsePDB("rec_final.pdb")
```

    File ‘7L10.pdb’ already there; not retrieving.
    


    @> 2609 atoms and 1 coordinate set(s) were parsed in 0.02s.
    @> 4638 atoms and 1 coordinate set(s) were parsed in 0.03s.



```python
# make your chemical space aware of your receptor (important for the next step! )
cs.add_protein("rec_final.pdb")
```


```python
# build and score the entire chemical space
cs.evaluate()
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
      <th>0</th>
      <td>[H]Oc1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01704c8d60&gt;</td>
      <td>3.215</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[H]c1nc([H])c(OC([H])([H])[H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01704c9b70&gt;</td>
      <td>3.231</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[H]c1nc([H])c(N([H])[H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e0170790ae0&gt;</td>
      <td>3.188</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[H]OC([H])([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01704cab10&gt;</td>
      <td>3.225</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[H]c1nc([H])c(C(=O)N([H])OC([H])([H])[H])c([H]...</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01707918f0&gt;</td>
      <td>3.39</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[H]c1nc([H])c(N([H])C(=O)N([H])[H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01cd52d800&gt;</td>
      <td>3.551</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# verify that the score has been computed
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
      <td>[H]Oc1c([H])nc([H])c([H])c1[H]</td>
      <td>3.215</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAIAAAAiOjnJAAAABmJLR0QA/wD/AP+gvaeTAAASPklEQVR4nO3da1RU5RoH8P8MKIgoFy94QSs1AUnxQpZgrWppeXLpMfFKoCmLMK8oIkWmaURmJqaWcbyFmR46tjwpXU6mZjdUvBVGqUmaKWIKgQoDw8xzPmzCYoYRZvY7e294fssvzov7fcA/e/bs/V50RATG5KZXugDWOHGwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJwcFiQnCwmBAcLCYEB4sJ4eq0ns6dOxcXF/fjjz8C6Nu3b4sWLZzWtW0Gg+H48eMAAgIC0tPTu3XrpnRFjYHOaTus+vr6FhcXO6cvu3l5eRUVFen1fCJ3lJPOWAsWLCguLtbpdLGxsR4eHiEhIao6Y504caKioiI9Pb2kpCQhISEtLU3porSPxDMYDP7+/gAmTJjghO7sFhUVBaB9+/Y3b95UuhbNc0awUlJSAPTq1ctgMDihO7sZjcYBAwYAWLJkidK1aJ7wYBUUFLRu3RrAZ599Jrovx3399dc6na5Fixbnzp1TuhZtEx6s6OhoABEREaI7ksu4ceMAREZGKl2Iton9VHj06NGBAwe6urr+8MMPPXr0ENeRjC5cuBAYGFheXn7gwIEHHnhA6XK0SuCnQiKaM2eO2WyeP3++1VRFRUUpewPCy8tr27ZttV7s0qVLQkLCSy+9FB8fn5OTw7ce7CTuZJiRkQHAz8+vpKTE6hf4+fkp+723adPGamFlZWVdu3YFsGnTJnE/n8ZN1FvhjRs3AgICLl26lJGRMWnSJKtfs3fvXoPBIKL3emrevPnQoUOtNr333ntRUVF+fn6nTp3y8vJycmGNgaDAJicnAxgwYIDJZBLUhVBms1m6wEpKSlK6Fk0SEqz8/Hx3d3edTvfVV1+JOL5zHD16VK/XN2/e/PTp00rXoj1CrkwTExMNBkNUVNTgwYNFHN85+vfvP2nSpMrKygULFihdiwbJHtV9+/YB8PDwOH/+vOwHd7LLly9Ld3c//fRTpWvRGJnPWCaTae7cuQCSk5OlD1aa5ufn99xzzwGYN2+e0WhUuhxNkTen69atA9C1a9dG8xy3oqLi7rvvBrB27Vqla9ESOYNVXFzcrl07ADt27JDxsIrbuXMnAB8fn6tXrypdi2bIGaz4+HgAgwcPNpvNMh5WDR599FEAs2bNUroQzZDtBulPP/3Up08fs9l89OjRkJAQWY6pHnl5eSEhIUR0/Pjx3r17K12OBsh28S5d3sbGxja+VAHo1atXXFycyWSSzsrs9mQ572VlZQHw9va+cuWKLAdUoaKiojZt2gD48MMPla5FA2QIVmVlZUBAAIC0tDTHj6Zmq1evBtC9e3eVD4VVAxmCtWLFCgCBgYGVlZWOH03NjEajdIH16quvKl2L2jkarMLCQm9vbwAff/yxLAWp3Oeffw6gVatWly5dUroWVXM0WLGxsQCGDx8uSzWaMHLkSABTp05VuhBVc+h2w4kTJ0JDQ/V6fW5urnSZ1RScPXs2ODjYaDRmZ2cPHDhQ6XJUyqHbDfHx8SaTafbs2U0nVQC6d+8uDbmOj4935NeykbP7XJeZmQmgXbt2xcXFsp1ANaK0tLRjx44Atm3bpnQtKmVnsMrKyu68804A6enp8hakFRs2bADg7+9/48YNpWtRIzuDtWTJEgB9+/atqqqStyCtMJlM9957L4DFixcrXYsa2ROs3377rWXLlgC++OIL2QvSkG+//VaaNv3LL78oXYvq2BOsyMhIAOPGjZO9Gs2ZOHEigPHjxytdiOo0OFjSr6m7uzv/mhLRhQsXpJP3gQMHlK5FXRp2u6HmM3ZiYqJ08d7E+fv7z58/H3/eeVG6HDVpUAw3bdoEoHPnzvxRqEZZWdkdd9wBYP369UrXoiINCFbNzZutW7eKK0iLtm/fDqB9+/Z//PGH0rWoRQOClZSUBGDQoEGNb+Sx4x588EEA8+fPV7oQtajvs8L8/Pzg4OCKioqDBw/yAzJLx48fDw0NdXV1zc3N7dmzp9LlKK++F+8JCQkGg2Hy5MmcKqv69es3ZcqUyspK6Vpe1aqqkJOD3buxaxcOHYLV+ZK//ILcXFy/bqWJCLm5yM29TS/1Oa3t3bsXgKen58WLFwWfQTWssLBQWpfmk08+UbqWOpSWUlISeXkRcOtPq1Y0dy7VeuD70EME0EcfWTlIWVn1P7Tp9sGqqqqShk2+8sorDfgemqTly5cDCAoKUuNg2mvXqH9/AsjXl2bNog0baMMGmj2b2rYlgIKD6fLlW1/shGCtXbsWQLdu3crLyxv2nTQ9FRUV0gXWG2+8oXQtFsaOJYDuv59qTbu9do3Cwgigf/zj1ouig1VUVNS2bVsAO3furF/5Td2uXbsA+Pj4/P7770rX8hfff08AtWxJFy5Yab10iVq1IoC+/bb6FYeDdZuL98mTJ1+9evWRRx4ZNWrUbS7WGABgxIgRw4YNKy4uXrx4sdK1/MX77wPAE0/A399Ka8eOGD8eADIz5erQVrCysrJ2794NIDU1Va7+moKVK1c2a9YsPT39+++/V7qWPx0+DAA2lit78EEAyMmRq0NbqybHxcUBCA4Ovu+++yxbb9686eLi4u7uLlcpmlNRUWE0Gj09PWu9HhQUNG3atDVr1gwdOjQwMNCRLtZ36NDz8mVHjoCkJDz+OKSDWD1dSbp0AYBafcXHw/K8W8/R2DbeJqWlY6zu/5GVldW5c+eUlBTbb7SN27Jlyzp16rRr1y7LpszMzObNmzu+lPeRwMC/3Rqw48+GDUREAQEEkI3NQbKzCaBOnar/Kl1j+fiQn5+VP/W4xrJ1xho5cuTGjRu3bNmSlJTk5ub216aWLVtevHgxNTU1Ojq6ESywZofCwsLU1NTS0tJaPxkARqNx0aJFlZWVM2bMGDt2rCO9BLi4wMFBE9I8l9atAVi/4SkpKQEAb++/vbh1Kx5/vPZXlpfDw+P2/doIndFovOeeewAsX77csjUiIgJAdHS07eQ2Vk899RSAUaNGWTa9/vrrAHr06KGimfgTJxJAL79c5xekpRFAI0dW/1X07QYbE3/Pnz/v4eGh9aWR7WNjQeUrV65IU8M/svq/opS33iKABg+u8wuGDCGAVqyo/qsTbpCOGDECQExMjGXT888/Dy0v5m4fs9ksrQb97LPPWrY+/fTTAIYOHer8wmwpKqKWLeu8zDpwgHQ6cnenwsLqV5wQrJ9//tnNzU2v1x8+fLhW082bN6ULrIyMjNsep9HYunUr6tjK5cSJEy4uLq6uridPnlSkNltWriSAvL2p1qeNTz+lNm0IoGXLbr3ohGARkfTEPiwszHIk1pYtW+r6KTdKNdvsbN682bL1oYceAjB37lyn11UPZjMlJlZnokcPGj2axoyp/rQI0Jw59Ne3HecEq2bs6Pbt2y2qNYeHhwNITk6uz6G07oUXXgDQv39/y3f/999/H4Cvr++1a9cUqa1evv6axoypfvAsPZAePZosp/E99RQFBdGXX1o5gsFAQUEUFGS7n/qOIF2/fj3qmPh75MiRJrI1yK+//ip9XvnS4ideXl4uzS55++23FamtwQwGEjmqoL7Bsj3xV9rfa/To0XKWpj7STaknn3zSsmnp0qUApFVonF+YCjVgzPs333xT137JNVuDaGLjZ/vUbBdtuZVLzdTw/fv3K1GaGjVs+tf48eMBTJw40bLp5ZdfbsS/siaTKTQ0FMDSpUstW6OiogCMGTPG+YWpVsOCZWPir8FgkPbnfeutt+QrTy3S09MBdOnSxXIrl+zsbGlqeH5+viK1qVODp9hLw4z69etn+bFox44d0seiRrY1SElJSYcOHQBkZmbWajKbzdLskoULFypSm2o1OFg1E383SI/N/07aCXfOnDly1KYW8+bNAxAeHm55G2/z5s0AOnfufP36dUVqUy17VpuRdn63OvH35MmTrq6u0vQ6OcpT3pkzZ6QHDzk5ObWarl+/3qlTJwDvvvuuIrWpmZ0Lr0n7JS9YsMCy6ZlnngkLmzhhwm+OFaYWw4cPBxAbG2vZJG1leP/99/PUcEt2BuvYsWPSTdFTp07Vavr9d5OPDwG0e7fD1Sltz5490uCOgoKCWk1nz56V9r0+ePCgIrWpnP2L206dGjN48JSoqMuWTdLYnu7dST3jkexgNBqDg4MBrKgZTPIXTzzxBIDJkyc7vS5tsD9YBQWm1q2tD8QwGumeewig115zqDhlrVy5sq7xejw1/LYc2pni1VcJoKAgspz3u2dP9exti/cQbbh27Zqvry+ArKysWk1VVVV9+vQBkJqaqkhtmuBQsCoqqGdPAmj1aiutw4cTQNauejVg2rRpAIYMGWLZ9OabbwK46667eGq4DY7upfPf/1bP5rCc93vmDLm5kV5PFp/T1c7GTZOaqeEffPCBIrVphQzbyj32GAE0Y4aVpoQEAigsjLT1eVy6zSuttlrL7NmzATz88MPOr0pbZAhWXh41a0YuLvTdd7WbSkupQwcC6N//drwfJ7HxYCovL69Zs2YuLi7fWX6r7O/k2bp31iwC6JFHrDT9618EkL8/aWI53JpH6evWrbNsHTZsGIDp06c7vzDNkSdYRUXVg10t16QxmSg0lACyNqFadVJSUuoa/KPSZWTUSp5gEdGaNQRQt25Wxrt+8w3pdNSiBVkMEFSXgoKCuoYr1ix8tWrVKkVq0xzZglVVRb17E0BW1/0bN44AioyUqzchoqOjAURERFg2qXqpPlWSLVhEtHcvAeTpSZbbJf/6K3l4kE5nfd6HGkhTQtzc3M6cOVOrSQOLi6qPnMEion/+kwCaMsVK06JFBFC/fqTCWdNms1laqsnqJLaYmBgAI0aMcH5h2iVzsM6erb4peuhQ7aayMrrjDgJo40Z5+5RBRkZGXdNujx075uLiYnUcB7NB5mARUVISATRokJWbou+9RzodzZwpe58OqRmvZ3WhAN5ywj7yB6u0lDp2JIAsd9wxm9X4eCc5ObmupU14kxy7yR8sItq0iQDq3FkDN0Xz8/Ol8XqWizHxtl6OEBIsk4kGDiSAFi0ScXg52Vg+7sUXX5TmIzXZfa8dISRY9OdN0bFjBR1eHvv27QPg4eFhObmZt051kKhgEdGJE+KOLQ9pFIPVJXp5s2cH1XdbOTv88Qf27AGA8HB06lS79fPPUVyMIUPg4yOo/9srLS1ds2ZNQkJCrUXFs7Ozw8PD3d3d8/LyeIdiO4nL7PHj1WswPfaYlVbpyfSRI+L6t5PtdXVYPTm6EHl9/O9/+M9/nNCPPN55552cnBx/f//ExESla9Ew4cG68040b474eJSWiu5KBtevX1+4cCGA5cuXSxfvzD7Cg9WlC6ZPx6VLeOEF0V3JICUlpaCgYNCgQRMmTFC6Fm1zxlvhokVo2xZr1+LQISf0Zr/9+/enpaXp9fpVq1bpdDqly9E2ZwTLxwdLl8JsxsyZju7fIVRkZKTRaOzXrx/ve+04W3vpyCguDu+8g8OHsW4dZs78W9PNm+jf39Hj3333lTNnHnDkCIWFhSUlJTqdbuPGjY5Ww5wWLL0eq1cjLAzPP4+ICHTseKvJZMLp044e39PT9bTDR9Hr9TExMSEhIY5Ww5wWLAD33YeYGKxfj+RkbN5863VPT/z0k6MH1+tbmM0OHaWoqMjX1zdA2iuLOcx5wQLwyivYuRNbtmD69Fsv6vWQ43+zBcCZUBFnXLzXaNMGy5bBbEZCQn336WQa5dRgAZg6FWFh+OornDzp5J6ZUzk7WDod3n4brq6oqHByz8ypnB0sAL17Y8YM53fLnErgxbu3N8aORVCQlaalS3HlCqqqlBwzw4QSOB6LNWUKvBWypoCDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE4KDxYTgYDEhOFhMCA4WE+L/HHCD/CHxvbkAAAEbelRYdHJka2l0UEtMIHJka2l0IDIwMjQuMDkuNgAAeJx7v2/tPQYgEABiJgYIYIfiBkYOBgsgzczIxOagAWKwsDlABFjYIQLMcAFMBjcDowIjUwYTE3MCM0sGEwtrAitbBhMbY4ITI1ABGyMbKwszk3gqkM3IALPZc5KKw/bcaFUQZ5JKp72qp9ASENtz0hK7h27L9oPYeben7Uewv+1vNNdXRRK3R1IPZttD/CWUAnIWAz+QjI9Pzs8tKC1JTQkoyi8oZkP2eF5prmNRfi5YWXBJalFqfnJGaq5Lfl4qkiwjyMVYlIDFha2WMIICE5tNMJ9ygOScPQOCEvOykdlMbAwMZOtloUAvMwV6WSnQy0SBXgYK9IIVCIsBAD+/k6Qh8LavAAAA33pUWHRNT0wgcmRraXQgMjAyNC4wOS42AAB4nH2RXQ7CIAzH3zlFLyAphY3xuK8YYwaJTu/gu/ePZThhiaGlSYFf2vJHQLTbdH294Wc0CQGAleWcg6dGRLFATGCYzxcP49oP+8kYHn69g2XH6EeyX8OynygIQLKxypADlLhZkewcwQhKYuJOFVAziFJTApUk51B3f0ADPt4r23H3KtlwSb53tJGV3m0CvyUrFW05ZIWb/XRQK+k3BD9l/aJTlslw6CyGiZGfHL3J7+INtHl6w2HzjIaDylHKxnG/fznn4gO14W7gEmjPVQAAAGZ6VFh0U01JTEVTIHJka2l0IDIwMjQuMDkuNgAAeJzzTzZMzktOTjZUqNEw0jM1NzQx1jHQsTbUM4AxDfSMjQxNjCx1dA31jCwtDUx0rIEsQ3MLU3NUIUsjkJABkjRcFm4GTESzBgBRHRl7lY6a9QAAAABJRU5ErkJggg==" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>1</th>
      <td>[H]c1nc([H])c(OC([H])([H])[H])c([H])c1[H]</td>
      <td>3.231</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>2</th>
      <td>[H]c1nc([H])c(N([H])[H])c([H])c1[H]</td>
      <td>3.188</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>3</th>
      <td>[H]OC([H])([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>3.225</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>4</th>
      <td>[H]c1nc([H])c(C(=O)N([H])OC([H])([H])[H])c([H]...</td>
      <td>3.39</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
    <tr>
      <th>5</th>
      <td>[H]c1nc([H])c(N([H])C(=O)N([H])[H])c([H])c1[H]</td>
      <td>3.551</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td style="text-align: center;"><div style="width: 200px; height: 200px" data-content="rdkit/molecule"><img src="data:image/png;base64,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" alt="Mol"/></div></td>
    </tr>
  </tbody>
</table>
</div>



```python
# access the Pandas dataframe directly 
cs.df
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
      <th>0</th>
      <td>[H]Oc1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01704c8d60&gt;</td>
      <td>3.215</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[H]c1nc([H])c(OC([H])([H])[H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01704c9b70&gt;</td>
      <td>3.231</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[H]c1nc([H])c(N([H])[H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e0170790ae0&gt;</td>
      <td>3.188</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[H]OC([H])([H])c1c([H])nc([H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01704cab10&gt;</td>
      <td>3.225</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[H]c1nc([H])c(C(=O)N([H])OC([H])([H])[H])c([H]...</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01707918f0&gt;</td>
      <td>3.39</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[H]c1nc([H])c(N([H])C(=O)N([H])[H])c([H])c1[H]</td>
      <td>&lt;fegrow.package.RMol object at 0x7e01cd52d800&gt;</td>
      <td>3.551</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# you can save the entire ChemSpace into an .SDF file, which can be used to recover ChemSpace
cs.to_sdf("cs_optimised_molecules.sdf")

# or access the molecules directly
cs[0].to_file("best_conformers0.pdb") 
```


```python
# recreate the chemical space
cs = ChemSpace.from_sdf("cs_optimised_molecules.sdf")
```

    Dask can be watched on http://192.168.178.20:33405/status


    /home/dresio/code/fegrow/fegrow/package.py:595: UserWarning: ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. Use a Dask cluster with processes as a work around (see the documentation for an example of this workaround) .
      warnings.warn("ANI uses TORCHAni which is not threadsafe, leading to random SEGFAULTS. "
    /home/dresio/software/mambaforge/envs/fegrow-onechannel/lib/python3.11/site-packages/distributed/node.py:187: UserWarning: Port 8989 is already in use.
    Perhaps you already have a cluster running?
    Hosting the HTTP server on port 33405 instead
      warnings.warn(



```python
# search the Enamine database for the best 3 scoring molecules in your chemical space 
# and enrich your chemical space by adding them to the chemical space
# (relies on https://sw.docking.org/)
# cs.add_enamine_molecules(3)
```
