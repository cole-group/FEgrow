# FEgrow
An interactive set of tools for making co-generic ligands. 

# Installation - Conda
Download the code, and use the `env.yml` to create the FEgrow environment with the necessary dependancies:
```
conda env create -f env.yml
conda activate fegrow
```

### Acknowledgments
 - RDKit
   - Constrained Embedding (https://github.com/JoshuaMeyers/Snippets/blob/master/200405_constrained_conformers.ipynb)
   - Grafting (https://pschmidtke.github.io/blog/rdkit/3d-editor/2021/01/23/grafting-fragments.html)
 - Prody (http://prody.csb.pitt.edu/)
 - TeachOpenCADD (https://projects.volkamerlab.org/teachopencadd/)
 - OpenChemistry (https://www.openchemistry.org/, molecules as R-groups)
 - Py3Dmol
 - PDBFixer (https://github.com/openmm/pdbfixer)


--------------------------------

The MIT License (MIT)
Copyright © 2022 Daniel J Cole

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.