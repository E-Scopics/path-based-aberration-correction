# Path-Based Model for Aberration Correction in Ultrasound Imaging

We propose here a code sample that implements the method proposed in:

> B. Hériard-Dubreuil, A. Besson, C. Cohen-Bacrie and J.-P. Thiran, "A Path-Based Model for Aberration Correction in Ultrasound Imaging" [in preparation].

This code is for research purposes only. No commercial use. Do not redistribute.
If you use the code or the algorithm for research purpose, please cite the above paper.

## Code Layout

The proposed code sample is implemented in JAX. Gradients are computed automatically with JAX.
It is divided in three parts:

- [utils.py](utils.py) that implements the coherence factor computation from raw data, for plane wave or single transducer emission, in JAX.

- [example_sa.ipynb](example_sa.ipynb), a jupyter notebook that uses the above functions and perform aberration correction with SA acquisitions.

- [example_pw.ipynb](example_pw.ipynb), a jupyter notebook that uses the above functions and perform aberration correction with PW acquisitions.

## Requirements

The following package are required:

- `numpy`
  
- `matplotlib`
  
- `jax`

A guide for JAX installation can be found [here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier).

To launch ***.ipynb**, please use jupyterlab or jupyter notebook (https://jupyter.org/).


## Copyright

Copyright (C) 2024 E-Scopics. All rights reserved.
This code is the exclusive propriety of E-Scopics.
