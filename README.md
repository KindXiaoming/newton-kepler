# Newton–Kepler

This is the github repo for the paper "From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers"(TBA).

Experiments and analysis for probing inductive biases of transformers on physics-inspired tasks: **1D sine waves** and **2D Kepler orbits** (Newtonian gravity).

## Overview

This folder contains:

- **Training scripts** for classification and regression transformers on discretized trajectories.
- **Analysis notebooks** that reproduce figures and analyze training dynamics, spatial representations, scaling, and block-size (context length) effects.
- **Dataset generation** for sine waves and Kepler orbits.

Notebooks expect either pretrained checkpoints (Fig 2) or results from the training scripts; each notebook describes its dependencies in the first cells.

## Structure

| Path | Description |
|------|-------------|
| **fig2_vafa_spatial_map/** | Spatial map analysis of the pretrained transformer by Vafa et al. Loads a pretrained checkpoint and analyzes representations. **Requires checkpoint** (see below). |
| **fig3a_1d_sine_wave_embedding_evolution.ipynb** | 1D sine wave: embedding evolution over training. |
| **fig3b_1d_sine_wave_scaling_law.ipynb** | 1D sine wave: scaling law analysis. Uses results from `sine.py`. |
| **fig4_kepler_noisy_context_learning.ipynb** | Kepler: noisy context learning. |
| **fig5_regression_classification_comparison.ipynb** | Regression vs classification (Kepler). Uses results from `kepler.py` and `kepler_cv.py`. |
| **fig6_block_size_kepler_newton.ipynb** | Block size (context length) experiments for Kepler/Newton. Uses results from `kepler_cv_blocksize.py`. |
| **generate_dataset.py** | Generate and save large sine wave trajectory dataset. |
| **generate_kepler_cv.py** | Generate Kepler orbit trajectories (continuous variables) for training. |
| **sine.py** | Train transformers on 1D sine wave data. Saves to `./results/sine/`. |
| **kepler.py** | Train classification transformers on Kepler data. Saves to `./results/kepler/`. |
| **kepler_cv.py** | Train regression transformers on Kepler (continuous) data. Saves to `./results/kepler_cv/`. |
| **kepler_cv_blocksize.py** | Train with varying block size (context length). Saves to `./results/kepler_cv_blocksize/`. |
| **model.py**, **model_cv.py** | Model definitions (shared by training scripts). |

## Checkpoint (Fig 2)

The **fig2_vafa_spatial_map** analysis loads a pretrained checkpoint. If `./ckpt` or `./ckpt.pt` is missing, the notebook will raise an error and ask you to download it.

1. Download the pretrained checkpoint (e.g. `ckpt.pt`) from the project’s release or documentation (e.g. the Google Drive folder linked in the notebook error message).
2. Place the file in **fig2_vafa_spatial_map/** as `./ckpt.pt`, or place the checkpoint file(s) in a `./ckpt/` folder there.
3. Re-run the notebook.

Run the notebook from **fig2_vafa_spatial_map/** so that `./ckpt.pt` or `./ckpt` is in the current working directory.

## Quick start

**Self-contained notebooks**
* `fig3a_*.ipynb` contains a 1D sine-wave example.
* `fig4_*.ipynb` contains a 2D Kepler example. 

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. The list includes: PyTorch (>=2.0), NumPy, SciPy, Matplotlib, scikit-learn, PyYAML, and optionally tqdm for progress bars. Figure notebooks may use extra plotting options; install any missing packages as prompted.
