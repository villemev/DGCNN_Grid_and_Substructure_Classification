# DGCNN-Based Estimation of Structural Grids and Substructures from IFC-Derived Geometry

This repository contains a Python implementation of a learning-based preprocessing framework for inferring **analytical structural grids** and **structural substructures** from **IFC-derived geometry**.

The code supports two complementary tasks on **unordered point sets** (typically column plan locations):
1. **Substructure grouping** (permutation-invariant): partitions points into locally consistent grid regions.
2. **Grid-index classification (UV)**: assigns discrete grid indices *(u, v)* to points within a (sub)structure.

The predicted abstractions (substructures + grid indices) are intended to act as a stable analytical backbone for downstream workflows, enabling more deterministic **alignment, snapping, and FEM-ready reconstruction** than rule-based IFC-to-FEM translation alone.

---

## Key features

- **Dynamic Graph CNN (DGCNN / EdgeConv)** backbones for point-set learning  
- **Permutation-invariant substructure training** via Hungarian-aligned cross-entropy
- **Adaptive kNN neighborhoods** for UV grid-index learning (bounded k)
- **Optional uniqueness regularization** to discourage duplicate grid-intersection assignments
- **Mask-aware batching** for variable-size structures (padding + boolean masks)
- Synthetic-data training support (orthogonal, skewed, curved, fan-shaped layouts)
- Designed for integration into **IFC-to-FEM preprocessing** pipelines (interactive inference)

---

## Repository structure (core files)

- `DGCNN_classification_UV_scripts.py`  
  Core model + utilities for UV grid-index prediction (DGCNN + attention + masking)

- `DGCNN_classification_UV_runner.py`  
  Training / evaluation runner for the UV model

- `DGCNN_classification_structure_scripts.py`  
  Core model + Hungarian alignment utilities for substructure grouping

- `DGCNN_classification_structure_runner.py`  
  Training / evaluation runner for the substructure model

- `JSON/`  
  Example JSON files illustrating the expected **training input format**  
  (synthetic point sets, labels, masks, and metadata used by the generators)

> Note: The repository is organized so the model logic lives in `*_scripts.py`, while `*_runner.py` files define the training protocol, logging, and checkpoints.

---

## Method overview

### Input representation
- Each structure is represented as an **unordered 2D point set** *(x, y)*.
- Variable-length point sets are padded to a fixed maximum and accompanied by a **mask**.
- Per-structure normalization: **centroid centering** and scaling by **median nearest-neighbor distance**.

### Task A — Substructure grouping
- Outputs a substructure ID per point.
- Uses fixed kNN neighborhood size and directional refinement.
- Trained with **Hungarian alignment** to resolve label permutation ambiguity.

### Task B — UV grid-index classification
- Outputs discrete *(u, v)* indices per point, with an explicit **outlier class** per axis.
- Uses **adaptive kNN** and a lightweight attention-based global context module.
- Trained with cross-entropy on u and v (optionally + duplicate assignment penalty).
