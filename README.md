# Digital Twin Spatiotemporal Graph Fusion Modeling for Low-Quality Multi-View Equipment Monitoring under Harsh Operating Conditions

[Bin Xiao's GitHub profile](https://github.com/XiaoBinSJTU)

## Paper information

- **Authors:** Bin Xiao, Liang Chen, Yu Zheng, and Jinsong Bao
- **Manuscript status:** Under Review

## Motivation

![Motivation: low-quality multi-view disturbance propagation](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/06-dtsgfm-disturbance.svg)

## Overall framework

![Overall DTSGFM framework](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/06-dtsgfm-framework.svg)

## Research highlights

DTSGFM addresses multi-view equipment monitoring when noise, drift, missing segments, and temporal offsets disturb both state representations and graph propagation. The method combines quality-aware temporal encoding, physically bounded cross-view alignment, Bayesian confidence for directed relations, and topology-consistency correction. This repository exposes the core differentiable implementation and a synthetic smoke test, without distributing industrial data, trained parameters, or paper-level experimental results.

## Method overview

### 1. Angle-domain alignment and quality-aware masking

Available views are aligned to a mechanical reference within a bounded shift interval. View confidence is estimated from missingness, roughness, and amplitude stability, then controls continuous masking for reconstruction supervision.

### 2. Delay-aware cross-view correspondence

`DelayAwareAlignment` evaluates candidate temporal shifts in the encoded feature space. A physical offset prior regularizes attention and produces aligned view features together with current response-offset estimates.

### 3. Bayesian directed hypergraph propagation

`BayesianDirectedHypergraph` applies relation-specific transformations along fixed directed physical paths. Local posterior mean and variance estimates are converted into confidence-weighted propagation strengths.

### 4. Topology-consistency correction and joint objective

The model compares observed node states with physical reference states, applies a learned consistency gate, and preserves plausible local residuals. The public loss combines classification, masked reconstruction, cross-view state consistency, and physical consistency terms.

## Method-to-code map

| Method component | Public implementation |
| --- | --- |
| Hierarchical temporal representation | [`src/dtsgfm/models.py`](src/dtsgfm/models.py), `HierarchicalEncoder` |
| Full two-layer fusion model | [`src/dtsgfm/models.py`](src/dtsgfm/models.py), `DTSGFM.forward` |
| Angle alignment and quality estimation | [`src/dtsgfm/preprocessing.py`](src/dtsgfm/preprocessing.py), `angle_domain_align`, `estimate_view_quality` |
| Quality-aware continuous masking | [`src/dtsgfm/preprocessing.py`](src/dtsgfm/preprocessing.py), `apply_quality_aware_mask` |
| Delay-aware correspondence | [`src/dtsgfm/features.py`](src/dtsgfm/features.py), `DelayAwareAlignment` |
| Bayesian directed propagation | [`src/dtsgfm/features.py`](src/dtsgfm/features.py), `BayesianDirectedHypergraph` |
| Joint training objective | [`src/dtsgfm/losses.py`](src/dtsgfm/losses.py), `total_loss` |
| Safe public configuration | [`configs/demo.yaml`](configs/demo.yaml) and [`src/dtsgfm/config.py`](src/dtsgfm/config.py) |
| Forward/loss/backward smoke test | [`examples/minimal_demo.py`](examples/minimal_demo.py) |

See [`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md) for tensor contracts and a more detailed mapping.

## Quick start

The following commands are intended for Windows PowerShell with Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python examples\minimal_demo.py
python -m pytest -q
```

### Input contract

- `signals`: floating tensor with shape `[batch, 4, sequence_length]`.
- `view_mask`: availability tensor with shape `[batch, 4]`.
- Standardized view order in this reference interface: mechanical, acoustic, electrical, thermal.

The model returns logits and intermediate aggregate tensors needed by the public loss functions.


## Author

**Bin Xiao** — [XiaoBinSJTU](https://github.com/XiaoBinSJTU)

