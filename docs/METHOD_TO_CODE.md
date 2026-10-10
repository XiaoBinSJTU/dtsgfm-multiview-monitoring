# Method-to-Code Mapping

This document maps the manuscript's core computational stages to the public implementation. 

## Tensor interface

The reference model accepts:

- `signals`: `[B, V, L]`, where `B` is batch size, `V=4` standardized views, and `L` is sequence length;
- `view_mask`: `[B, V]`, with `1` for an available view and `0` for an unavailable view.

The public view order is mechanical, acoustic, electrical, and thermal. This fixed interface supports the released physical relation template. 

## Data-layer modules

### Angle-domain alignment

- **Code:** `src/dtsgfm/preprocessing.py`
- **Entry:** `angle_domain_align`
- **Role:** evaluates bounded shifts against the mechanical reference and selects the highest-correlation aligned sequence for each available view.

### View-quality estimation

- **Code:** `src/dtsgfm/preprocessing.py`
- **Entry:** `estimate_view_quality`
- **Role:** converts missingness, roughness, and amplitude instability into a confidence value in `[0, 1]`, then applies view availability.

### Quality-aware masked reconstruction

- **Code:** `src/dtsgfm/preprocessing.py`, `src/dtsgfm/models.py`, and `src/dtsgfm/losses.py`
- **Entries:** `apply_quality_aware_mask`, `same_phase_smooth_target`, `reconstruction_loss`
- **Role:** assigns a larger continuous mask to lower-confidence views and trains the shared encoder through robust reconstruction supervision.

### Hierarchical encoding

- **Code:** `src/dtsgfm/models.py`
- **Entry:** `HierarchicalEncoder`
- **Role:** applies two resolution-reducing temporal convolution stages to every standardized view using shared parameters.

### Delay-aware cross-view correspondence

- **Code:** `src/dtsgfm/features.py`
- **Entry:** `DelayAwareAlignment.forward`
- **Role:** scores candidate feature shifts with query-key correspondence and a physical offset prior, returning aligned features and expected offsets.

## Physical-layer modules

### Directed physical relation template

- **Code:** `src/dtsgfm/features.py`
- **Entry:** `BayesianDirectedHypergraph.edges`
- **Role:** defines the public reference relation paths and relation types over the four standardized view nodes.

### Bayesian relation confidence

- **Code:** `src/dtsgfm/features.py`
- **Entry:** `BayesianDirectedHypergraph.forward`
- **Role:** updates a local propagation-strength posterior from source-target agreement and converts posterior uncertainty plus offset deviation into relation confidence.

### Quality-constrained propagation

- **Code:** `src/dtsgfm/features.py`
- **Entry:** `BayesianDirectedHypergraph.forward`
- **Role:** combines source availability, view quality, relation confidence, and relation-specific transforms to form physical reference states.

### Topology-consistency correction

- **Code:** `src/dtsgfm/models.py`
- **Entry:** `DTSGFM.forward`
- **Role:** blends observed and physical reference states through a confidence/deviation gate and a controlled residual correction.

## Joint objective

- **Code:** `src/dtsgfm/losses.py`
- **Entry:** `total_loss`
- **Terms:** task classification, masked reconstruction, aligned-state consistency, and quality-weighted physical consistency.

The loss returns a differentiable scalar and detached component summaries. The minimal demo verifies a real backward pass without performing research evaluation.

## Module switches

`DTSGFMOptions` and `ABLATION_OPTIONS` in `src/dtsgfm/models.py` expose component boundaries. They are retained to make the implementation easier to inspect and extend. 


