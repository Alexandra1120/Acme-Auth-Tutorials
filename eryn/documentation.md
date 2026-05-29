# Eryn Documentation

This page documents the actual Eryn package used by the ACME tutorial notebook. It is based on the upstream repository at <https://github.com/mikekatz04/Eryn> and is intended to complement the existing joint EM-GW parameter-estimation notebook, not replace it.

Eryn is a general-purpose MCMC sampler built on an `emcee`-like ensemble-sampler structure, but extended to support:

- multiple model branches,
- unknown source counts through reversible-jump MCMC,
- parallel tempering,
- customizable proposal moves,
- backends for persistent chain storage.

In the ACME notebook, Eryn is used in a comparatively simple way: as the sampler engine for joint electromagnetic and gravitational-wave inference for a Galactic binary. The package itself is more general than that single use case.

## Repository Layout

The upstream repository is organized as a standard Python package:

```text
Eryn/
  README.md
  pyproject.toml
  src/
    eryn/
      ensemble.py
      state.py
      model.py
      prior.py
      backends/
      moves/
      utils/
  docs/
  examples/
  tests/
```

### `src/eryn/`

This is the main package.

- `ensemble.py`: the main sampler driver.
- `state.py`: state containers for walkers, branches, and supplemental information.
- `model.py`: lightweight wrapper for likelihood/prior computation callables.
- `prior.py`: prior containers and helper distributions.
- `backends/`: in-memory and HDF-based chain storage.
- `moves/`: proposal mechanisms, including reversible-jump and tempering tools.
- `utils/`: periodic transforms, plotting helpers, updates, stopping rules, and utility functions.

### `docs/`

The upstream documentation is organized around the same conceptual modules:

- ensemble
- state
- backend
- moves
- prior
- temper
- utils

### `examples/`

The repository includes tutorial notebooks and example usage. These are the best place to see the intended package idioms in a full sampling workflow.

## Installation

The upstream README gives the simplest installation path as:

```bash
pip install eryn
```

For development from source:

```bash
git clone https://github.com/mikekatz04/Eryn.git
cd Eryn
pip install .
```

The core Python dependencies described in the README are lightweight:

- `numpy`
- `matplotlib`
- `tqdm`
- `corner`

Some optional features can also use packages such as `h5py`, `scipy`, and `cupy` depending on the backend or prior path being used.

## Core Concepts

Eryn’s documentation and code use a tree-like metaphor for inference problems.

### Walkers, Branches, and Leaves

- walkers: ensemble members, as in `emcee`
- branches: different model types or source classes
- leaves: individual components or sources within a branch

This is what lets Eryn support reversible-jump problems where the number of active sources is not fixed in advance.

### State

At any given sampler step, the current configuration is represented by a `State` object. This can contain:

- coordinates for each branch,
- log-likelihood values,
- log-prior values,
- active/inactive leaf indicators,
- optional supplemental information carried through the run.

### Priors

Priors are collected into `ProbDistContainer` objects. These let you define priors over:

- individual parameters,
- tuples of parameters,
- named parameters,
- branch-specific parameter sets.

### Moves

Proposal logic is modular. A run can combine one or more move objects with weights. Eryn includes:

- ensemble-style red-blue moves,
- Gaussian and other Metropolis-Hastings proposals,
- reversible-jump proposals,
- multiple-try proposals,
- grouped moves,
- temperature-control proposals.

## Main Sampler Class

### `eryn.ensemble.EnsembleSampler`

`EnsembleSampler` is the central class and main user entry point.

It manages:

- ensemble walkers,
- branch dimensionality and naming,
- leaf limits for each branch,
- likelihood evaluation,
- prior evaluation,
- parallel tempering,
- proposal scheduling,
- reversible-jump moves,
- backend storage,
- optional update and stopping hooks.

The constructor is deliberately broad because it covers both simple fixed-dimension runs and more advanced tree-based reversible-jump configurations.

Key ideas from the implementation:

- `ndims` can be a scalar, list, or dict by branch.
- `nleaves_max` and `nleaves_min` control variable source counts.
- `moves` and `rj_moves` are separate proposal schedules.
- `tempering_kwargs` activates `TemperatureControl` and sets the number of temperatures.
- `periodic` can wrap parameters with periodic boundary conditions.
- `backend` stores the chain either in memory or in HDF5.

## State And Supplemental Data

### `eryn.state.State`

`State` is the sampler’s runtime container for the current ensemble configuration.

It is designed to support more complicated shapes than a plain `(nwalkers, ndim)` array. That is necessary because Eryn can track:

- multiple temperatures,
- multiple walkers,
- multiple branches,
- multiple leaves per branch.

### `eryn.state.BranchSupplemental`

`BranchSupplemental` is a helper for carrying extra branch- or leaf-aligned objects through the sampler while preserving Eryn’s indexing conventions.

This is useful when a likelihood or move needs additional structured information attached to each branch or leaf, beyond just the coordinate arrays.

## Priors

### `eryn.prior.ProbDistContainer`

`ProbDistContainer` is the package’s main prior aggregator. It stores a mapping from parameter indices or names to probability distributions that implement `logpdf` and `rvs`.

The code supports:

- integer-indexed parameters,
- string-named parameters,
- tuples for coupled priors,
- optional CuPy-backed evaluation paths.

### Built-In Prior Helpers

The main convenience distributions exposed in `prior.py` are:

- `uniform_dist(...)`
- `log_uniform(...)`
- `MappedUniformDistribution(...)`

These are enough for many tutorial and moderately complex problems, while still allowing custom priors to be supplied.

## Proposal Moves

The `moves/` package is one of the main extension surfaces in Eryn.

### Move Categories

The upstream docs group the implemented moves into:

- Move base class: `Move`
- Metropolis-Hastings base and implementations: `MHMove`, `GaussianMove`, `DistributionGenerate`
- red-blue proposals: `RedBlueMove`, `StretchMove`
- group proposals: `GroupMove`, `GroupStretchMove`
- reversible-jump proposals: `ReversibleJumpMove`, `DistributionGenerateRJ`
- multiple-try proposals: `MultipleTryMove`, `MTDistGenMove`, `MTDistGenMoveRJ`
- utility proposal combiners: `CombineMove`

### Parallel Tempering

Parallel tempering is controlled through:

- `eryn.moves.tempering.TemperatureControl`
- `eryn.moves.tempering.make_ladder`

Tempering is activated from the sampler constructor using `tempering_kwargs`.

## Backends

Backends store the evolving chain state.

### `eryn.backends.Backend`

The default backend stores information in memory.

### `eryn.backends.HDFBackend`

The HDF backend stores chains and related diagnostics in an HDF5 file, which is useful for:

- long-running jobs,
- restarts,
- post-processing outside the current Python process.

The backend layer is modeled after the familiar `emcee` approach, but adapted to Eryn’s more complicated state shapes.

## Utility Modules

The `utils/` package contains several configuration helpers used in practical runs.

### Important Utility Classes

- `PeriodicContainer`: periodic parameter handling for wrapped coordinates
- `TransformContainer`: parameter transformations between sampling and model spaces
- update helpers from `updates.py`
- stopping helpers from `stopping.py`
- plotting helpers via `PlotContainer`

These are not just convenience extras. In many applications, especially astrophysical ones, the transform and periodic utilities are part of the actual model interface that the sampler sees.

## Short API Reference

| Class or function | Signature | Key arguments |
| --- | --- | --- |
| `EnsembleSampler` | `EnsembleSampler(nwalkers, ndims, log_like_fn, priors, ..., moves=None, rj_moves=None, tempering_kwargs={}, backend=None, vectorize=False, periodic=None, update_fn=None, stopping_fn=None, ...)` | `nwalkers`: ensemble size; `ndims`: parameter dimensions per branch; `log_like_fn`: likelihood callable; `priors`: prior container(s); `moves`: proposal schedule; `rj_moves`: reversible-jump moves; `tempering_kwargs`: activate parallel tempering; `backend`: storage backend |
| `State` | `State(...)` | holds coordinates, log-probability information, and branch structure for a sampler step |
| `BranchSupplemental` | `BranchSupplemental(obj_info, base_shape, copy=False)` | carry extra branch- or leaf-aligned information through the sampler |
| `ProbDistContainer` | `ProbDistContainer(priors_in, use_cupy=False, return_gpu=False)` | aggregate priors over parameter indices or names |
| `uniform_dist` | `uniform_dist(min, max, use_cupy=False, return_gpu=False)` | simple uniform prior helper |
| `log_uniform` | `log_uniform(min, max)` | log-uniform prior helper |
| `MappedUniformDistribution` | `MappedUniformDistribution(min, max, use_cupy=False, return_gpu=False)` | uniform prior mapped to zero-to-one support internally |
| `Backend` | `Backend(...)` | in-memory chain storage |
| `HDFBackend` | `HDFBackend(...)` | persistent HDF5 chain storage |
| `TemperatureControl` | `TemperatureControl(...)` | parallel-tempering ladder and swaps |
| `StretchMove` | `StretchMove(...)` | default ensemble-style red-blue proposal |
| `DistributionGenerateRJ` | `DistributionGenerateRJ(...)` | RJ proposal mechanism for variable source counts |
| `TransformContainer` | `TransformContainer(...)` | transform sampled parameters into model parameters |
| `PeriodicContainer` | `PeriodicContainer(...)` | periodic wrapping for angular or cyclic variables |

## How The ACME Notebook Uses Eryn

The local tutorial notebook is a specialized Eryn application rather than a generic Eryn tour.

In [parameter_estimation_eryn_ucb_gw.ipynb](parameter_estimation_eryn_ucb_gw.ipynb), the workflow is:

1. generate Galactic-binary gravitational-wave signals with `GBGPU`,
2. generate an electromagnetic light-curve model,
3. define a joint EM-GW likelihood,
4. define priors with `ProbDistContainer` and `uniform_dist`,
5. use `TransformContainer` to map sampled coordinates into physical waveform parameters,
6. initialize and run `EnsembleSampler`,
7. store and analyze the resulting chains, optionally through `HDFBackend`.

The notebook imports exactly the public surfaces one would expect for that use case:

- `EnsembleSampler`
- `HDFBackend`
- `ProbDistContainer`
- `uniform_dist`
- `State`
- `TransformContainer`

That is a useful signal that this page should focus on Eryn as a sampler framework, while the notebook remains the concrete astrophysical example.

## Minimal Eryn Usage Pattern

The upstream README presents Eryn as a package you normally start with:

```python
from eryn.ensemble import EnsembleSampler
```

A minimal conceptual pattern is:

```python
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist

priors = ProbDistContainer({
    0: uniform_dist(0.0, 1.0),
    1: uniform_dist(-1.0, 1.0),
})

sampler = EnsembleSampler(
    nwalkers=32,
    ndims=2,
    log_like_fn=my_log_likelihood,
    priors=priors,
)
```

From there, more advanced features are layered in as needed:

- multiple branches,
- variable leaf counts,
- parallel tempering,
- custom move schedules,
- HDF backends,
- transformations and periodic variables.

## Relationship To The Existing Tutorial

The existing ACME material remains the hands-on notebook:

- [Eryn Introduction](intro.md)
- [Joint EM-GW Parameter Estimation Notebook](parameter_estimation_eryn_ucb_gw.ipynb)

This new page is the code-reference companion that explains what Eryn itself is, how the package is structured, and which parts of the package the notebook is actually using.

## References

- Eryn repository: <https://github.com/mikekatz04/Eryn>
- Eryn documentation: <https://mikekatz04.github.io/Eryn/index.html>
- Eryn paper: <https://arxiv.org/abs/2303.02164>
- Zenodo record: <https://zenodo.org/records/17162828>
- `emcee` reference: <https://emcee.readthedocs.io/en/stable/>