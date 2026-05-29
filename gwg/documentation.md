# GWG Documentation

This page documents the actual upstream `gwg` package structure from <https://gitlab.in2p3.fr/Nikos/gwg>, not just the ACME tutorial notebook. The package is compact: the public API is effectively built from three Python modules plus a few example and helper scripts.

`gwg` estimates the Galactic-binary confusion foreground in the LISA band by combining two stages:

1. generate noiseless TDI data and per-source optimal SNR values from a source catalogue,
2. iteratively subtract resolvable sources while re-estimating the residual confusion PSD.

## Repository Layout

The upstream repository has five top-level areas that matter for users:

```text
gwg/
  __init__.py
  generate_data.py
  gwg.py
  utils.py
example/
gendata/
README.md
setup.py
```

### `gwg/`

This is the installable Python package.

- `__init__.py`: re-exports symbols from `gwg.py` and `generate_data.py`.
- `generate_data.py`: waveform synthesis, TDI accumulation, and source-by-source SNR computation.
- `gwg.py`: noise interpolation, smoothing, SNR evaluation, source subtraction, and the main iterative loop.
- `utils.py`: shared utilities, including the LISA noise model, `FrequencySeries`, demo catalogue generation, HDF5 I/O, and logger setup.

### `example/`

This directory contains notebooks, a pipeline script, and example plots. It is the best place to see how the package is intended to be driven from user code.

### `gendata/`

This contains older standalone helper scripts for generating and merging data files. These are useful for understanding historical workflows, but they are not the core package API.

### `setup.py`

The package metadata is minimal. Installation registers a single package named `gwg`.

## Public API Surface

The package initializer is small:

```python
from .gwg import *
from .generate_data import *
```

In practice, the main public entry points are:

- `gwg.generate_data(...)`
- `gwg.icloop(...)`
- `gwg.get_instr_noise(...)`
- `gwg.psd_smooth_moving_average(...)`
- `gwg.fast_batch_snr(...)`
- `gwg.generate_template(...)`
- `gwg.utils.lisa_noise`
- `gwg.utils.FrequencySeries`
- `gwg.utils.make_demo_catalog(...)`
- `gwg.utils.to_h5(...)`
- `gwg.utils.load_h5(...)`

The practical user-facing API is therefore centered on two calls: `generate_data` for data synthesis and `icloop` for iterative subtraction.

### Short API Reference

| Function | Signature | Key arguments |
| --- | --- | --- |
| `gwg.generate_data` | `generate_data(cat, noise, GB, T=None, dt=None, AET=False, oversample=1, logger=None, t0=0.0, duty_cycle=1, tdi2=True, noprogbar=False, batch_size=5000, gbgpu_available=False)` | `cat`: source catalogue; `noise`: instrument PSDs; `GB`: waveform object; `T`, `dt`: observation settings; `AET`: return `A/E/T`; `batch_size`: waveform batching; `gbgpu_available`: GPU path toggle |
| `gwg.icloop` | `icloop(AET, GB, cat, noise=None, deltaf=2000, maxiter=10, snr_thresh2=11, kappa=.95, tol=0.1, use_gbgpu=False, oversample=4, duty_cycle=1, dt=None, T=None, t0=0.0, tdi2=True, doplot=False, logger=None, nproc=1, batch_size=5000, tag="", **kwargs)` | `AET`: input TDI data; `cat`: source catalogue; `noise`: PSD model; `deltaf`: smoothing window; `snr_thresh2`: subtraction threshold; `tol`: convergence criterion; `maxiter`: iteration cap; `nproc`: multiprocessing |
| `gwg.get_instr_noise` | `get_instr_noise(noise, f, logger=None)` | `noise`: callable, dict, or array; `f`: target frequency grid |
| `gwg.psd_smooth_moving_average` | `psd_smooth_moving_average(AET, noise, N, methoduse="median", extra_smooth=False, extra_smooth_type="whittaker", norm=1/0.7023319615912207, order=1e7, logger=None)` | `AET`: residual channels; `noise`: instrument PSD; `N`: window size; `methoduse`: `mean` or `median`; `extra_smooth_type`: smoothing backend |
| `gwg.fast_batch_snr` | `fast_batch_snr(f, N, X, Y, Z, isnotaet=True, return_gpu=False)` | `f`: per-source frequency chunks; `N`: PSD dictionary; `X`, `Y`, `Z`: waveform channels; `isnotaet`: convert from `XYZ`; `return_gpu`: keep GPU arrays |
| `gwg.generate_template` | `generate_template(pvals, tmplt_clss, param_order=[...], **kwargs)` | `pvals`: source parameters; `tmplt_clss`: waveform backend such as `GBGPU`; `param_order`: field mapping passed into waveform generation |
| `gwg.utils.lisa_noise` | `lisa_noise(f=None, L=2.5e9, gen=2)` | `f`: frequency grid; `L`: arm length; `gen`: TDI generation used for PSD construction |
| `gwg.utils.FrequencySeries` | `FrequencySeries(array, df=None, kmin=None, t0=0, units=None, name=None, fs=None)` | `array`: frequency-domain data; `df`, `kmin`: grid definition; `fs`: explicit frequency axis |
| `gwg.utils.make_demo_catalog` | `make_demo_catalog(n_sources=100, n=-1.5, seed=12345, fmin=1e-4, fmax=5e-3, gamma=1, offset=-28)` | `n_sources`: catalogue size; `fmin`, `fmax`: frequency range; `gamma`, `offset`: amplitude distribution controls |
| `gwg.utils.to_h5` | `to_h5(fname, cat=None, tdi=None, S=None)` | `fname`: output file; `cat`: catalogue; `tdi`: TDI products; `S`: smoothed PSD output |
| `gwg.utils.load_h5` | `load_h5(fname, key=None)` | `fname`: HDF5 file; `key`: one of `cat`, `tdi`, or `S` |

## Core Modules

### `gwg.utils`

`utils.py` provides the shared data model and small infrastructure pieces used everywhere else.

#### `lisa_noise`

`lisa_noise` is a small class that builds an analytic LISA PSD for the `A`, `E`, and `T` channels from a frequency grid. Internally it computes:

- the transfer-function factor `cxx`,
- the test-mass noise term,
- the optical metrology noise term,
- the final `A/E` and `T` PSDs.

This is the simplest built-in way to construct the `noise` object expected throughout the package.

#### `FrequencySeries`

This helper wraps frequency-domain arrays as `xarray.DataArray` objects with a frequency coordinate and metadata such as `df`, `kmin`, and `t0`.

GWG uses this as its standard frequency-domain container for TDI channels and smoothed PSD outputs.

#### Catalogue and I/O Helpers

- `make_demo_catalog(...)` creates a synthetic recarray catalogue with the parameter names expected by the waveform layer.
- `to_h5(...)` stores catalogues, TDI products, and smoothed PSDs to HDF5 via `pandas`.
- `load_h5(...)` loads those saved products back into either a recarray or a dictionary.

These helpers encode the package's expected on-disk data model.

#### Other Utilities

- `XYZ2AET(...)` converts TDI `X`, `Y`, `Z` to `A`, `E`, `T`.
- `init_logger(...)` and `close_logger(...)` manage logging.

### `gwg.generate_data`

`generate_data.py` owns the forward-modeling stage.

#### Main Responsibility

`generate_data(...)` takes a catalogue, a noise description, and a waveform generator, then:

1. validates `T` and `dt`,
2. appends a `snr2` field to the catalogue,
3. builds an output frequency grid,
4. interpolates the instrumental noise to that grid,
5. generates waveforms in batches,
6. computes each source's optimal SNR squared,
7. accumulates the contribution of each source into global TDI channels,
8. optionally converts the output from `XYZ` to `AET`.

The function returns:

- `tdi_sum`: a dictionary of `FrequencySeries` objects,
- `cat`: the input catalogue augmented with `snr2`.

#### GPU Path

The module contains explicit GPU-aware logic:

- it tries to import `cupy` as `xp`,
- it uses `complex64` buffers on GPU for scatter-add operations,
- it provides `XYZ2AET_gpu(...)` for GPU-native channel conversion,
- it moves arrays back to CPU before packaging results in `FrequencySeries`.

The code is therefore designed around a dual backend model: NumPy for CPU execution and CuPy for accelerated waveform accumulation.

#### Dependency on the Waveform Layer

`generate_data(...)` does not build waveforms itself. It delegates waveform generation to `gwg.generate_template(...)`, which expects a supported waveform object such as `GBGPU`.

### `gwg.gwg`

`gwg.py` implements the reduction loop and most of the analysis logic.

#### `get_instr_noise(...)`

This function normalizes the noise input into the dictionary format used internally:

- callable: evaluated on the requested frequency grid,
- dictionary: reused directly or interpolated,
- NumPy array: interpreted as `f, A, E, T` columns.

The return value is always a dictionary with keys `A`, `E`, `T`, and `f`.

#### `psd_smooth_moving_average(...)`

This function estimates the total PSD from the current residual TDI data.

It first computes $2 \, df \, |AET|^2$ channel by channel, then applies one of two local estimators:

- running mean,
- running median.

After that, it adds the instrumental noise and can optionally apply a second smoothing stage:

- `whittaker`,
- `savgol`,
- kernel smoothing fallback.

It returns two dictionaries:

- `S`: the final smoothed PSD,
- `Sr`: the pre-extra-smoothing PSD.

#### `fast_batch_snr(...)`

This is the internal batched SNR engine. Given waveform chunks and a noise model, it:

- converts `XYZ` to `AET` when needed,
- gathers the matching noise samples for each source and frequency bin,
- evaluates the inner product,
- returns SNR squared values, optionally alongside the assembled TDI data cube.

This function is used both during data generation and during iterative subtraction.

#### `generate_template(...)`

This is the abstraction boundary between GWG and the waveform provider. The code currently supports:

- `GBGPU`,
- legacy `FastGB`-style classes when present.

Its role is to standardize the return values as `f, X, Y, Z` irrespective of the waveform backend.

#### `local_subtraction(...)`

This function performs one subtraction pass over a catalogue chunk. For each batch it:

- generates templates,
- computes current SNR values against the present PSD estimate,
- selects sources above `snr_thresh2`,
- accumulates only the loud-source waveforms,
- returns channel-wise subtraction buffers and a boolean mask of removed sources.

It is written so it can be used either in a single process or inside the multiprocessing branch of `icloop(...)`.

#### `icloop(...)`

This is the main GWG algorithm.

At a high level it performs:

1. convert input data to `AET` if needed,
2. coerce raw arrays into `FrequencySeries` containers,
3. interpolate the instrumental noise to the TDI grid,
4. optionally pre-filter the catalogue using the stored `snr2` and a safety factor `kappa`,
5. estimate an initial smoothed PSD `S0`,
6. subtract all sources whose updated SNR exceeds the threshold,
7. recompute the PSD as `S1`,
8. test convergence,
9. repeat until convergence, no more loud sources remain, or `maxiter` is reached,
10. recompute final SNR values for the recovered source catalogue.

It returns four objects:

- `AET`: residual TDI channels after subtraction,
- `S1`: final smoothed PSD,
- `S1r`: corresponding pre-extra-smoothing PSD,
- `scat`: the catalogue of recovered or subtracted sources.

## Data Model

### Catalogue Format

The code assumes a NumPy recarray-like catalogue. The tutorial and helper functions use fields such as:

- `Amplitude`
- `Frequency`
- `FrequencyDerivative`
- `FrequencySecondDerivative`
- `InitialPhase`
- `Inclination`
- `Polarization`
- `EclipticLongitude`
- `EclipticLatitude`

The code may append:

- `snr2`: optimal SNR squared.

The example scripts also show older catalogues with an optional `Name` field.

### Noise Format

Noise can be supplied in three forms:

- callable returning a dictionary,
- dictionary with `A`, `E`, `T`, and `f`,
- NumPy array with columns `f`, `A`, `E`, `T`.

Internally, GWG normalizes all of them to the dictionary form.

### TDI Format

TDI channels are represented as dictionaries keyed by `A`, `E`, `T` or `X`, `Y`, `Z`, with each channel stored as a `FrequencySeries` or an equivalent array that can be wrapped into one.

## Runtime Flow

The actual package structure suggests the following mental model:

```text
Catalogue recarray + noise model + waveform class
  -> generate_data(...)
  -> TDI data plus catalogue with snr2
  -> icloop(...)
  -> residual AET
  -> smoothed PSD estimates
  -> recovered source catalogue
```

The boundary is clean:

- `generate_data` creates the synthetic dataset,
- `icloop` performs the iterative reduction,
- `utils` defines the common containers and storage helpers.

## Optional and External Dependencies

The upstream code relies on a small core plus several optional accelerators.

### Core Dependencies

- `numpy`
- `scipy`
- `xarray`
- `pandas`
- `matplotlib`
- `tqdm`
- `astropy`
- `lisaconstants`
- `gbgpu`

### Optional Dependencies

- `cupy` for GPU-backed arrays and faster batch accumulation,
- `whittaker-eilers` for extra PSD smoothing,
- legacy `FastGB` pathways in some code paths and older helper scripts.

## Notable Implementation Details

- The package is intentionally small and function-oriented rather than class-heavy.
- Public imports are wildcard exports, so the effective API is broader than a curated stable surface.
- `icloop(...)` uses stored `snr2` values as a pre-selection heuristic before the more expensive iterative subtraction begins.
- The smoothing layer is central to the convergence behavior; it is not just cosmetic post-processing.
- The example and `gendata/` scripts include older workflow pieces that are useful context but are not required to use the modern `generate_data(...)` and `icloop(...)` interface.

## Minimal Code-Oriented Usage

The minimal package-level flow, matching the upstream module design, is:

```python
import gwg
from gbgpu.gbgpu import GBGPU
from lisatools.detector import EqualArmlengthOrbits

cat = gwg.utils.make_demo_catalog(n_sources=1000)

noise_model = gwg.utils.lisa_noise(f=fvec)
noise = {
    "A": noise_model._noise_psd_AA(),
    "E": noise_model._noise_psd_AA(),
    "T": noise_model._noise_psd_TT(),
    "f": fvec,
}

GB = GBGPU(force_backend="cpu", orbits=EqualArmlengthOrbits())

tdi, cat = gwg.generate_data(cat, noise, GB, T=duration, dt=dt, AET=True)
AET, S1, S1r, recovered = gwg.icloop(tdi, GB, cat, noise, deltaf=2000)
```

## Companion Material

- Introductory overview: [GWG Introduction](intro.md)
- Executable walkthrough: [GWG Tutorial Notebook](gwg_tutorial.ipynb)
- Slides: {download}`20260508_gwg_tutorial.pdf`

## References

- Upstream source repository: <https://gitlab.in2p3.fr/Nikos/gwg>
- ACME Hands-on session page: <https://www.acme-astro.eu/hands-on-sessions/>
- Session event page: <https://indico.global/event/17933/>
- Reference paper: <https://doi.org/10.1103/PhysRevD.104.043019>