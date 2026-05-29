# Introduction

```{image} example.png
:alt: GWG example output
:align: center
:width: 70%
```

The ACME GWG session was held on May 8, 2026 and presented by Nikolaos Karnesis from Aristotle University of Thessaloniki. The session introduces `gwg`, a package for estimating the gravitational-wave confusion noise signal from compact Galactic binaries as measured by LISA.

LISA will measure a very large number of transient and quasi-monochromatic sources. A full Global Fit strategy can model this population with a large-scale blocked Gibbs sampling scheme, but that approach can be computationally expensive for forecast studies. GWG provides a lower-cost alternative based on SNR criteria and simplifying assumptions, acting as a practical simulation of the Global Fit pipeline.

The original event page is available on [Indico](https://indico.global/event/17933/), and the source code is available at [gitlab.in2p3.fr/Nikos/gwg](https://gitlab.in2p3.fr/Nikos/gwg). The session slides are included here as {download}`20260508_gwg_tutorial.pdf`.

## Installation

The GWG README recommends using a dedicated conda environment:

```bash
conda create -n gwg_env -c conda-forge gsl numpy scipy jupyter pandas matplotlib xarray h5py tqdm pytables astropy python=3.12
conda activate gwg_env
```

Install `lisaanalysistools`:

```bash
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools
pip install .
```

Install the Galactic-binary waveform package:

```bash
pip install gbgpu
```

Then install GWG:

```bash
git clone https://gitlab.in2p3.fr/Nikos/gwg.git
cd gwg
pip install .
```

Optional smoothing and waveform dependencies can also be useful:

```bash
pip install whittaker-eilers
pip install fastgb
```

Depending on the machine, the LISA-related packages may need to be installed from source. GPU execution also requires a compatible CuPy installation.

## Usage

GWG needs TDI A, E, and T noise curves. The package includes a basic analytic LISA noise utility:

```python
noise = gwg.utils.lisa_noise(f=fvec)

lisa_noise = {
    "A": noise._noise_psd_AA().astype(np.float64),
    "E": noise._noise_psd_AA().astype(np.float64),
    "T": noise._noise_psd_TT().astype(np.float64),
    "f": fvec.astype(np.float64),
}
```

The basic workflow has two stages:

1. Generate a compact Galactic binary catalogue and the associated A, E, T frequency-domain data.
2. Run the iterative confusion-noise reduction loop.

For data generation, create a `GBGPU` waveform object and call `gwg.generate_data`:

```python
import gwg
from gbgpu.gbgpu import GBGPU
from lisatools.detector import EqualArmlengthOrbits

GB = GBGPU(force_backend="cpu", orbits=EqualArmlengthOrbits())
cat = gwg.utils.load_h5(my_h5_catalog_name, key="cat")

tdi, cat = gwg.generate_data(cat, lisa_noise, GB, AET=True)
gwg.utils.to_h5(my_h5_data_name, cat=cat, tdi=tdi)
```

Once the data are available, run the main loop:

```python
AET, S1, S1r, cat = gwg.icloop(
    AET,
    GB,
    cat,
    lisa_noise,
    Df,
    maxiter=3,
    snr_thresh2=snr_thresh2**2,
    tol=0.1,
    doplot=True,
    logger=logger,
    oversample=4,
    methoduse="median",
)
```

The notebook in this section walks through a compact tutorial example using a generated demo catalogue and writes `minimal_gwg_output.h5` locally when executed.

## Citation

For scientific use, cite [Karnesis, Babak, Pieroni, Cornish, and Littenberg, Phys. Rev. D 104, 043019](https://doi.org/10.1103/PhysRevD.104.043019). The implementation is by Nikos Karnesis, Stas Babak, and Maude Le Jeune, with code optimisation by Federico Pozzoli.
