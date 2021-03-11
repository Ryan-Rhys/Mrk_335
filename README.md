# Modelling the Multiwavelength Variability of Mrk-335 using Gaussian Processes

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This code repository contains all source code for the paper given in the title, where we interpolate the gaps in the observational
lightcurves of Mrk 335 using a Gaussian process and use these "GP Lightcurves" to perform a 
cross-correlation analysis.

<img src="bootstrap_slowish.gif" width="600" title="Bootstrap Uncertainty Computation">|


## Installation

```
conda create -n mrk python==3.7
conda install astropy scikit-learn matplotlib
conda install -c conda-forge statsmodels
pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow
pip install scipy
```

## Gaussian Process Fitting to Observational Data

X-ray            |  UVW2
:-------------------------:|:-------------------------:
<img src="xray_gp.png" width="400" title="X-ray Band GP Lightcurve">|    <img src="uv_gp.png" width="400" title="UVW2 Band GP Lightcurve">

## Lightcurve Simulations

The simulations folder contains code for performing lightcurve simulations according
to the Timmer and Konig algorithm.

## Structure Function Computation

<img src="sf.png" width="800" title="Histograms of the Observational Data">|

## Autocorrelation

## Log-Normality Tests

The folder log_normal_tests contains code for distribution testing of the observational
data from Mrk 335.

<img src="repo_hists.png" width="800" title="Histograms of the Observational Data">|
