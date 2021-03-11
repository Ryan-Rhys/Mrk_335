# Modelling the Multiwavelength Variability of Mrk-335 using Gaussian Processes

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This code repository contains all source code for the paper given in the title, where we interpolate the gaps in the observational
lightcurves of Mrk 335 using a Gaussian process and use these "GP Lightcurves" to perform a 
cross-correlation analysis. The gif below shows the process for obtaining boostrap uncertainty estimates
on the parameters for the power law fits to the Gaussian process structure functions.

<p align="center">
  <img width="600" src="bootstrap_slowish.gif" title="Bootstrap Uncertainty Computation">
</p>

## Installation

We recommend using a conda environment.

```
conda create -n mrk python==3.7
conda install astropy scikit-learn matplotlib
conda install -c conda-forge statsmodels
pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow
pip install scipy
```

## Gaussian Process Fitting to Observational Data

The gp_fit_real_data folder contains the code for fitting Gaussian processes to
the observational data

X-ray            |  UVW2
:-------------------------:|:-------------------------:
<img src="xray_gp.png" width="400" title="X-ray Band GP Lightcurve">|    <img src="uv_gp.png" width="400" title="UVW2 Band GP Lightcurve">

## Lightcurve Simulations

The simulations folder contains code for performing lightcurve simulations according
to the Timmer and Konig algorithm.

<img src="sim_table.png" width="800" title="Simulation Results Table">

## Structure Function Computation

The structure_function folder contains the code for computing structure functions
of both the observational lightcurves and the Gaussian process-interpolated lightcurves
of Mrk 335.

<img src="sf.png" width="800" title="Gaussian Process Structure Functions">

## Log-Normality Tests

The folder log_normal_tests contains code for distribution testing of the observational
data from Mrk 335, including both graphical distribution tests such as PP-plots, ECDFs and histograms
as well as statistical hypothesis testing using the Kolmogorov-Smirnov test.

<img src="repo_hists.png" width="800" title="Histograms of the Observational Data">
