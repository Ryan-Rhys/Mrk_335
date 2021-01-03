# Modelling the Multiwavelength Variability of Mrk-335 using Gaussian Processes

This code repository contains all source code for the paper.

## Installation

```
conda create -n mrk python==3.7
conda install astropy scikit-learn matplotlib
conda install -c conda-forge statsmodels
pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow
```

## Gaussian Process Fitting to Observational Data

## Lightcurve Simulations

The simulations folder contains code for performing lightcurve simulations according
to the Timmer and Konig algorithm.

## Structure Function Computation

## Autocorrelation

## Log-Normality Tests

The folder log_normal_tests contains code for distribution testing of the observational
data from Mrk 335.
