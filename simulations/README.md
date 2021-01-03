# Lightcurve Simulations

Run the script light_curve_simulations.py

```
python light_curve_simulations.py
```

in order to generate gapped X-ray (Count rate) and UV lightcurves (Magnitudes) according to the 
Timmer and Konig algorithm. The simulated lightcurves are stored in the sim_curves folder.

The scripts fit_uv_sims.py and fit_xray_sims.py fit a Gaussian Process to the simulated UV and X-ray lightcurves
respectively.

```
python fit_uv_sims.py
```

Will generate relevant scores, RSS and NLML values, and store them in the uv_sims_stand folder. The scripts
stats.py will compute summary statistics for these scores. The fourier_methods.py and ts_gen.py scripts are required
for the lightcurve simulation script.
