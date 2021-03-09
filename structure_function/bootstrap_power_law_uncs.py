# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
Code for computing bootstrap uncertainties for the power law fits to the UVW2 and X-ray bands.
"""

from astropy.modeling import fitting
from astropy.modeling.powerlaws import PowerLaw1D, BrokenPowerLaw1D
from matplotlib import pyplot as plt
import numpy as np


kernel = 'Matern'  # ['Matern' 'RQ'] are the options
n_samples = 50
bootstrap_samples = 500
broken = False


def bootstrap_sample(tau, mean, std, bootstrap_samples, broken=False):
    """
    Compute bootstrap statistics.

    :param tau: numpy array of tau values.
    :param mean: numpy array of mean sf values.
    :param std: numpy array of standard errors for sf values
    :param bootstrap_samples: int specifying number of bootstrap samples
    :broken: bool specifying whether or not to fit a broken power law as opposed to a power law.
    """

    n_data = len(tau)
    indices = []
    indices2 = []
    breaks = []

    for i in range(bootstrap_samples):

        np.random.seed(i)

        sample_indices = np.random.choice(n_data - 1, n_data - 1)
        tau_bs = tau[sample_indices]
        mean_bs = mean[sample_indices]
        std_bs = std[sample_indices]

        # Sanity check

        # plt.scatter(tau_bs, mean_bs)
        # plt.show()
        # plt.close()

        if broken:
            power_law_model = BrokenPowerLaw1D(alpha_1=1, alpha_2=1, x_break=120)
        else:
            power_law_model = PowerLaw1D(alpha=1)

        # Fit the GP UVW2 structure functions using a power law or broken power law
        fitter_plm = fitting.SimplexLSQFitter()

        # Large number of iterations required for optimiser convergence (maxiter argument)

        plm = fitter_plm(power_law_model, tau_bs, mean_bs, weights=1/std_bs, maxiter=1000)

        if broken:
            alpha1 = plm.alpha_1.value
            alpha2 = plm.alpha_2.value
            x_break = plm.x_break.value
            if x_break > 100 or x_break < 45:
                continue
            indices.append(alpha1)
            indices2.append(alpha2)
            breaks.append(x_break)

        else:
            alpha1 = plm.alpha.value
            indices.append(alpha1)

    if broken:

        bootstrap_mean = np.mean(indices)
        bootstrap_se = np.sqrt(np.sum((np.array(indices) - bootstrap_mean) ** 2) / len(indices))
        bootstrap_mean2 = np.mean(indices2)
        bootstrap_se2 = np.sqrt(np.sum((np.array(indices2) - bootstrap_mean2) ** 2) / len(indices2))
        break_mean = np.mean(breaks)
        break_se = np.sqrt(np.sum((np.array(breaks) - break_mean) ** 2) / len(breaks))

        return bootstrap_mean, bootstrap_se, bootstrap_mean2, bootstrap_se2, break_mean, break_se

    else:

        bootstrap_mean = np.mean(indices)
        bootstrap_se = np.sqrt(np.sum((np.array(indices) - bootstrap_mean) ** 2) / bootstrap_samples)

        return bootstrap_mean, bootstrap_se


if __name__ == '__main__':

    # Discard the first two points because tau < 10
    uv_gp_tao_plot = np.loadtxt(f'sf_gp_data/uv/{kernel}_{n_samples}_times')[2:]
    uv_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/uv/mean_structure_function_{n_samples}_{kernel}')[2:]

    # Divide by the square root of the number of samples in order to compute the standard error from the standard deviation.
    uv_gp_std_structure_function = np.loadtxt(f'sf_gp_data/uv/std_structure_function_{n_samples}_{kernel}')[2:]/np.sqrt(n_samples)
    uv_gp_mean_structure_function *= 1e29  # multiply by a large number to prevent error in fitting power laws.
    uv_gp_std_structure_function *= 1e29

    bootstrap_mean, bootstrap_se = bootstrap_sample(uv_gp_tao_plot, uv_gp_mean_structure_function,
                                                   uv_gp_std_structure_function, bootstrap_samples, broken)

    print(bootstrap_mean)
    print(bootstrap_se)

    # Discard the first two points because tau < 10
    x_ray_gp_tao_plot = np.loadtxt(f'sf_gp_data/xray/{kernel}_{n_samples}_times')[2:]
    x_ray_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/xray/mean_structure_function_{n_samples}_{kernel}')[2:]

    # Divide by sqrt the sample size to convert standard deviation to standard error
    x_ray_gp_std_structure_function = np.loadtxt(f'sf_gp_data/xray/std_structure_function_{n_samples}_{kernel}')[2:]/np.sqrt(n_samples)

    if kernel == 'Matern':

        broken=True
        bootstrap_mean, bootstrap_se, bootstrap_mean2, bootstrap_se2, break_mean, break_se = \
            bootstrap_sample(x_ray_gp_tao_plot, x_ray_gp_mean_structure_function, x_ray_gp_std_structure_function, bootstrap_samples,
                             broken)

        print(bootstrap_mean)
        print(bootstrap_se)
        print(bootstrap_mean2)
        print(bootstrap_se2)
        print(break_mean)
        print(break_se)

    else:

        bootstrap_mean, bootstrap_se = bootstrap_sample(x_ray_gp_tao_plot, x_ray_gp_mean_structure_function,
                                                        x_ray_gp_std_structure_function, bootstrap_samples, broken)

        print(bootstrap_mean)
        print(bootstrap_se)
