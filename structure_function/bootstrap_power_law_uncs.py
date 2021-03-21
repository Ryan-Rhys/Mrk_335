# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
Code for computing bootstrap uncertainties for the power law fits to the UVW2 and X-ray bands.
"""

from astropy.modeling import fitting
from astropy.modeling.powerlaws import PowerLaw1D, BrokenPowerLaw1D
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

kernel = 'Matern'  # ['Matern' 'RQ'] are the options
n_samples = 50
bootstrap_samples = 200
broken = False


def bootstrap_sample(tau, mean, std, bootstrap_samples, broken=False, uv=False, gif=True):
    """
    Compute bootstrap statistics.

    :param tau: numpy array of tau values.
    :param mean: numpy array of mean sf values.
    :param std: numpy array of standard errors for sf values
    :param bootstrap_samples: int specifying number of bootstrap samples
    :param broken: bool specifying whether or not to fit a broken power law as opposed to a power law.
    :param uv: bool specifying whether the fit is to the UVW2 band
    :param gif: bool specifying whether to collect figures for gif of bootstrapping procedure
    """

    n_data = len(tau)
    indices = []
    indices2 = []
    breaks = []
    amplitudes = []

    for i in range(bootstrap_samples):

        np.random.seed(i)

        sample_indices = np.random.choice(n_data - 1, n_data - 1)
        tau_bs = tau[sample_indices]
        mean_bs = mean[sample_indices]
        std_bs = std[sample_indices]

        if broken:
            if uv:
                power_law_model = BrokenPowerLaw1D(alpha_1=-1, alpha_2=0, x_break=110)
            else:
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
            amp = plm.amplitude.value
            if uv:
                if x_break < 110 or x_break > 150:
                    continue
            else:
                if x_break > 100 or x_break < 45:
                    continue
            indices.append(alpha1)
            indices2.append(alpha2)
            breaks.append(x_break)
            amplitudes.append(amp)

        else:
            alpha1 = plm.alpha.value
            amp = plm.amplitude.value
            indices.append(alpha1)
            amplitudes.append(amp)

        if gif:
            if uv:

                # Discard the first two points because tau < 10
                uv_gp_tao_plot = np.loadtxt(f'sf_gp_data/uv/{kernel}_{n_samples}_times')[2:]

                # Observational Data

                uv_tao_plot = np.loadtxt(f'sf_data/uv/times_most_recent')[2:]
                uv_mean_structure_function = np.loadtxt(f'sf_data/uv/structure_function_most_recent')[2:]
                uv_std_structure_function = np.loadtxt(f'sf_data/uv/errors_most_recent')[2:]

                fig, ax = plt.subplots(1)
                color = 'tab:blue'
                plt.xscale('log')
                ax.errorbar(uv_tao_plot, uv_mean_structure_function, yerr=uv_std_structure_function, fmt='o', color=color,
                            markersize=3, linewidth=0.5, label='Observational')
                ax.set_yscale('log')
                plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
                ax.set_ylabel('Observational SF', color=color, fontsize=12)
                ax.yaxis.set_minor_locator(mticker.NullLocator())
                plt.yticks([0.1, 1], color=color)
                plt.xlim([10, 600])

                color = 'tab:red'
                ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
                ax2.plot(uv_gp_tao_plot, plm(uv_gp_tao_plot), label='Broken Power Law', color='k')
                ax2.errorbar(tau_bs, mean_bs, yerr=std_bs.flatten(),
                             fmt='x', color=color, markersize=3, linewidth=0.5, label='GP')
                ax2.set_yscale('log')
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.yaxis.set_minor_locator(mticker.NullLocator())
                plt.tight_layout()
                fig.legend(loc=4, bbox_to_anchor=[0.25, 0.25, 0.6, 0])
                plt.savefig(f'gif_figures/uv_gif_{kernel}_{i}.png')
                plt.close()

            else:

                # Discard the first two points because tau < 10
                x_ray_gp_tao_plot = np.loadtxt(f'sf_gp_data/xray/{kernel}_{n_samples}_times')[2:]

                # Load observational data
                x_ray_tao_plot = np.loadtxt(f'sf_data/xray/times_most_recent')[2:]
                x_ray_mean_structure_function = np.loadtxt(f'sf_data/xray/structure_function_most_recent')[2:]
                x_ray_std_structure_function = np.loadtxt(f'sf_data/xray/errors_most_recent')[2:]

                # plot on the same axis

                fig, ax = plt.subplots(1)
                plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
                color = 'tab:blue'
                ax.set_ylabel('Observational SF', fontsize=12, color=color)
                ax.errorbar(x_ray_tao_plot, x_ray_mean_structure_function, yerr=x_ray_std_structure_function,
                            fmt='o', markersize=3, linewidth=0.5, color=color, label='Observational')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.tick_params(axis='y', labelcolor=color)
                ax.set_ylim([0.5, 3.5])
                ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                plt.xticks([0.1, 1])
                ax.minorticks_off()
                if kernel == 'Matern':
                    scale_factor = 0.9
                else:
                    scale_factor = 1.35
                color = 'tab:red'
                ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
                if kernel == 'RQ':
                    ax2.plot(x_ray_gp_tao_plot, scale_factor * plm(x_ray_gp_tao_plot), color='k', label='Power Law')
                else:
                    ax2.plot(x_ray_gp_tao_plot, scale_factor * plm(x_ray_gp_tao_plot), color='k',
                             label=f'Break at {np.ceil(x_break)} days')
                ax.errorbar(tau_bs, scale_factor * mean_bs,
                            yerr=std_bs, fmt='x', color=color, markersize=3, linewidth=0.5,
                            label='GP')
                ax2.set_yscale('log')
                ax2.set_xscale('log')
                ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim([0.5, 3.5])
                plt.xlim([10, 700])
                plt.tight_layout()
                fig.legend(loc=4, bbox_to_anchor=[0.275, 0.67, 0.15, 0])
                ax2.yaxis.set_minor_locator(mticker.NullLocator())
                plt.savefig(f'gif_figures/xray_{kernel}_{i}.png')

                plt.close()

    if broken:

        bootstrap_mean = np.mean(indices)
        bootstrap_se = np.sqrt(np.sum((np.array(indices) - bootstrap_mean) ** 2) / len(indices))
        bootstrap_mean2 = np.mean(indices2)
        bootstrap_se2 = np.sqrt(np.sum((np.array(indices2) - bootstrap_mean2) ** 2) / len(indices2))
        break_mean = np.mean(breaks)
        break_se = np.sqrt(np.sum((np.array(breaks) - break_mean) ** 2) / len(breaks))
        amplitude_mean = np.mean(amplitudes)
        amplitude_se = np.sqrt(np.sum((np.array(amplitudes) - amplitude_mean) ** 2) / len(amplitudes))

        return bootstrap_mean, bootstrap_se, bootstrap_mean2, bootstrap_se2, break_mean, break_se, \
               amplitude_mean, amplitude_se

    else:

        bootstrap_mean = np.mean(indices)
        bootstrap_se = np.sqrt(np.sum((np.array(indices) - bootstrap_mean) ** 2) / bootstrap_samples)
        amplitude_mean = np.mean(amplitudes)
        amplitude_se = np.sqrt(np.sum((np.array(amplitudes) - amplitude_mean) ** 2) / len(amplitudes))

        return bootstrap_mean, bootstrap_se, amplitude_mean, amplitude_se


if __name__ == '__main__':

    # Discard the first two points because tau < 10
    uv_gp_tao_plot = np.loadtxt(f'sf_gp_data/uv/{kernel}_{n_samples}_times')[2:]
    uv_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/uv/mean_structure_function_{n_samples}_{kernel}')[2:]

    # Divide by the square root of the number of samples in order to compute the standard error from the standard deviation.
    uv_gp_std_structure_function = np.loadtxt(f'sf_gp_data/uv/std_structure_function_{n_samples}_{kernel}')[2:]/np.sqrt(n_samples)
    uv_gp_mean_structure_function *= 1e29  # multiply by a large number to prevent error in fitting power laws.
    uv_gp_std_structure_function *= 1e29

    bootstrap_mean, bootstrap_se, bootstrap_mean2, bootstrap_se2, break_mean, break_se, amplitude_mean, amplitude_se = \
        bootstrap_sample(uv_gp_tao_plot, uv_gp_mean_structure_function, uv_gp_std_structure_function, bootstrap_samples, broken=True, uv=True)

    print(bootstrap_mean)
    print(bootstrap_se)
    print(bootstrap_mean2)
    print(bootstrap_se2)
    print(break_mean)
    print(break_se)

    # Plotting bootstrapped parameter values for power law fits to UVW2 data

    power_law_model = BrokenPowerLaw1D(alpha_1=bootstrap_mean, alpha_2=bootstrap_mean2, x_break=break_mean, amplitude=amplitude_mean)
    fitter_plm = fitting.SimplexLSQFitter()
    # Set iterations to 0 to just plot the best power law parameters
    plm = fitter_plm(power_law_model, uv_gp_tao_plot, uv_gp_mean_structure_function, weights=1/uv_gp_std_structure_function,
                      maxiter=0)

    # Observational Data

    uv_tao_plot = np.loadtxt(f'sf_data/uv/times_most_recent')[2:]
    uv_mean_structure_function = np.loadtxt(f'sf_data/uv/structure_function_most_recent')[2:]
    uv_std_structure_function = np.loadtxt(f'sf_data/uv/errors_most_recent')[2:]

    fig, ax = plt.subplots(1)
    color = 'tab:blue'
    plt.xscale('log')
    ax.errorbar(uv_tao_plot, uv_mean_structure_function, yerr=uv_std_structure_function, fmt='o', color=color,
                markersize=3, linewidth=0.5, label='Observational')
    ax.set_yscale('log')
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    ax.set_ylabel('Observational SF', color=color, fontsize=12)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    plt.yticks([0.1, 1], color=color)
    plt.xlim([10, 600])

    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(uv_gp_tao_plot, plm(uv_gp_tao_plot), label='Broken Power Law', color='k')
    ax2.errorbar(uv_gp_tao_plot, uv_gp_mean_structure_function, yerr=uv_gp_std_structure_function.flatten(),
                 fmt='x', color=color, markersize=3, linewidth=0.5, label='GP')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.25, 0.25, 0.6, 0])
    plt.savefig(f'cosmetic_figures/gp_structure_function_uv_{kernel}_on_same_axis_'
                f'alpha_is_{str(100*bootstrap_mean)}_alpha2_is_{str(100*bootstrap_mean2)}_x_break_is_{str(100*break_mean)}.png')
    plt.close()

    # Discard the first two points because tau < 10
    x_ray_gp_tao_plot = np.loadtxt(f'sf_gp_data/xray/{kernel}_{n_samples}_times')[2:]
    x_ray_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/xray/mean_structure_function_{n_samples}_{kernel}')[2:]

    # Divide by sqrt the sample size to convert standard deviation to standard error
    x_ray_gp_std_structure_function = np.loadtxt(f'sf_gp_data/xray/std_structure_function_{n_samples}_{kernel}')[2:]/np.sqrt(n_samples)

    if kernel == 'Matern':

        broken=True
        bootstrap_mean_xray, bootstrap_se_xray, bootstrap_mean2_xray, bootstrap_se2_xray, break_mean_xray, break_se_xray, \
        amplitude_mean_xray, amplitude_se_xray = bootstrap_sample(x_ray_gp_tao_plot, x_ray_gp_mean_structure_function,
                                                                  x_ray_gp_std_structure_function, bootstrap_samples, broken)

        print(bootstrap_mean_xray)
        print(bootstrap_se_xray)
        print(bootstrap_mean2_xray)
        print(bootstrap_se2_xray)
        print(break_mean_xray)
        print(break_se_xray)

    else:

        bootstrap_mean_xray, bootstrap_se_xray, amplitude_mean_xray, amplitude_se_xray = \
            bootstrap_sample(x_ray_gp_tao_plot, x_ray_gp_mean_structure_function, x_ray_gp_std_structure_function,
                             bootstrap_samples, broken)

        print(bootstrap_mean_xray)
        print(bootstrap_se_xray)

    if kernel == 'Matern':
        power_law_model = BrokenPowerLaw1D(alpha_1=bootstrap_mean_xray, alpha_2=bootstrap_mean2_xray, x_break=break_mean_xray, amplitude=amplitude_mean_xray)
    else:
        power_law_model = PowerLaw1D(alpha=bootstrap_mean_xray, amplitude=amplitude_mean_xray)
    fitter_plm = fitting.SimplexLSQFitter()
    # Set iterations to 0 to just plot the best power law parameters
    plm_xray = fitter_plm(power_law_model, x_ray_gp_tao_plot, x_ray_gp_mean_structure_function, weights=1/x_ray_gp_std_structure_function,
                      maxiter=0)

    # Load observational data

    x_ray_tao_plot = np.loadtxt(f'sf_data/xray/times_most_recent')[2:]
    x_ray_mean_structure_function = np.loadtxt(f'sf_data/xray/structure_function_most_recent')[2:]
    x_ray_std_structure_function = np.loadtxt(f'sf_data/xray/errors_most_recent')[2:]

    # plot on the same axis

    fig, ax = plt.subplots(1)
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    color = 'tab:blue'
    ax.set_ylabel('Observational SF', fontsize=12, color=color)
    ax.errorbar(x_ray_tao_plot, x_ray_mean_structure_function, yerr=x_ray_std_structure_function,
                fmt='o', markersize=3, linewidth=0.5, color=color, label='Observational')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0.5, 3.5])
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.xticks([0.1, 1])
    ax.minorticks_off()
    if kernel == 'Matern':
        scale_factor = 0.9
    else:
        scale_factor = 1.35
    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    if kernel == 'RQ':
        ax2.plot(x_ray_gp_tao_plot, scale_factor*plm_xray(x_ray_gp_tao_plot), color='k', label='Power Law')
    else:
        ax2.plot(x_ray_gp_tao_plot, scale_factor*plm_xray(x_ray_gp_tao_plot), color='k', label='Broken Power Law')
    ax.errorbar(x_ray_gp_tao_plot, scale_factor*x_ray_gp_mean_structure_function, yerr=x_ray_gp_std_structure_function, fmt='x', color=color, markersize=3, linewidth=0.5, label='GP')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.5, 3.5])
    plt.xlim([10, 700])
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.275, 0.75, 0.15, 0])
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    if kernel == 'Matern':
        plt.savefig(f'cosmetic_figures/gp_structure_function_xray_{kernel}_on_same_axis_{int(scale_factor*100)}_'
                    f'_alpha_1_is{str(100*bootstrap_mean_xray)}_alpha_2_is{str(100*bootstrap_mean2_xray)}'
                    f'_x_break_is{str(100*break_mean_xray)}.png')
    else:
        plt.savefig(f'cosmetic_figures/gp_structure_function_xray_{kernel}_on_same_axis_{int(scale_factor*100)}_'
                    f'_alpha_1_is{str(100*bootstrap_mean_xray)}.png')

    plt.close()
