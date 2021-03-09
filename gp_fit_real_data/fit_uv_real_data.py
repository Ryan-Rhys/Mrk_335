# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script fits a homoscedastic GP to the Mrk 335 UV data.
"""

import pickle

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary, set_trainable
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

fix_noise = True  # Whether to fix the noise level
generate_samples = False  # Whether to generate samples from the best-fit kernels.
plot_mean = False  # Whether to plot the GP mean or the samples
n_samples = 1000  # number of samples to generate

m = None


def objective_closure():
    """
    Objective function for GP-optimization
    """
    return -m.log_marginal_likelihood()


if __name__ == '__main__':

    np.random.seed(42)  # Set the same random seed for training

    # Load the UV band data.

    with open('../processed_data/uv/uv_fl_times.pickle', 'rb') as handle:
        time = pickle.load(handle).reshape(-1, 1)
    with open('../processed_data/uv/uv_band_flux.pickle', 'rb') as handle:
        uv_band_flux = pickle.load(handle).reshape(-1, 1)
    with open('../processed_data/uv/uv_band_flux_errors.pickle', 'rb') as handle:
        uv_band_flux_errors = pickle.load(handle).reshape(-1, 1)

    snr = np.mean(uv_band_flux)/np.mean(uv_band_flux_errors)  # signal to noise ratio is ca. 30 so we ignore measurement noise.
    uv_band_flux_orig = uv_band_flux # original uv band flux for plotting.

    # We standardise the outputs

    flux_scaler = StandardScaler()
    uv_band_flux = flux_scaler.fit_transform(uv_band_flux)

    # We do kernel selection by comparison of the negative log marginal likelihood.

    k1 = gpflow.kernels.RBF()
    k2 = gpflow.kernels.Matern12()
    k3 = gpflow.kernels.Matern32()
    k4 = gpflow.kernels.Matern52()
    k5 = gpflow.kernels.RationalQuadratic()
    kernel_list = [k1, k2, k3, k4, k5]

    kernel_dict = {kernel_list[0]: 'RBF_Kernel', kernel_list[1]: 'Matern_12_Kernel', kernel_list[2]: 'Matern_32_Kernel',
                   kernel_list[3]: 'Matern_52_Kernel', kernel_list[4]: 'Rational_Quadratic_Kernel'}

    for k in kernel_list:

        name = kernel_dict[k]

        # GP uses a constant mean function, where the constant is set to be the empirical average of the standardised
        # counts

        m = gpflow.models.GPR(data=(time, uv_band_flux),
                              mean_function=Constant(np.mean(uv_band_flux)),
                              kernel=k,
                              noise_variance=1)

        if fix_noise:

            # Fix a noise level to be the average experimental error observed in the dataset (0.037) for magnitudes
            # Noise level is 2.0364e-15 for the flux values.
            # Standardisation destroys this information so setting noise to be mean of standardised values divided by
            # the SNR in the orignal space.

            fixed_noise = np.mean(np.abs(uv_band_flux/snr))
            set_trainable(m.likelihood.variance, False)  # We don't want to optimise the noise level in this case.
            m.likelihood.variance = fixed_noise

        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
        print_summary(m)

        # We specify the grid of time points on which we wish to predict the count rate
        time_test = np.arange(54236, 58630, 1, dtype=np.float64).reshape(-1, 1)
        mean, var = m.predict_y(time_test)

        log_lik = m.log_marginal_likelihood()

        lower = mean[:, 0] - 2 * np.sqrt(var[:, 0])  # 1 standard deviation is common in astrophysics
        upper = mean[:, 0] + 2 * np.sqrt(var[:, 0])

        if generate_samples:

            # Sample from posterior of best-fit kernels.

            if name == 'Matern_12_Kernel' or name == 'Rational_Quadratic_Kernel':

                samples = tf.squeeze(m.predict_f_samples(time_test, n_samples))
                samples = flux_scaler.inverse_transform(samples)
                np.savetxt('samples/uv/uv_samples_{}_noise_{}_n_samples_{}.txt'.format(name, fixed_noise, n_samples), samples, fmt='%.50f')

        np.savetxt('experiment_params/uv/real_mean_and_{}.txt'.format(name), mean, fmt='%.2f')
        np.savetxt('experiment_params/uv/real_error_upper_and{}.txt'.format(name), upper, fmt='%.2f')
        np.savetxt('experiment_params/uv/real_error_lower_and{}.txt'.format(name), lower, fmt='%.2f')
        file = open('experiment_params/uv/trainables_and{}.txt'.format(name), "w")
        file.write('log likelihood of model is :' + str(log_lik))
        file.close()

        # For plotting we transform the counts back to the original domain.

        mean = flux_scaler.inverse_transform(mean)
        upper = flux_scaler.inverse_transform(upper)
        lower = flux_scaler.inverse_transform(lower)

        # Plot the results

        if name == 'Matern_12_Kernel':
            uncertainty_color = "#00b764"
        else:
            uncertainty_color = '#0d9a00'

        if plot_mean:

            if name == 'Matern_12_Kernel':
                mean_color = "#00b764"
            else:
                mean_color = "#0d9a00"

            fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
            ax.scatter(time, uv_band_flux_orig, marker='+', s=10, color='k')
            plt.xlabel('Time (days)', fontsize=16, fontname='Times New Roman')
            plt.ylabel('UVW2 Band Flux', fontsize=16, fontname='Times New Roman')
            plt.ylim(1e-14, 1.75e-13)
            plt.xlim(54150, 58700)
            plt.xticks([55000, 56000, 57000, 58000], fontsize=12)
            plt.yticks([5e-14, 1e-13, 1.5e-13], fontsize=12)
            line, = plt.plot(time_test, mean, lw=1, color=mean_color, alpha=0.75)
            _ = plt.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            # Create an inset

            axins = zoomed_inset_axes(ax, 2.5, loc=2)  # zoom-factor: 2.5, location: top-left
            axins.scatter(time, uv_band_flux_orig, marker='+', s=10, color='k')
            inset_line, = axins.plot(time_test, mean, lw=1, color=mean_color, alpha=0.75)
            _ = axins.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            x1, x2, y1, y2 = 56465, 56750, 0.45e-13, 0.75e-13  # specify the limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")  # mark inset
            plt.yticks(visible=False)
            plt.xticks(visible=False)
            axins.set_xticks([])
            axins.set_yticks([])
            #ax.invert_yaxis()  # Only for magnitudes

            if fix_noise:
                plt.savefig('experiment_figures/uv/{}_and_{}_log_lik_and_{}_noise_color_{}_mean.png'.format(name, log_lik, fixed_noise, mean_color))
            else:
                plt.savefig('experiment_figures/uv/{}_and_{}_log_lik_{}_color_mean.png'.format(name, log_lik, mean_color))
            plt.close()

        else:
            if name == 'Matern_12_Kernel':
                sample_color = "#00b764"
            else:
                sample_color = "#0d9a00"

            # Generate a sample

            sample = m.predict_f_samples(time_test)
            sample = flux_scaler.inverse_transform(sample)

            fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
            ax.scatter(time, uv_band_flux_orig, marker='+', s=10, color='k')
            plt.xlabel('Time (days)', fontsize=16, fontname='Times New Roman')
            plt.ylabel('UVW2 Band Flux', fontsize=16, fontname='Times New Roman')
            plt.ylim(1e-14, 1.75e-13)
            plt.xlim(54150, 58700)
            plt.xticks([55000, 56000, 57000, 58000], fontsize=12)
            plt.yticks([5e-14, 1e-13, 1.5e-13], fontsize=12)
            line, = plt.plot(time_test, sample, lw=1, color=sample_color, alpha=0.75)
            _ = plt.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            # Create an inset

            axins = zoomed_inset_axes(ax, 2.5, loc=2)  # zoom-factor: 3.2, location: top-left
            axins.scatter(time, uv_band_flux_orig, marker='+', s=10, color='k')
            inset_line, = axins.plot(time_test, sample, lw=1, color=sample_color, alpha=0.75)
            _ = axins.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            x1, x2, y1, y2 = 56475, 56725, 0.45e-13, 0.75e-13 # specify the limits  # 12.95, 13.48 are magnitude limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")  # mark inset
            plt.yticks(visible=False)
            plt.xticks(visible=False)
            axins.set_xticks([])
            axins.set_yticks([])
            #ax.invert_yaxis()  # Only for magnitude units

            if fix_noise:
                plt.savefig('experiment_figures/uv/{}_and_{}_log_lik_and_{}_noise_color_{}_sample.png'.format(name,
                                                                                                              log_lik,
                                                                                                              fixed_noise,
                                                                                                              sample_color))
            else:
                plt.savefig('experiment_figures/uv/{}_and_{}_log_lik_{}_color_sample.png'.format(name, log_lik, sample_color))

            plt.close()

        print('{} ML: {}'.format(k, m.log_marginal_likelihood()))
