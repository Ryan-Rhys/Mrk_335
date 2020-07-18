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
folder = 'uv'  # Folder in which to save the results
generate_samples = True  # Whether to generate samples from the best-fit kernels.
plot_mean = False  # Whether to plot the GP mean or the samples

m = None


def objective_closure():
    """
    Objective function for GP-optimization
    """
    return -m.log_marginal_likelihood()


if __name__ == '__main__':

    np.random.seed(42)  # Set the same random seed for training

    # Load the UV band data.

    with open('../processed_data/uv/uv_times.pickle', 'rb') as handle:
        time = pickle.load(handle).reshape(-1, 1)
    with open('../processed_data/uv/uv_band_count_rates.pickle', 'rb') as handle:
        uv_band_count_rates = pickle.load(handle).reshape(-1, 1)
    with open('../processed_data/uv/uv_band_count_errors.pickle', 'rb') as handle:
        uv_band_count_errors = pickle.load(handle).reshape(-1, 1)

    # We standardise the outputs

    count_scaler = StandardScaler()
    counts = count_scaler.fit_transform(uv_band_count_rates)

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

        m = gpflow.models.GPR(data=(time, counts),
                              mean_function=Constant(np.mean(counts)),
                              kernel=k,
                              noise_variance=1)

        if fix_noise:

            # Fix a noise level to be the average experimental error observed in the dataset (0.037)

            fixed_noise = np.mean(uv_band_count_errors)
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

                samples = tf.squeeze(m.predict_f_samples(time_test, 10000))
                samples = count_scaler.inverse_transform(samples)
                np.savetxt('samples/uv/uv_samples_{}_noise_{}.txt'.format(name, fixed_noise), samples, fmt='%.2f')

        np.savetxt('experiment_params/' + folder + '/real_mean_and_{}.txt'.format(name), mean, fmt='%.2f')
        np.savetxt('experiment_params/' + folder + '/real_error_upper_and{}.txt'.format(name), upper, fmt='%.2f')
        np.savetxt('experiment_params/' + folder + '/real_error_lower_and{}.txt'.format(name), lower, fmt='%.2f')
        file = open('experiment_params/' + folder + '/trainables_and{}.txt'.format(name), "w")
        file.write('log likelihood of model is :' + str(log_lik))
        file.close()

        # For plotting we transform the counts back to the original domain.

        mean = count_scaler.inverse_transform(mean)
        upper = count_scaler.inverse_transform(upper)
        lower = count_scaler.inverse_transform(lower)

        # Plot the results

        uncertainty_color = "#00b764"

        if plot_mean:

            mean_color = "#00b764"

            fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
            ax.scatter(time, uv_band_count_rates, marker='+', s=10, color='k')
            plt.xlabel('Time (days)')
            plt.ylabel('UV Band Count Rate')
            plt.title('UV Lightcurve Mrk 335 {}'.format(name))
            plt.ylim(12.4, 14.75)
            plt.xlim(54150, 58700)
            plt.xticks([55000, 56000, 57000, 58000])
            plt.yticks([13, 13.5, 14, 14.5])
            line, = plt.plot(time_test, mean, lw=1, color=mean_color, alpha=0.75)
            _ = plt.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            # Create an inset

            axins = zoomed_inset_axes(ax, 2.5, loc=2)  # zoom-factor: 2.5, location: top-left
            axins.scatter(time, uv_band_count_rates, marker='+', s=10, color='k')
            inset_line, = axins.plot(time_test, mean, lw=1, color=mean_color, alpha=0.75)
            _ = axins.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            x1, x2, y1, y2 = 56475, 56725, 12.95, 13.48  # specify the limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")  # mark inset
            plt.yticks(visible=False)
            plt.xticks(visible=False)
            axins.set_xticks([])
            axins.set_yticks([])

            if fix_noise:
                plt.savefig('experiment_figures/' + folder + '/{}_and_{}_log_lik_and_{}_noise_color_{}_mean.png'.format(name, log_lik, fixed_noise, mean_color))
            else:
                plt.savefig('experiment_figures/' + folder + '/{}_and_{}_log_lik_{}_color_mean.png'.format(name, log_lik, mean_color))

            plt.close()

        else:

            sample_color = "#ff3203"

            # Generate a sample

            sample = m.predict_f_samples(time_test)
            sample = count_scaler.inverse_transform(sample)

            fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
            ax.scatter(time, uv_band_count_rates, marker='+', s=10, color='k')
            plt.xlabel('Time (days)')
            plt.ylabel('UV Band Count Rate')
            plt.title('UV Lightcurve Mrk 335 {}'.format(name))
            plt.ylim(12.4, 15.5)
            plt.xlim(54150, 58700)
            plt.xticks([55000, 56000, 57000, 58000])
            plt.yticks([13, 14, 15])
            line, = plt.plot(time_test, sample, lw=1, color=sample_color, alpha=0.75)
            _ = plt.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            # Create an inset

            axins = zoomed_inset_axes(ax, 3.2, loc=2)  # zoom-factor: 3.2, location: top-left
            axins.scatter(time, uv_band_count_rates, marker='+', s=10, color='k')
            inset_line, = axins.plot(time_test, sample, lw=1, color=sample_color, alpha=0.75)
            _ = axins.fill_between(time_test[:, 0], lower, upper, color=uncertainty_color, alpha=0.2)

            x1, x2, y1, y2 = 56475, 56725, 12.95, 13.48  # specify the limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")  # mark inset
            plt.yticks(visible=False)
            plt.xticks(visible=False)
            axins.set_xticks([])
            axins.set_yticks([])

            if fix_noise:
                plt.savefig(
                    'experiment_figures/' + folder + '/{}_and_{}_log_lik_and_{}_noise_color_{}_sample.png'.format(name,
                                                                                                                log_lik,
                                                                                                                fixed_noise,
                                                                                                                sample_color))
            else:
                plt.savefig(
                    'experiment_figures/' + folder + '/{}_and_{}_log_lik_{}_color_sample.png'.format(name, log_lik,
                                                                                                   sample_color))

            plt.close()

        print('{} ML: {}'.format(k, m.log_marginal_likelihood()))
