# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script fits a Gaussian Process to uv simulations in units of flux as of 15th February 2021.
"""

import logging
import time as real_time  # avoid aliasing with time variable in code

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import set_trainable
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from simulation_utils import load_sim_data, rss_func

logging.getLogger('tensorflow').disabled = True
gpflow.config.set_default_float(np.float64)
fix_noise = True
generate_samples = True  # Whether to generate samples to be used in structure function computation.
start_sim_number = 0  # Simulation number to start-up - workaround for computation time growth per iteration in large loop
f_plot = True  # Whether to plot the simulated lightcurves.

TIMINGS_FILE = '../processed_data/uv_simulations/uv_sim_times.pickle'
GAPPED_FILE = f'sim_curves/w2_lightcurves.dat'
GROUND_TRUTH_FILE = f'sim_curves/w2_lightcurves_no_gaps.dat'


if __name__ == '__main__':

    tf.random.set_seed(42)

    # Units of flux as of 15th February 2021

    train_times, test_times, gapped_flux, ground_truth_flux_matrix = load_sim_data(TIMINGS_FILE,
                                                                                   GAPPED_FILE,
                                                                                   GROUND_TRUTH_FILE)
    n_sims = gapped_flux.shape[0]

    # We do kernel selection by comparison of the negative log marginal likelihood.

    score_dict = {'RBF Kernel': 0, 'Matern_12 Kernel': 0, 'Matern_32 Kernel': 0, 'Matern_52_Kernel': 0,
                  'Rational Quadratic Kernel': 0}

    rss_dict = {'RBF Kernel': 0, 'Matern_12 Kernel': 0, 'Matern_32 Kernel': 0, 'Matern_52_Kernel': 0,
                'Rational Quadratic Kernel': 0}

    nlpd_dict = {'RBF Kernel': 0, 'Matern_12 Kernel': 0, 'Matern_32 Kernel': 0, 'Matern_52_Kernel': 0,
                'Rational Quadratic Kernel': 0}

    for i in range(start_sim_number, n_sims):
        tf.random.set_seed(i)
        start_time = real_time.time()
        print(i)
        k1 = gpflow.kernels.RBF()
        k2 = gpflow.kernels.Matern12()
        k3 = gpflow.kernels.Matern32()
        k4 = gpflow.kernels.Matern52()
        k5 = gpflow.kernels.RationalQuadratic()
        kernel_list = [k1, k2, k3, k4, k5]

        kernel_dict = {kernel_list[0]: 'RBF Kernel', kernel_list[1]: 'Matern_12 Kernel',
                       kernel_list[2]: 'Matern_32 Kernel', kernel_list[3]: 'Matern_52_Kernel',
                       kernel_list[4]: 'Rational Quadratic Kernel'}

        best_log_lik = -1000000  # set to arbitrary large negative value
        best_kernel = ''
        best_rss = 1000000000000000  # set to arbitrary large value
        best_rss_kernel = ''
        best_nlpd = 100000000000000  # set to arbitrary large value
        best_nlpd_kernel = ''

        gapped_rates = np.reshape(gapped_flux[i, :], (-1, 1))
        ground_truth_rates = ground_truth_flux_matrix[i, :]

        # Standardize the count rates

        flux_scaler = StandardScaler()
        gapped_rates = flux_scaler.fit_transform(gapped_rates)

        for k in kernel_list:

            name = kernel_dict[k]

            m = gpflow.models.GPR(data=(train_times, gapped_rates),
                                  mean_function=Constant(np.mean(gapped_rates)),
                                  kernel=k,
                                  noise_variance=np.float64(0.001))

            if fix_noise:
                fixed_noise = np.float64(0.001)  # was 0.05 previously
                set_trainable(m.likelihood.variance, False)  # We don't want to optimise the noise level in this case.
                m.likelihood.variance.assign(fixed_noise)

            opt = gpflow.optimizers.Scipy()

            # If Cholesky decomposition error, then skip

            try:
                opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
            except Exception:
                continue

            # Measure negative log predictive density (NLPD) in standardised space

            y_labels = flux_scaler.transform(tf.reshape(ground_truth_rates, (-1, 1)))
            nlpd = -m.predict_log_density((tf.reshape(tf.cast(test_times, dtype=tf.float64), (-1, 1)),
                                            tf.reshape(tf.cast(y_labels, dtype=tf.float64), (-1, 1))))

            avg_test_nlpd = (nlpd/len(y_labels)).numpy()[0]

            # Measure residual sum of squares (RSS) in standardised space and take average squared residual

            mean, _ = m.predict_y(test_times)
            mean = flux_scaler.inverse_transform(mean)
            num_points = len(mean)  # number of points where GP prediction and ground truth are compared.

            rss = rss_func(np.squeeze(mean), ground_truth_rates)/num_points

            # Measure the log marginal likelihood (NLML) in standardised space

            log_lik = m.log_marginal_likelihood()  # metric is intrinsic to the fit. Note this measure LML and not NLML

            if log_lik > best_log_lik:
                best_kernel = name
                best_log_lik = log_lik

            if rss < best_rss:
                best_rss_kernel = name
                best_rss = rss

            if avg_test_nlpd < best_nlpd:
                best_nlpd_kernel = name
                best_nlpd = avg_test_nlpd

            np.savetxt('uv_sims_stand/mean/mean_{}_iteration_{}.txt'.format(name, i), mean, fmt='%.50f')
            np.savetxt('uv_sims_stand/log_lik/log_lik_{}_iteration_{}.txt'.format(name, i),
                       np.array(log_lik).reshape(-1, 1), fmt='%.50f')
            np.savetxt('uv_sims_stand/rss/rss_{}_iteration_{}.txt'.format(name, i), np.array(rss).reshape(-1, 1),
                       fmt='%.50f')
            np.savetxt('uv_sims_stand/nlpd/nlpd_{}_iteration_{}.txt'.format(name, i), np.array(avg_test_nlpd).reshape(-1, 1),
                       fmt='%.50f')

            if f_plot:

                # Plot the gapped data points observed by GP

                plt.scatter(train_times, flux_scaler.inverse_transform(gapped_rates), marker='+', s=10, color='k', label='Observations')
                plt.xlabel('Time (days)')
                plt.ylabel('UVW2 Band Flux')
                plt.legend(loc=3)
                plt.tight_layout()
                plt.savefig('residuals_figures/uv/data_{}_iteration_{}.png'.format(name, i))
                plt.close()

                # Plot the ground truth light curve

                plt.plot(test_times, ground_truth_rates, lw=1, alpha=0.2, label='Ground Truth Light Curve')
                plt.xlabel('Time (days)')
                plt.ylabel('UVW2 Band Flux')
                plt.legend(loc=3)
                plt.tight_layout()
                plt.savefig('residuals_figures/uv/ground_truth_{}_iteration_{}.png'.format(name, i))
                plt.close()

                line, = plt.plot(test_times, mean, lw=2, label='GP Fit')
                plt.xlabel('Time (days)')
                plt.ylabel('UVW2 Band Flux')
                plt.legend(loc=3)
                plt.tight_layout()
                plt.savefig('residuals_figures/uv/gp_fit_{}_iteration_{}.png'.format(name, i))
                plt.close()

                fig, ax = plt.subplots(1)
                plt.scatter(test_times, mean, marker='+', s=3, color="#00b764", label='GP Predictive Mean')
                plt.scatter(test_times, ground_truth_rates, marker='o', s=3, color='k', label='Ground Truth Light Curve')
                # Residual plot
                difference = np.squeeze(mean) - ground_truth_rates
                #plt.yticks([13.3, 13.7])  # old plotting code for magnitude units
                plt.yticks([60, 70, 80])
                plt.xticks([55500, 55525, 55550])
                plt.xlim(55500, 55550)
                #plt.ylim(13.2, 13.8)  # old plotting code for magnitude units
                plt.ylim(60, 80)
                ax.vlines(test_times, mean, ground_truth_rates, color='k', linewidth=0.2)
                plt.xlabel('Time (days)', fontsize=16, fontname='Times New Roman')
                plt.ylabel('UVW2 Band Flux', fontsize=16, fontname='Times New Roman')
                plt.legend()
                plt.savefig('residuals_figures/uv/residual_plot_{}_iteration_{}.png'.format(name, i))
                plt.close()

            if generate_samples:

                # Sample from posterior of best-fit kernels.

                if name == 'Matern_12 Kernel' or name == 'Rational Quadratic Kernel':
                    samples = np.squeeze(m.predict_f_samples(test_times, 1))
                    samples = flux_scaler.inverse_transform(samples)
                    np.savetxt('SF_samples/uv/SF_uv_samples_{}_iteration_{}.txt'.format(name, i), samples, fmt='%.25f')

        end_time = real_time.time()
        print(f'iteration time is {end_time - start_time}')

        print('best kernel is: ' + best_kernel)
        score_dict[best_kernel] += 1
        print('best rss kernel is: ' + best_rss_kernel)
        rss_dict[best_rss_kernel] += 1
        print('best rss is: ' + str(best_rss))
        print('best nlpd kernel is ' + best_nlpd_kernel)
        nlpd_dict[best_nlpd_kernel] += 1
        print('best nlpd is: ' + str(best_nlpd))

        print(str(score_dict))
        print(str(rss_dict))
        print(str(nlpd_dict))

        file = open('uv_sims_stand/log_lik_scores/log_lik_scores.txt', "w")
        file.write(str(score_dict))
        file.close()

        file = open('uv_sims_stand/rss_scores/rss_scores.txt', "w")
        file.write(str(rss_dict))
        file.close()

        file = open('uv_sims_stand/nlpd_scores/nlpd_scores.txt', "w")
        file.write(str(nlpd_dict))
        file.close()
