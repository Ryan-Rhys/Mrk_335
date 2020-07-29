# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script fits a Gaussian Process to x-ray simulations.
"""

import logging

logging.getLogger('tensorflow').disabled = True

import gpflow
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from simulation_utils import load_sim_data, rss_func

fix_noise = True

TIMINGS_FILE = 'processed_data/xray_simulations/x_ray_sim_times.pickle'
GAPPED_FILE = 'sim_curves/xray_lightcurves.dat'
GROUND_TRUTH_FILE = 'sim_curves/xray_lightcurves_no_gaps.dat'

if __name__ == '__main__':

    time, test_times, gapped_count_rates, ground_truth_count_rates = load_sim_data(TIMINGS_FILE,
                                                                                   GAPPED_FILE,
                                                                                   GROUND_TRUTH_FILE)
    n_sims = gapped_count_rates.shape[0]

    # Standardize the timings

    time_scaler = StandardScaler()
    train_times = time_scaler.fit_transform(time)
    test_times = time_scaler.transform(test_times)

    # Add jitter ot the count rates to avoid numerical issues with log transform of zero values.

    jitter = 1e-10
    ground_truth_count_rates += jitter

    # Log transform the count rates

    log_gapped_count_rates = np.log(gapped_count_rates)
    log_ground_truth_count_rates = np.log(ground_truth_count_rates)

    # Standardize the log count rates

    count_rate_scaler = StandardScaler()
    gapped_rates_matrix = count_rate_scaler.fit_transform(log_gapped_count_rates)
    ground_truth_rates_matrix = count_rate_scaler.fit_transform(log_ground_truth_count_rates)

    # We do kernel selection by comparison of the negative log marginal likelihood.
    # Problems with OU kernel in GPflow 1.5.1

    score_dict = {'RBF Kernel': 0, 'Matern_12 Kernel': 0, 'Matern_32 Kernel': 0, 'Matern_52_Kernel': 0,
                  'Rational Quadratic Kernel': 0}

    rss_dict = {'RBF Kernel': 0, 'Matern_12 Kernel': 0, 'Matern_32 Kernel': 0, 'Matern_52_Kernel': 0,
                'Rational Quadratic Kernel': 0}

    for i in range(0, n_sims):
        print(i)
        with tf.Session(graph=tf.Graph()):
            k1 = gpflow.kernels.RBF(input_dim=1)
            k2 = gpflow.kernels.Matern12(input_dim=1)
            k3 = gpflow.kernels.Matern32(input_dim=1)
            k4 = gpflow.kernels.Matern52(input_dim=1)
            k5 = gpflow.kernels.RationalQuadratic(input_dim=1)
            kernel_list = [k1, k2, k3, k4, k5]

            kernel_dict = {kernel_list[0]: 'RBF Kernel', kernel_list[1]: 'Matern_12 Kernel',
                           kernel_list[2]: 'Matern_32 Kernel', kernel_list[3]: 'Matern_52_Kernel',
                           kernel_list[4]: 'Rational Quadratic Kernel'}

            best_log_lik = -1000000  # set to arbitrary large negative value
            best_kernel = ''
            best_rss = 1000000000000000  # set to arbitrary large value
            best_rss_kernel = ''

            gapped_rates = gapped_rates_matrix[i, :].reshape(-1, 1)
            ground_truth_rates = ground_truth_rates_matrix[i, :]

            for k in kernel_list:

                name = kernel_dict[k]
                m = gpflow.models.GPR(train_times, gapped_rates, kern=k)

                if fix_noise:
                    fixed_noise = 0.001  # was 0.05 previously
                    m.likelihood.variance = fixed_noise
                    m.likelihood.variance.trainable = False

                opt = gpflow.train.ScipyOptimizer()
                opt.minimize(m, disp=False)

                mean, var = m.predict_y(test_times)
                rss = rss_func(mean.reshape(len(test_times), ), ground_truth_rates.reshape(len(test_times), ))

                log_lik = m.compute_log_likelihood()

                lower = mean[:, 0] - 2 * np.sqrt(var[:, 0])  # 1 standard deviation is common in astrophysics
                upper = mean[:, 0] + 2 * np.sqrt(var[:, 0])

                if log_lik > best_log_lik:
                    best_kernel = name
                    best_log_lik = log_lik

                if rss < best_rss:
                    best_rss_kernel = name
                    best_rss = rss

                np.savetxt('xray_sims_stand/mean/mean_{}_iteration_{}.txt'.format(name, i), mean, fmt='%.2f')
                np.savetxt('xray_sims_stand/var/var_{}_iteration_{}.txt'.format(name, i), var, fmt='%.2f')
                np.savetxt('xray_sims_stand/log_lik/log_lik_{}_iteration_{}.txt'.format(name, i),
                           np.array(log_lik).reshape(-1, 1), fmt='%.2f')
                np.savetxt('xray_sims_stand/rss/rss_{}_iteration_{}.txt'.format(name, i), np.array(rss).reshape(-1, 1),
                           fmt='%.2f')
                file = open('xray_sims_stand/trainables/trainables_{}_and{}.txt'.format(name, i), "w")
                file.write(str(m.read_trainables()))
                file.close()

                plt.plot(train_times, gapped_rates, '+', markersize=7, mew=0.2, label='observations')
                plt.plot(test_times, ground_truth_rates, lw=1, alpha=0.2, label='ground truth light curve')
                plt.xlabel('Standardised Time')
                plt.ylabel('Standardised Log Xray Band Count Rate')
                plt.title('X-ray Lightcurve Mrk 335 {}'.format(name))
                line, = plt.plot(test_times, mean, lw=2, label='GP fit')
                _ = plt.fill_between(test_times[:, 0], lower, upper, color=line.get_color(), alpha=0.2)
                plt.legend(loc=3)
                plt.show()

            print('best kernel is: ' + best_kernel)
            score_dict[best_kernel] += 1
            print('best rss kernel is: ' + best_rss_kernel)
            rss_dict[best_rss_kernel] += 1
            print('best rss is: ' + str(best_rss))

            print(str(score_dict))
            print(str(rss_dict))

        file = open('xray_sims_stand/log_lik_scores/log_lik_scores.txt', "w")
        file.write(str(score_dict))
        file.close()

        file = open('xray_sims_stand/rss_scores/rss_scores.txt', "w")
        file.write(str(rss_dict))
        file.close()
