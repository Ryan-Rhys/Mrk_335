# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script fits a homoscedastic GP to the Mrk 335 xray data.
"""

import pickle

import gpflow
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

fix_noise = True
log_count_rate = True

if log_count_rate:
    folder = 'log_xray'
else:
    folder = 'xray'

if __name__ == '__main__':

    np.random.seed(42)  # 27th December added to produce deterministic samples

    with open('../processed_data/xray/x_ray_times.pickle', 'rb') as handle:
        time = pickle.load(handle).reshape(-1, 1)
    with open('../processed_data/xray/x_ray_band_count_rates.pickle', 'rb') as handle:
        x_ray_band_count_rates = pickle.load(handle).reshape(-1, 1)

    if log_count_rate:
        x_ray_band_count_rates = np.log(x_ray_band_count_rates)

    band_scaler = StandardScaler()
    x_ray_band_count_rates = band_scaler.fit_transform(x_ray_band_count_rates, x_ray_band_count_rates)  # second argument is a dummy argument as StandardScaler expects train inputs and targets
    x_ray_band_count_rates_plot = band_scaler.inverse_transform(x_ray_band_count_rates)  # plot in the original domain (be that log or otherwise)

    time_test = np.arange(54236, 58630, 1).reshape(-1, 1)
    time_scaler = StandardScaler()
    time_test = time_scaler.fit_transform(time_test, time_test)
    time = time_scaler.transform(time)

    time_plot = time_scaler.inverse_transform(time)  # for plotting int the original domain
    time_test_plot = time_scaler.inverse_transform(time_test)

    # We do kernel selection by comparison of the negative log marginal likelihood.

    k1 = gpflow.kernels.RBF(input_dim=1)
    k2 = gpflow.kernels.Matern12(input_dim=1)
    k3 = gpflow.kernels.Matern32(input_dim=1)
    k4 = gpflow.kernels.Matern52(input_dim=1)
    k5 = gpflow.kernels.RationalQuadratic(input_dim=1)
    kernel_list = [k1, k2, k3, k4, k5]

    kernel_dict = {kernel_list[0]: 'RBF_Kernel', kernel_list[1]: 'Matern_12_Kernel', kernel_list[2]: 'Matern_32_Kernel',
                   kernel_list[3]: 'Matern_52_Kernel', kernel_list[4]: 'Rational_Quadratic_Kernel'}

    for k in kernel_list:
        name = kernel_dict[k]

        m = gpflow.models.GPR(time, x_ray_band_count_rates, kern=k)

        if fix_noise:

            fixed_noise = 0.01  # was 0.05 previously. Treating this as jitter now in log domain.
            m.likelihood.variance = fixed_noise
            m.likelihood.variance.trainable = False

        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m, disp=False)
        mean, var = m.predict_y(time_test)

        log_lik = m.compute_log_likelihood()

        lower = mean[:, 0] - 2 * np.sqrt(var[:, 0])  # 1 standard deviation is common in astrophysics
        upper = mean[:, 0] + 2 * np.sqrt(var[:, 0])

        # Sample from posterior of best-fit kernels

        if name == 'Matern_12_Kernel' or name == 'Rational_Quadratic_Kernel':

            samples = m.predict_f_samples(time_test, 10000).squeeze()
            np.savetxt('samples/xray/x_ray_samples_{}_noise_{}.txt'.format(name, fixed_noise), samples, fmt='%.2f')

        if fix_noise:
            np.savetxt('experiment_params/' + folder + '/real_mean_and_{}_and_{}_fixed_noise.txt'.format(name, fixed_noise), mean, fmt='%.2f')
            np.savetxt('experiment_params/' + folder + '/real_error_upper_and{}_and_{}_fixed_noise.txt'.format(name, fixed_noise), upper, fmt='%.2f')
            np.savetxt('experiment_params/' + folder + '/real_error_lower_and{}_and_{}_fixed_noise.txt'.format(name, fixed_noise), lower, fmt='%.2f')
            file = open('experiment_params/' + folder + '/trainables_and{}_and_{}_fixed_noise.txt'.format(name, fixed_noise), "w")
            file.write(str(m.read_trainables()))
            file.write('log likelihood of model is :' + str(log_lik))
            file.close()

        else:
            np.savetxt('experiment_params/' + folder + '/real_mean_and_{}.txt'.format(name), mean, fmt='%.2f')
            np.savetxt('experiment_params/' + folder + '/real_error_upper_and{}.txt'.format(name), upper, fmt='%.2f')
            np.savetxt('experiment_params/' + folder + '/real_error_lower_and{}.txt'.format(name), lower, fmt='%.2f')
            file = open('experiment_params/' + folder + '/trainables_and{}.txt'.format(name), "w")
            file.write(str(m.read_trainables()))
            file.write('log likelihood of model is :' + str(log_lik))
            file.close()

        mean = band_scaler.inverse_transform(mean)
        upper = band_scaler.inverse_transform(upper)
        lower = band_scaler.inverse_transform(lower)

        plt.plot(time_plot, x_ray_band_count_rates_plot, '+', markersize=7, mew=0.2)  # time plot are original times
        plt.xlabel('Time (days)')
        if log_count_rate:
            plt.ylabel('Log X-ray Band Count Rate')
        else:
            plt.ylabel('Xray Band Count Rate')
        plt.title('X-ray Lightcurve Mrk 335 {}'.format(name))
        line, = plt.plot(time_test_plot, mean, lw=2)
        #_ = plt.fill_between(time_test_plot[:, 0], lower, upper, color=line.get_color(), alpha=0.2)

        if fix_noise:
            plt.savefig('experiment_figures/' + folder + '/{}_and_{}_log_lik_and_{}_noise_mean_only.png'.format(name, log_lik, fixed_noise))
        else:
            plt.savefig('experiment_figures/' + folder + '/{}_and_{}_log_lik.png'.format(name, log_lik))

        plt.close()

        print('{} ML: {}'.format(k, m.compute_log_likelihood()))
        print(m.as_pandas_table())
