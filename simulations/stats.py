# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script computes aggregate statistics for the lightcurve simulations. Important to note that the statistics are
computed for standardised values for LML and in real space for average squared residuals
"""

import numpy as np

NSIMS = 1000
XRAY = True  # Toggles whether to compute statistics for x-ray or uv simulations

if __name__ == "__main__":

    if XRAY:
        directory = 'xray_sims_stand'
        tag = 'xray'
    else:
        directory = 'uv_sims_stand'
        tag = 'uv'

    matern12_log_lik_list = []
    matern32_log_lik_list = []
    matern52_log_lik_list = []
    rq_log_lik_list = []
    rbf_log_lik_list = []

    matern12_rss_list = []
    matern32_rss_list = []
    matern52_rss_list = []
    rq_rss_list = []
    rbf_rss_list = []

    for i in range(0, NSIMS):
        with open(directory + '/log_lik/log_lik_Matern_12 Kernel_iteration_{}.txt'.format(i)) as f:
            matern12_log_lik = f.readlines()
            matern12_log_lik_list.append(float(matern12_log_lik[0].strip()))
        with open(directory + '/log_lik/log_lik_Matern_32 Kernel_iteration_{}.txt'.format(i)) as f:
            matern32_log_lik = f.readlines()
            matern32_log_lik_list.append(float(matern32_log_lik[0].strip()))
        with open(directory + '/log_lik/log_lik_Matern_52_Kernel_iteration_{}.txt'.format(i)) as f:
            matern52_log_lik = f.readlines()
            matern52_log_lik_list.append(float(matern52_log_lik[0].strip()))
        with open(directory + '/log_lik/log_lik_Rational Quadratic Kernel_iteration_{}.txt'.format(i)) as f:
            rq_log_lik = f.readlines()
            rq_log_lik_list.append(float(rq_log_lik[0].strip()))
        with open(directory + '/log_lik/log_lik_RBF Kernel_iteration_{}.txt'.format(i)) as f:
            rbf_log_lik = f.readlines()
            rbf_log_lik_list.append(float(rbf_log_lik[0].strip()))
        with open(directory + '/rss/rss_Matern_12 Kernel_iteration_{}.txt'.format(i)) as f:
            matern12_rss = f.readlines()
            matern12_rss_list.append(float(matern12_rss[0].strip()))
        with open(directory + '/rss/rss_Matern_32 Kernel_iteration_{}.txt'.format(i)) as f:
            matern32_rss = f.readlines()
            matern32_rss_list.append(float(matern32_rss[0].strip()))
        with open(directory + '/rss/rss_Matern_52_Kernel_iteration_{}.txt'.format(i)) as f:
            matern52_rss = f.readlines()
            matern52_rss_list.append(float(matern52_rss[0].strip()))
        with open(directory + '/rss/rss_Rational Quadratic Kernel_iteration_{}.txt'.format(i)) as f:
            rq_rss = f.readlines()
            rq_rss_list.append(float(rq_rss[0].strip()))
        with open(directory + '/rss/rss_RBF Kernel_iteration_{}.txt'.format(i)) as f:
            rbf_rss = f.readlines()
            rbf_rss_list.append(float(rbf_rss[0].strip()))

    matern12_av_ll = np.mean(matern12_log_lik_list)
    matern12_av_rss = np.mean(matern12_rss_list)
    matern32_av_ll = np.mean(matern32_log_lik_list)
    matern32_av_rss = np.mean(matern32_rss_list)
    matern52_av_ll = np.mean(matern52_log_lik_list)
    matern52_av_rss = np.mean(matern52_rss_list)
    rq_av_ll = np.mean(rq_log_lik_list)
    rq_av_rss = np.mean(rq_rss_list)
    rbf_av_ll = np.mean(rbf_log_lik_list)
    rbf_av_rss = np.mean(rbf_rss_list)

    matern12_std_ll = np.std(matern12_log_lik_list)
    matern12_std_rss = np.std(matern12_rss_list)
    matern32_std_ll = np.std(matern32_log_lik_list)
    matern32_std_rss = np.std(matern32_rss_list)
    matern52_std_ll = np.std(matern52_log_lik_list)
    matern52_std_rss = np.std(matern52_rss_list)
    rq_std_ll = np.std(rq_log_lik_list)
    rq_std_rss = np.std(rq_rss_list)
    rbf_std_ll = np.std(rbf_log_lik_list)
    rbf_std_rss = np.std(rbf_rss_list)

    with open(f'simulation_stats/{tag}_simulation_stats.txt', 'w') as f:

        f.write(f'Matern 1/2 average LML is {matern12_av_ll}')
        f.write(f'\nMatern 1/2 average LML Error is {matern12_std_ll}')

        f.write(f'\nMatern 1/2 average RSS is {matern12_av_rss}')
        f.write(f'\nMatern 1/2 average RSS Error is {matern12_std_rss}')

        f.write(f'\nMatern 3/2 average LML is {matern32_av_ll}')
        f.write(f'\nMatern 3/2 average LML Error is {matern32_std_ll}')

        f.write(f'\nMatern 3/2 average RSS is {matern32_av_rss}')
        f.write(f'\nMatern 3/2 average RSS Error is {matern32_std_rss}')

        f.write(f'\nMatern 5/2 average LML is {matern52_av_ll}')
        f.write(f'\nMatern 5/2 average LML Error is {matern52_std_ll}')

        f.write(f'\nMatern 5/2 average RSS is {matern52_av_rss}')
        f.write(f'\nMatern 5/2 average RSS Error is {matern52_std_rss}')

        f.write(f'\nRQ average LML is {rq_av_ll}')
        f.write(f'\nRQ average LML Error is {rq_std_ll}')

        f.write(f'\nRQ average RSS is {rq_av_rss}')
        f.write(f'\nRQ average RSS Error is {rq_std_rss}')

        f.write(f'\nRBF average LML is {rbf_av_ll}')
        f.write(f'\nRBF average LML Error is {rbf_std_ll}')

        f.write(f'\nRBF average RSS is {rbf_av_rss}')
        f.write(f'\nRBF average RSS Error is {rbf_std_rss}')
