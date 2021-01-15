# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script contains the utilities for plotting kernel functions.
"""

from matplotlib import pyplot as plt
import numpy as np


def rational_quadratic(alpha, lengthscale, kernel_variance, r):
    """
    The rational quadratic kernel as featured in equation 4.19 on pg. 86 of Rasmussen and Williams. The rational quadratic
    kernel can be seen as a scale mixture (an infinite sum) of squared exponential kernels with different characteristic lengthscales.

    :param alpha: as alpha goes to infinity the RQ kernel becomes the SQE kernel.
    :param lengthscale: the lengthscale
    :param kernel_variance: the kernel variance
    :param r: The absolute distance in input space
    :return: The kernel function evaluated at a list of values r.
    """

    fract = (r/lengthscale)**2 * 1/(2*alpha)
    k_rq = (1 + fract)**(-alpha)
    k_rq *= kernel_variance

    return k_rq


def ornstein_uhlenbeck(lengthscale, kernel_variance, r):
    """
    The Ornstein-Uhlenbeck kernel (special case of exponential kernel in 1 dimension) defined on pg. 85 of Rasmussen
    and Williams.

    :param lengthscale: The lengthscale
    :param kernel_variance: The kernel variance
    :param r: The absolute distance in input space
    :return: The kernel function evaluated at a list of values r.
    """

    k_ou = np.exp(-r/lengthscale)
    k_ou *= kernel_variance

    return k_ou


def squared_exponential(lengthscale, kernel_variance, r):
    """
    The Squared exponential (RBF) kernel.

    :param lengthscale: The lengthscale
    :param kernel_variance: The kernel variance
    :param r:
    :return: The kernel function evaluated at a list of values r.
    """

    scaled_squared_dist = (r/lengthscale)**2
    k_sqe = np.exp(-0.5*scaled_squared_dist)
    k_sqe *= kernel_variance

    return k_sqe


def matern12(lengthscale, kernel_variance, r):
    """
    The Matern -1/2 kernel.

    :param lengthscale: The lengthscale
    :param kernel_variance:  The kernel variance
    :param r: The absolute distance in input space
    :return: the kernel function evaluated at a list of values r.
    """

    scaled_distance = (r/lengthscale)
    k = kernel_variance*np.exp(-scaled_distance)

    return k


kernel = 'RQ'  # One of ['Matern', 'RQ']


if __name__ == '__main__':

    r_vals = np.arange(0, 10000, 10)

    if kernel == 'Matern':
        #autocorr_vals = matern12(0.010366143599172154, 0.9868114735198913, r_vals)
        autocorr_vals = matern12(13.1488, 0.428, r_vals) # in days
        #autocorr_vals = matern12(1136056, 0.428, r_vals)  # in seconds
    else:
        # autocorr_vals = rational_quadratic(0.00321, 10e-5, 10.115, r_vals)
        autocorr_vals = rational_quadratic(0.00321, 0.000954, 10.115, r_vals)

    plt.loglog(r_vals, autocorr_vals, label=f'{kernel}')
    #plt.yticks(np.arange(0, 1, 5))
    plt.title(f'{kernel} Covariance for Mrk-335 X-ray Dataset')
    #plt.title('Rational Quadratic Covariance for Mrk-335 X-ray Dataset')
    plt.xlabel('Days')
    plt.ylabel('Autocorrelation')
    plt.tick_params(axis='both', which='minor', labelsize=7)
    plt.yticks([])
    plt.legend()
    plt.savefig(f'figures/{kernel}.png')
    plt.close()
