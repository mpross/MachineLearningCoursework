#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import itertools


def fit(fitter, data, weights=None, free_parameters=None, n=0, binning=None):
    """
    Plots the fitted distribution. 

    Parameters
    ----------
    fitter : The fitter object
    data : numpy-array containing the data
    weights : numpy-array containing the weights. If None is passed a weight of one is assumed for each event.
    free_parameters : Use these free parameters instead of the ones fitted by the fitter
    n : Plot the n-th distribution in case of a multi-dimensional fit
    """
    if free_parameters is None:
        if fitter.r is None:
            raise RuntimeError("Please call fit first on the fitter object")
        free_parameters = fitter.r.x

    data, weights = fitter._ensure_dimension(data, weights)
    parameters = fitter.mapping(free_parameters)

    data = data[n]
    weights = weights[n]
    parameters = parameters[n]
    distribution = fitter.distributions[n]

    frozen_distribution = distribution(**parameters)
    if binning is None:
        binning = dict(range=(np.min(data), np.max(data)), bins=min(int(np.sqrt(len(data))) + 1, 100))
    space = np.linspace(*binning['range'], binning['bins']*10)

    content, boundaries = np.histogram(data, **binning, weights=weights)
    plt.errorbar((boundaries[1:] + boundaries[:-1]) / 2, content, yerr=np.sqrt(content), color='black', fmt='s', markersize=8, label='Data', zorder=3)

    weight = np.sum(weights) / binning['bins'] * (binning['range'][1] - binning['range'][0])

    plt.fill_between(space, weight * fitter.normalisation(parameters) * frozen_distribution.pdf(space), label='Fit', color='gray')

    for name, distribution, norm_name, shape_names in zip(frozen_distribution.dist._components,
                                                          frozen_distribution.dist._distributions,
                                                          frozen_distribution.dist._distribution_norms,
                                                          frozen_distribution.dist._distribution_shapes):
        norm = frozen_distribution.kwds[norm_name]
        shapes = {'_'.join(k.split('_')[1:]) : frozen_distribution.kwds[k] for k in shape_names}
        plt.plot(space, weight * norm * distribution.pdf(space, **shapes), label=name)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlim(binning['range'])
    plt.ylim((0, None))


def pull(fitter, toy_experiments, i):
    """
    Plots the pull distribution of the i-th free parameter

    Parameters
    ----------
    fitter : The fitter object
    toy_experiments : Result returned by the toy method if the fitter
    i : Plot the pull of the i-th free parameter
    """
    X = np.linspace(-5, 5, 1000)
    plt.plot(X, scipy.stats.norm.pdf(X), lw=2)
    pulls = np.array([r.x[i] - t[i] for t, r, _ in toy_experiments])
    uncertainty = np.array([[r.x[i] - u[i][0], u[i][1] - r.x[i]] for _, r, u in toy_experiments])
    pulls = np.where(pulls > 0, pulls / uncertainty[:, 0], pulls / uncertainty[:, 1])
    plt.title("Mean = {:.2f}   Standard Deviation = {:.2f}".format(pulls.mean(), pulls.std()))
    plt.hist(pulls, normed=True)


def stability(fitter, toy_experiments, i):
    """
    Plots the stability of the fit with respect to the i-th free parameter

    Parameters
    ----------
    fitter : The fitter object
    toy_experiments : Result returned by the toy method if the fitter
    i : Plot the pull of the i-th free parameter
    """
    values = [[t[i], r.x[i]] for t, r, _ in toy_experiments]
    values = sorted(values, key=lambda x: x[0])
    true_values = []
    fitted_mean = []
    fitted_std = []
    fitted_mean_error = []
    for key, group in itertools.groupby(values, lambda x: x[0]):
        group = np.array(list(group))
        true_values.append(key)
        fitted_mean.append(group[:,1].mean())
        fitted_std.append(group[:,1].std())
        fitted_mean_error.append(group[:,1].std() / np.sqrt(len(group)))

    #plt.errorbar(true_values, fitted_mean, yerr=fitted_std, fmt='', ls='')
    plt.errorbar(true_values, fitted_mean, yerr=fitted_mean_error, fmt='s', markersize=8, ls='')
    plt.xlabel('True Value')
    plt.ylabel('Fitted Value')
    plt.plot([true_values[0], true_values[-1]], [true_values[0], true_values[-1]], ls='--')


def likelihood_profile(fitter, profile_values, likelihood_profile, i):
    """
    Plots the profile log likelihood of the fit with respect to the i-th free parameter

    Parameters
    ----------
    fitter : The fitter object
    profile_values : list of numpy array containing the values used to obtain the likelihood profile
    likelihood_profile : the result of the likelihood_profile call of the fitter object
    i : Plot the pull of the i-th free parameter
    """
    plt.plot([fitter.r.x[i]], [fitter.r.fun], marker='s', markersize=8)
    plt.hlines([fitter.r.fun+(n**2/2.0) for n in range(1, 4)],
	       xmin=np.min(profile_values[i]), xmax=np.max(profile_values[i]), linestyles='--')
    plt.plot(profile_values[i], likelihood_profile)
    plt.ylim(fitter.r.fun, fitter.r.fun+8)
    plt.ylabel("Negative Log Likelihood")

