#!/usr/bin/env python3

import numpy as np
import scipy.optimize
import collections
import functools


class Fitter(object):
    """
    Implements extended unbinned maximum likelihood fits for scipy.stats distributions.
    The distribution has to be implemented completely using scipy.stats distributions.
    A mapping function has to be provided, which maps the free parameters of the fit to the scipy.stats shape parameters.
    An optional normalisation function can be provided, which maps the shape parameters of each distribution to its overall normalisation.
    The fit itself is performed using scipy.optimize.
    """
    def __init__(self, mapping, distributions, normalisation=None, prior=None, method='nelder-mead', ugly_and_fast=False):
        """
        Parameters
        ----------
        mapping : user-defined function, which maps the free parameters to the shape parameters of the distribution.
                  For a one-dimensional fit the function maps np.array(free-parameters) -> dict(shape parameters)
                  For a multi-dimenional fit the function maps np.array(free-parameters) -> list of dict(shape parameters) for each distribution.
        distributions : A scipy.stats distribution (implies a one-dimensional fit) or a list of scipy.stats distributions (implies a multi-dimensional fit).
        normalisation : user-defined function, which maps the shape parameters of a distribution (returned by the mapping) to the overall norm of the distribution.
                        If None is given, the norm 1.0 is assumed, which reduced the fit to an unbinned M.L fit, instead of a extended unbinned M.L fit.
        prior : user-defined function, which maps the shape parameters of all distributions to a prior.
        method : The method passed to scipy.optimize.minimize
        ugly_and_fast : If true, the calculation of the uncertainty and likelihood profile will speed-up, but loose accuracy.
                        This is achieved by just evaluating the loss-function with the optimal-parameters, instead of fitting all parameters again.
        """
        self.is_multi_dimensional = isinstance(distributions, collections.Sequence)
        self.mapping = mapping if self.is_multi_dimensional else lambda p: [mapping(p)]
        self.distributions = distributions if self.is_multi_dimensional else [distributions]
        self.method = method
        self.prior = prior if self.is_multi_dimensional or prior is None else lambda p: prior(p[0])
        self.normalisation = normalisation
        self.ugly_and_fast = ugly_and_fast
        self.r = None

    def loss(self, free_parameters, data, weights, mapping):
        """
        Calculates the extended unbinned maximum likelihood fit.
        It is assumed that the pdf of the distributions is normed (integral over the whole range is one).

        Parameters
        ----------
        free_parameters : sequence containing the free parameters of the fit
        data : list of numpy array containing the data for each distribution
        weights : list of numpy array containing the weights for each distribution
        mapping : the mapping which maps the free parameters to the shape parameters for each distribution
        """
        loss = 0.0 
        parameters = mapping(free_parameters)
        for d, w, p, distribution in zip(data, weights, parameters, self.distributions):
            N = np.sum(w)
            norm = 1.0 if self.normalisation is None else self.normalisation(p)
            average_number_of_events = norm * N
            loss += - N * np.log(average_number_of_events) + average_number_of_events
            loss += - np.sum(w * np.log(distribution.pdf(d, **p)))
        if self.prior is not None:
            loss += - np.log(self.prior(parameters))
        return loss

    def _ensure_dimension(self, data, weights):
        """
        Ensures that the data and weights are list of numpy arrays.
        If the fit is not multi-dimensional the user passed a single numpy-array instead of a list 
        for the data and weights array. We wrap this here in a list.
        If the user did not provide a weights numpy-array, we assume a weight of one for each event.
        """
        if not self.is_multi_dimensional:
            data = [data]
        if not self.is_multi_dimensional:
            if weights is not None:
                weights = [weights]
        if weights is None:
            weights = [np.ones(len(d)) for d in data]
        return data, weights

    def _fit(self, initial_parameters, data, weights, mapping):
        """
        Internal method of the class
        Performs the fit without transforming the data and weights into the correct format, because the other methods
        which call this function have already done this.
        """
        r = scipy.optimize.minimize(self.loss, initial_parameters, args=(data, weights, mapping), method=self.method)
        return r

    def _get_likelihood_profile_function(self, optimal_parameters, fixed):
        """
        Internal method of the class.
        Returns the likelihood profile function based on the "ugly_and_fast" parameter.
        """
        optimal_parameters_without_fixed = [p for i, p in enumerate(optimal_parameters) if i not in fixed]
        insert_fixed_parameters = lambda p, f: functools.reduce(lambda l, i: l[:i[1]] + [f[i[0]]] + l[i[1]:], enumerate(fixed), list(p))
        if self.ugly_and_fast:
            return lambda x, data, weights: self.loss(insert_fixed_parameters(optimal_parameters_without_fixed, x), data, weights, self.mapping)
        else:
            return lambda x, data, weights: self._fit(optimal_parameters_without_fixed, data, weights, lambda p: self.mapping(np.array(insert_fixed_parameters(p, x)))).fun
    
    def fit(self, initial_parameters, data, weights=None):
        """
        Fits the distributions to the passed data using the given initial parmameters as starting point for the optimization.

        Parameters
        ----------
        initial_parameters : sequence containing the initial guess for the free parmameters of the fit
        data : numpy-array or list of numpy-array containing the data
        weights : numpy-array or list of numpy-array containing the weights. If None is passed a weight of one is assumed for each event.
        """
        self.r = self._fit(initial_parameters, *self._ensure_dimension(data, weights), self.mapping)
        return self.r
    
    def get_uncertainties(self, parameter_boundaries, data, weights=None):
        """
        Returns the uncertainty of free parameters calculated from the likelihood profile.
        You have to call fit before calling this function

        Parameters
        ----------
        parameter_boundaries : list of 2-float-tuple containing the lower and upper boundary of the free parameters.
                               If None is passed for a free parameters instead of the tuple its uncertainty is not calculated
        data : numpy-array or list of numpy-array containing the data
        weights : numpy-array or list of numpy-array containing the weights. 
                  If None is passed a weight of one is assumed for each event.
        """
        if self.r is None:
            raise RuntimeError("Please call fit first")
        if len(parameter_boundaries) != len(self.r.x):
            raise RuntimeError("The number of provided boundaries does not match the number of fitted parameters")
        data, weights = self._ensure_dimension(data, weights)
        uncertainties = []
        for i, boundaries in enumerate(parameter_boundaries):
            if boundaries is None:
                uncertainties.append([None, None])
            else:
                lower_boundary, upper_boundary = boundaries
                likelihood_profile_function = self._get_likelihood_profile_function(list(self.r.x), [i])
                lower = lower_boundary
                upper = upper_boundary
                if lower_boundary is not None:
                    try:
                        lower = scipy.optimize.brentq(lambda x: likelihood_profile_function([x], data, weights) - (self.r.fun + 0.5), lower_boundary, self.r.x[i])
                    except ValueError:
                        print("Could not find a valid lower boundary. Setting lower boundary to the given boundary.")
                        pass
                if upper_boundary is not None:
                    try:
                        upper = scipy.optimize.brentq(lambda x: likelihood_profile_function([x], data, weights) - (self.r.fun + 0.5), self.r.x[i], upper_boundary)
                    except ValueError:
                        print("Could not find a valid upper boundary. Setting upper boundary to the given boundary.")
                        pass
                uncertainties.append([lower, upper])
        return uncertainties
    
    def likelihood_profile(self, parameter_values, data, weights=None):
        """
        Returns the likelihood profile of free parameters.
        You have to call fit before calling this function

        Parameters
        ----------
        parameter_values : list of numpy-array containing fixed values for the free parameters.
                           If None is passed instead of a numpy-array the corresponding parameter won't be fixed during the profile fit.
        data : numpy-array or list of numpy-array containing the data
        weights : numpy-array or list of numpy-array containing the weights. If None is passed a weight of one is assumed for each event.
        """
        if self.r is None:
            raise RuntimeError("Please call fit first")
        if len(parameter_values) != len(self.r.x):
            raise RuntimeError("The number of provided values does not match the number of fitted parameters")
        data, weights = self._ensure_dimension(data, weights)
        parameter_positions = [i for i, v in enumerate(parameter_values) if v is not None]
        likelihood_profile_function = self._get_likelihood_profile_function(list(self.r.x), parameter_positions)
        return np.array([likelihood_profile_function(list(parameters), data, weights) for parameters in zip(*[v for v in parameter_values if v is not None])])

    def toy(self, initial_parameters, true_parameters, sample_sizes, parameter_boundaries=None):
        """
        Performs toy fits by drawning fake data from the distributions given its true values.
        You can use this to perform
         - a stability test of the fit (by passing different values for the true parameters)
         - get the pull distribution of the fit (by passing the same value for the true parameters several times)

        Parameters
        ----------
        initial_parameters : sequence containing the initial guess for the free parmameters of the fit
        true_parameter : list of numpy-array containing true values for the free parameters.
        sample_sizes : list of integers or integer containing the number of samples drawn from the distribution for each experiment
        parameter_boundaries : If not None the uncertainty of the parameters is calculated (this can take some time!)
                               list of 2-float-tuple containing the lower and upper boundary of the free parameters.
                               If None is passed for a free parameters instead of the tuple its uncertainty is not calculated.
        """
        if not self.is_multi_dimensional:
            sample_sizes = [sample_sizes]
        result = []
        for i, true_parameter in enumerate(true_parameters):
            if i % 10 == 0:
                print(i)
            parameters = self.mapping(true_parameter)
            data = []
            for p, distribution, s in zip(parameters, self.distributions, sample_sizes):
                data.append(distribution.rvs(size=np.random.poisson(s), **p))
            weights = [np.ones(len(d)) for d in data]
            r = self._fit(initial_parameters, data, weights, self.mapping)
            if parameter_boundaries is None:
                uncertainties = None
            else:
                # Save cached result from user-fit
                old_r = self.r
                self.r = r
                #self.ugly_and_fast = True
                uncertainties = self.get_uncertainties(parameter_boundaries, data, weights)
                self.r = old_r
            result.append((true_parameter, r, uncertainties))
        return result
    
    def get_significance(self, parameter_values, data, weights=None):
        """
        Returns the significance of free parameters with respect to a null-hypothesis
        You have to call fit before calling this function

        Parameters
        ----------
        parameter_values : sequence containing fixed values for the free parameters under the null-hypothesis.
                           If None is passed for a parameter the corresponding parameter will be fitted.
        data : numpy-array or list of numpy-array containing the data
        weights : numpy-array or list of numpy-array containing the weights. If None is passed a weight of one is assumed for each event.
        """
        if self.r is None:
            raise RuntimeError("Please call fit first")
        data, weights = self._ensure_dimension(data, weights)
        parameter_positions = [i for i, v in enumerate(parameter_values) if v is not None]
        likelihood_profile_function = self._get_likelihood_profile_function(list(self.r.x), parameter_positions)
        n = likelihood_profile_function([v for v in parameter_values if v is not None], data, weights)
        return np.sqrt(2*(n-self.r.fun))
