#!/usr/bin/env python3

import numpy as np
from numpy.polynomial import chebyshev
import scipy
import scipy.stats
from scipy.stats import rv_continuous
import functools


def get_shape_parameters(distribution):
    """
    Returns all shape parameters of a scipy distribution
    including loc and scale
    Parameters
    ----------
    distribution : scipy.stats.rv_continuous
    """
    shape_parameters = []
    if distribution.shapes is not None:
        shape_parameters += distribution.shapes.split(', ')
    shape_parameters += ['loc', 'scale']
    return shape_parameters


class rv_mixture(rv_continuous):
    """ Mixture of distributions
    %(before_notes)s
    Notes
    -----
    The probability density function for `mixture` is given
    by the weighted sum of probability densities of all sub distributions
    mixture.pdf(x, shapes) = c_a a.pdf(x, shapes_a) + c_b b.pdf(x, shapes_b) + ...
    `mixture` takes all shape parameters of its sub distributions as a shape parameters.
    And in addition there is one shape parameter per sub distribution for the relative normalisation.
    The names of the shape parameters are the same as in the sub distribution with a prefix:
    E.g.
    rv_mixture([('Gauss', scipy.stats.norm), ('Gamma', scipy.stats.gamma)])
    has the following shape parameters (in this order):
      - Gauss_norm
      - Gamma_norm
      - Gauss_loc
      - Gauss_scale
      - Gamma_a
      - Gamma_loc
      - Gamma_scale

    So, first all the normalisation parameters, secondly the shape parameters of the individual distributions.
    Examples
    --------
    Create a new mixture distribution
    >>> import scipy.stats
    >>> import numpy as np
    >>> mixture = scipy.stats.rv_mixture([('Gauss', scipy.stats.norm), ('Gamma', scipy.stats.gamma)])
    >>> mixture.pdf(1.0, Gauss_norm=0.2, Gamma_norm=0.8, Gauss_loc=1.0, Gauss_scale=2.0, Gamma_a=2.0, Gamma_loc=2.0, Gamma_scale=1.0)
    0.039894228040143274
    >>> mixture.cdf(2.0, Gauss_norm=0.2, Gamma_norm=0.8, Gauss_loc=1.0, Gauss_scale=2.0, Gamma_a=2.0, Gamma_loc=2.0, Gamma_scale=1.0)
    0.13829249225480264
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3.0, 8.0, 100)
    >>> plt.title("Mixture PDF")
    >>> plt.plot(x, mixture.pdf(x, 0.3, 0.7, 1.0, 1.0, 3.0, 2.0, 1.0), label='Mixture')
    >>> plt.plot(x, 0.3 * scipy.stats.norm.pdf(x, 1.0, 1.0), label='Gauss')
    >>> plt.plot(x, 0.7 * scipy.stats.gamma.pdf(x, 3.0, 2.0, 1.0), label='Gamma')
    >>> plt.show()
    %(example)s
    """
    def __init__(self, distributions, *args, **kwargs):
        """
        Create a new mixture model given a number of distributions
        Parameters
        ----------
         distributions : list of tuples of the form (string, scipy.stats.rv_continuous)
           A list of names and scipy.stats.rv_continuous objects which define the mixture model.
           The names are used as prefix for the shape parameters of the mixture distribution.
        """
        self._ctor_argument = distributions
        self._distributions = []
        self._components = []
        self._distribution_norms = []
        self._distribution_shapes = []
        for component, distribution in distributions:
            self._distributions.append(distribution)
            self._components.append(component)
            self._distribution_norms.append('{}_norm'.format(component))
            self._distribution_shapes.append(['{}_{}'.format(component, s) for s in get_shape_parameters(distribution)])
        kwargs['shapes'] = ', '.join(sum(self._distribution_shapes, self._distribution_norms))
        super(rv_mixture, self).__init__(*args, **kwargs)

    def _extract_positional_arguments(self, parameters):
        """
        scipy passes the arguments as positional arguments,
        this method splits up the individual positional arguments
        into the norm arguments and the shape arguments for each individual component
        """
        norm_values, parameters = parameters[:len(self._distribution_norms)], parameters[len(self._distribution_norms):]
        shape_values = []
        for shape in self._distribution_shapes:
            shape_values.append(parameters[:len(shape)])
            parameters = parameters[len(shape):]
        norm_values = np.maximum(np.array(norm_values), 0.0)
        total_norm = np.sum(norm_values, axis=0)
        return norm_values / total_norm, shape_values

    def _pdf(self, x, *args):
        """
        The combined PDF is the sum of the distribution PDFs weighted by their individual norm factors
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        return np.sum((norm * distribution.pdf(x, *shape) for norm, distribution, shape in zip(norm_values, self._distributions, shape_values)), axis=0)

    def _cdf(self, x, *args):
        """
        The combined CDF is the sum of the distribution CDFs weighted by their individual norm factors
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        return np.sum((norm * distribution.cdf(x, *shape) for norm, distribution, shape in zip(norm_values, self._distributions, shape_values)), axis=0)

    def _rvs(self, *args):
        """
        Generates random numbers using the individual distribution random generator
        with a probability given by their individual norm factors
        """
        norm_values, shape_values = self._extract_positional_arguments(args)
        choices = self._random_state.choice(len(norm_values), size=self._size, p=norm_values)
        result = np.zeros(self._size)
        for i, (distribution, shape) in enumerate(zip(self._distributions, shape_values)):
            mask = choices == i
            result[mask] = distribution.rvs(size=mask.sum(), *shape, random_state=self._random_state)
        return result

    def _updated_ctor_param(self):
        """
        Set distributions as additional constructor argument
        """
        dct = super(rv_mixture, self)._updated_ctor_param()
        dct['distributions'] = self._ctor_argument
        return dct

    def _argcheck(self, *args):
        """
        We allow for arbitrary arguments.
        The sub distributions will check their arguments later anyway.
        The default _argcheck method restricts the arguments to be positive.
        """
        return True


class rv_support(rv_continuous):
    """
    Generates a distribution with a restricted support using a unrestricted distribution.

    %(before_notes)s

    Notes
    -----
    There are no additional shape parameters except for the loc and scale.
    The pdf and cdf are defined as stepwise functions from the provided histogram.
    In particular the cdf is not interpolated between bin boundaries and not differentiable.

    %(after_notes)s

    %(example)s

    data = scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5)
    hist = np.histogram(data, bins=100)
    template = scipy.stats.rv_histogram(hist)

    """

    def __init__(self, distribution, low, high, *args, **kwargs):
        """
        Create a new distribution using the given distribution
        Parameters
        ----------
        distribution : a scipy.stats distribution
        low : lower limit of the support
        high : higher limit of the support
        """
        self.distribution = distribution
        self.low = low
        self.high = high
        kwargs['shapes'] = distribution.shapes
        super(rv_support, self).__init__(*args, **kwargs)

    #@functools.lru_cache(maxsize=1000)
    def _get_norm(self, args):
        norm = self.distribution.cdf(self.high, *args) - self.distribution.cdf(self.low, *args)
        return np.where(norm <= 0, 1e9, norm)
    
    #@functools.lru_cache(maxsize=1000)
    def _get_low(self, args):
        return self.distribution.cdf(low, *args)

    def _pdf(self, x, *args):
        """
        PDF of the distribution
        """
        norm = self._get_norm(args)
        return np.where((x < self.low) | (x > self.high), 0.0, self.distribution.pdf(x, *args) / norm)
    
    def _cdf(self, x, *args):
        """
        CDF calculated from the distribution
        """
        low = self._get_low_probability_mass(args)
        norm = self._get_norm(args)
        return np.where(x < self.low, 0.0, np.where(x > self.high, 1.0, (self.distribution.cdf(x, *args) - low) / norm))
    
    def _rvs(self, *args):
        """
        Random numbers distributed like the original distribution with restricted support
        """
        rvs = self.distribution.rvs(*args, size=self._size)
        while len(rvs) < self._size[0]:
            rvs = np.append(rvs, self.distribution.rvs(*args, size=self._size))
        return rvs[:self._size[0]]
    
    def _updated_ctor_param(self):
        """
        Set additional constructor argument
        """
        dct = super(rv_support, self)._updated_ctor_param()
        dct['distribution'] = self.distribution
        dct['low'] = self.low
        dct['high'] = self.high
        return dct


class chebyshev_gen(rv_continuous):
    """
    Chebyshev polynom distribution

    %(before_notes)s

    Notes
    -----

    %(after_notes)s

    %(example)s
    """
    def __init__(self, degree, *args, **kwargs):
        """
        Parameters
        ----------
        degree : the degree of the chebyshev polynom as integer
        """
        self.degree = degree
        kwargs['shapes'] = ', '.join(['a_{}'.format(i) for i in range(self.degree + 1)])
        super(chebyshev_gen, self).__init__(*args, **kwargs)

    def _pdf(self, x, *args):
        """
        Return PDF of the polynom function
        """
        # TODO Fix support for large args
        coeffs = np.array([c[0] for c in args])
        coeffs = coeffs / coeffs.sum(axis=0)
        # +1 so that the pdf is always positive
        pdf_not_normed = chebyshev.chebval(x, coeffs) + 1
        integral_coeffs = chebyshev.chebint(coeffs)
        norm = chebyshev.chebval(1, integral_coeffs) - chebyshev.chebval(-1, integral_coeffs) + 2
        pdf = np.where(np.abs(x) > 1.0, 0.0, pdf_not_normed / norm)
        return pdf
    
    def _cdf(self, x, *args):
        """
        Return CDF of the polynom function
        """
        coeffs = np.array([c[0] for c in args])
        coeffs = coeffs / coeffs.sum(axis=0)
        coeffs = chebyshev.chebint(coeffs)
        cdf_not_scaled = chebyshev.chebval(x, coeffs) + x + 1 - chebyshev.chebval(-1, coeffs)
        integral_coeffs = chebyshev.chebint(coeffs)
        scale = chebyshev.chebval(1, coeffs) + 2 - chebyshev.chebval(-1, coeffs)
        cdf = np.where(x < -1.0, 0.0, np.where(x > 1.0, 1.0, cdf_not_scaled / scale))
        return cdf

    def _updated_ctor_param(self):
        """
        Set the n degree as additional constructor argument
        """
        dct = super(chebyshev_gen, self)._updated_ctor_param()
        dct['degree'] = self.degree
        return dct
    
    def _argcheck(self, *args):
        """
        We allow for semi-positive arguments.
        """
        return np.all(np.array(args) >= 0)

chebyshev_1 = chebyshev_gen(1, name='chebyshev_1', longname="A Polynom Function of degree 1")
chebyshev_2 = chebyshev_gen(2, name='chebyshev_2', longname="A Polynom Function of degree 2")
chebyshev_3 = chebyshev_gen(3, name='chebyshev_3', longname="A Polynom Function of degree 3")
chebyshev_4 = chebyshev_gen(4, name='chebyshev_4', longname="A Polynom Function of degree 4")
chebyshev_5 = chebyshev_gen(5, name='chebyshev_5', longname="A Polynom Function of degree 5")
chebyshev_6 = chebyshev_gen(6, name='chebyshev_6', longname="A Polynom Function of degree 6")
