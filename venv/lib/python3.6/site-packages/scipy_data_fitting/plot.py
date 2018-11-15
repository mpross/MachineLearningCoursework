import matplotlib.pyplot

class Plot:
    """
    Interface for creating and saving plots of `scipy_data_fitting.Fit`
    using [matplotlib][1].

    Many style options can be configured using [matplotlibrc][2].

    Example:

        #!python
        >>> fit = Fit()
        >>> # do things with fit until it's ready to plot
        >>> plot = Plot(fit=fit)
        >>> plot.save(fit.name + '.png')
        >>> plot.close()

    [1]: http://matplotlib.org/
    [2]: http://matplotlib.org/users/customizing.html
    """

    def __init__(self, fit=None):
        self.fit = fit
        """
        The `scipy_data_fitting.Fit` instance to use for the fit.
        """

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def options(self):
        """
        Dictionary of options which affect the plot style.

        Must contain the keys `data` and `fit` whose values are dictionaries.

        Options given in `data` and `fit` are passed as keyword arguments
        to [`matplotlib.pyplot.plot`][1] for the corresponding plot.

        Other options:

        - `points` is the number of points to use when generating the fit plot.

        Default:

            #!python
            {
                'data': {'marker': '.', 'linestyle': 'None'},
                'fit': {},
                'points': 300,
            }

        [1]: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
        """
        if not hasattr(self, '_options'):
            self._options = {
                'data': {'marker': '.', 'linestyle': 'None'},
                'fit': {},
                'points': 300,
            }
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    @property
    def figure(self):
        """
        The [`matplotlib.pyplot.figure`][1] instance.

        [1]: http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
        """
        if not hasattr(self, '_figure'): self._figure = matplotlib.pyplot.figure()
        return self._figure

    @property
    def plot(self):
        """
        The plot object, see [Pyplot][1].

        If one does not exist, it will be created first with [`add_subplot`][2].

        [1]: http://matplotlib.org/api/pyplot_api.html
        [2]: http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.add_subplot
        """
        if not hasattr(self, '_plot'):
            plot = self.figure.add_subplot(111)
            pointspace = self.fit.pointspace(num=self.options['points'])

            if any(v is not None for v in self.fit.data.error):
                plot_data = plot.errorbar
                self.options['data']['xerr'] = self.fit.data.error[0]
                self.options['data']['yerr'] = self.fit.data.error[1]
            else:
                plot_data = plot.plot

            plot_data(*pointspace['data'], **self.options['data'])
            plot.plot(*pointspace['fit'], **self.options['fit'])

            text = {}
            for v in ('independent', 'dependent'):
                meta = getattr(self.fit, v)
                text[v] = {
                  'name': meta['name'] if 'name' in meta else '',
                  'units': ' (' + meta['units'] + ')' if 'units' in meta else ''
                }

            plot.set_xlabel(text['independent']['name'] + text['independent']['units'])
            plot.set_ylabel(text['dependent']['name'] + text['dependent']['units'])

            self._plot = plot

        return self._plot

    def save(self, path, **kwargs):
        """
        Save the plot to the file at `path`.

        Any keyword arguments are passed to [`matplotlib.pyplot.savefig`][1].

        [1]: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
        """
        self.plot if not hasattr(self, '_plot') else None
        self.figure.savefig(path, **kwargs)

    def close(self):
        """
        Closes `scipy_data_fitting.Plot.figure` with [`matplotlib.pyplot.close`][1].

        This should always be called after the plot object is no longer needed,
        e.g. after saving it to disk with `scipy_data_fitting.Plot.save`.

        [1]: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.close
        """
        matplotlib.pyplot.close(self.figure)
