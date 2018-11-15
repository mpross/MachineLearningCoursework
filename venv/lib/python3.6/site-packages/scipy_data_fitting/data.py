import numpy

from .core import get_constant

class Data:
    """
    Provides an interface to load data from files into
    [`numpy.ndarray`][1] objects.

    Example:

        #!python
        >>> data = Data()
        >>> data.path = 'path/to/data.csv'
        >>> data.scale = (1, 'kilo')
        >>> data.array

    [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    """

    def __init__(self, name=None):
        self.name = name
        """
        The identifier name for this object.
        """

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def path(self):
        """
        Path to the file containing data to load.
        """
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def array(self):
        """
        Data as a [`numpy.ndarray`][1] in the form

            #!python
            [
                [ x1, x2, x3, ... ],
                [ y1, y2, y3, ...]
            ]

        By default, if unset, this will be set on first access
        by calling `scipy_data_fitting.Data.load_data`.

        When loaded from file, the x and y values will be scaled according
        to `scipy_data_fitting.Data.scale`.

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
        """
        if not hasattr(self, '_array'): self._array = self.load_data()
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    @property
    def error(self):
        """
        Error associated with the data as a two element tuple of the form `(x_error, y_error)`.

        Both `x_error` and `y_error` is a [`numpy.ndarray`][1] or `None`.

        The array dimensions depend on how the error is specified:

        1. Symmetric constant error is a zero dimensional array: `error`.
        2. Asymmetric constant error is a one dimensional array: `[lower_error, upper_error]`.
        3. Symmetric error which varies for each point is an array with length equal
          to the number of points; each element is a zero dimensional array: `error`
        4. Asymmetric error which varies for each point is an array with length equal
          to the number of points; each element is a one dimensional array: `[lower_error, upper_error]`.

        This property can be set manually. If setting constant errors (cases 1 and 2 above),
        it is not necessary to explicitly use [`numpy.ndarray`][1] as the type will be converted automatically.

        For error that varies at each point (cases 3 and 4 above),
        the errors can be loaded from the file given by `scipy_data_fitting.Data.path`
        setting `scipy_data_fitting.Data.error_columns`.

        Defaults to `(None, None)` unless `scipy_data_fitting.Data.error_columns` is set,
        in which case this will be set on first access by calling `scipy_data_fitting.Data.load_error`.

        When loaded from file, the x and y values will be scaled according
        to `scipy_data_fitting.Data.scale`.

        Examples:

            #!python
            # (x_error, y_error)
            (0.1, 0.5)

            # (x_error, no y_error)
            (0.1, None)

            # ([x_lower_error, x_upper_error], y_error)
            ([0.1, 0.5], 2)

            # ([x_lower_error, x_upper_error], [y_lower_error, y_upper_error])
            ([0.1, 0.5], [2, 0.5])

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
        """
        if not hasattr(self, '_error'):
            if any(v is not None for v in self.error_columns):
                self._error = self.load_error()
            else:
                return (None, None)
        return self._error

    @error.setter
    def error(self, value):
        self._error = tuple( numpy.array(v) if v is not None else None for v in value )

    @property
    def scale(self):
        """
        Tuple `(x_scale, y_scale)` that defines how to scale data
        imported by `scipy_data_fitting.Data.load_data`
        and `scipy_data_fitting.Data.load_error`.

        If a scale is specified as a string, it will treated as a named physical constant
        and converted to the corresponding number using [`scipy.constants`][1].

        [1]: http://docs.scipy.org/doc/scipy/reference/constants.html
        """
        if not hasattr(self, '_scale'): self._scale = (1, 1)
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = tuple( get_constant(v) for v in value )

    @property
    def error_columns(self):
        """
        Two element tuple that defines what columns in the file given
        by `scipy_data_fitting.Data.path` are the error values for each data point.

        The first element corresponds to the x error and the second to the y error.

        Each element is either an integer which gives the column,
        a two element tuple of integers, or `None`.

        In Python, indexes are zero-based, so the first column is `0`.

        Examples:

            #!python
            # (x_error, y_error)
            (2, 3)

            # (x_error, no y_error)
            (2, None)

            # ((x_lower_error, x_upper_error), y_error)
            ((2, 3), 4)

            # ((x_lower_error, x_upper_error), (y_lower_error, y_upper_error))
            ((2, 3), (4, 5))

        Defaults to `(None, None)`.
        """
        if not hasattr(self, '_error_columns'): return (None, None)
        return self._error_columns

    @error_columns.setter
    def error_columns(self, value):
        self._error_columns = value

    @property
    def genfromtxt_args(self):
        """
        Passed as keyword arguments to [`numpy.genfromtxt`][1]
        when called by `scipy_data_fitting.Data.load_data`.

        Default:

            #!python
            {
                'unpack': True,
                'delimiter': ',',
                'usecols': (0 ,1),
            }

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
        """
        if not hasattr(self, '_genfromtxt_args'):
            self._genfromtxt_args = {
                'unpack': True,
                'delimiter': ',',
                'usecols': (0 ,1),
            }
        return self._genfromtxt_args

    @genfromtxt_args.setter
    def genfromtxt_args(self, value):
        self._genfromtxt_args = value

    @property
    def genfromtxt_args_error(self):
        """
        Passed as keyword arguments to [`numpy.genfromtxt`][1]
        when called by `scipy_data_fitting.Data.load_error`.

        Even if defined here, the `usecols` value will always be reset based
        on `scipy_data_fitting.Data.error_columns` before being passed to [`numpy.genfromtxt`][1].

        If not set, this defaults to a copy of
        `scipy_data_fitting.Data.genfromtxt_args` on first access.

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
        """
        if not hasattr(self, '_genfromtxt_args_error'):
            self._genfromtxt_args_error = self.genfromtxt_args.copy()
        return self._genfromtxt_args_error

    @genfromtxt_args_error.setter
    def genfromtxt_args_error(self, value):
        self._genfromtxt_args_error = value

    def load_data(self):
        """
        Loads data from `scipy_data_fitting.Data.path` using [`numpy.genfromtxt`][1]
        and returns a [`numpy.ndarray`][2].

        Data is scaled according to `scipy_data_fitting.Data.scale`.

        Arguments to [`numpy.genfromtxt`][1] are controlled
        by `scipy_data_fitting.Data.genfromtxt_args`.

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
        [2]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
        """
        array = numpy.genfromtxt(self.path, **self.genfromtxt_args)
        for n, scale in enumerate(self.scale): array[n,:] *= self.scale[n]
        return array

    def load_error(self):
        """
        Loads error values from `scipy_data_fitting.Data.path` using [`numpy.genfromtxt`][1]
        and returns a two element tuple where each element is of a form described by
        cases 3 and 4 in `scipy_data_fitting.Data.error`.

        The columns to import are set by `scipy_data_fitting.Data.error_columns`.

        Values are scaled according to `scipy_data_fitting.Data.scale`.

        Arguments to [`numpy.genfromtxt`][1] are controlled
        by `scipy_data_fitting.Data.genfromtxt_args_error`.

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
        """
        usecols = []
        for v in self.error_columns:
            if v is None:
                pass
            elif isinstance(v, int):
                usecols.append(v)
            elif len(v) is 2:
                for n in v: usecols.append(n)
        self.genfromtxt_args_error['usecols'] = tuple(usecols)

        array = numpy.genfromtxt(self.path, **self.genfromtxt_args_error)

        error = []
        for n, v in enumerate(self.error_columns):
            if v is None:
                error.append(None)
            elif isinstance(v, int):
                if len(usecols) is 1:
                    error.append(array * self.scale[n])
                else:
                    error.append(array[0] * self.scale[n])
                    array = numpy.delete(array, (0), axis=(0))
            elif len(v) is 2:
                error.append(array[0:2] * self.scale[n])
                array = numpy.delete(array, (0, 1), axis=(0))

        return tuple(error)
