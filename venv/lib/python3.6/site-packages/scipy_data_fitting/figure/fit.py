import json
import numpy

class Fit:
    """
    Loads a fit from a json file into a `figure.Fit` object.
    """

    def __init__(self, path=None):
        self.path = path
        """
        Path to the json file containing fit to load.
        """

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def data(self):
        """
        Data points as a [`numpy.ndarray`][1] in the form

            #!python
            [
                [ x1, x2, x3, ... ],
                [ y1, y2, y3, ...]
            ]

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
        """
        if not hasattr(self, '_data'): self._load()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def fit(self):
        """
        Fit points as a [`numpy.ndarray`][1] in the form

            #!python
            [
                [ x1, x2, x3, ... ],
                [ y1, y2, y3, ...]
            ]

        [1]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
        """
        if not hasattr(self, '_fit'): self._load()
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def meta(self):
        """
        A dictionary of metadata related to the fit.
        """
        if not hasattr(self, '_meta'): self._load()
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def maps(self):
        """
        A dictionary of dictionaries.
        Each dictionary defines a map which is used to extend the metadata.

        The precise way maps interact with the metadata is defined by `figure.fit._extend_meta`.
        That method should be redefined or extended to suit specific use cases.
        """
        if not hasattr(self, '_maps'):
            maps = {}
            maps['tex_symbol'] = {}
            maps['siunitx'] = {}
            maps['value_transforms'] = {
                '__default__': lambda x: round(x, 2),
            }
            self._maps = maps
        return self._maps

    @maps.setter
    def maps(self, value):
        self._maps = value

    def _load(self):
        if not self.path: raise RuntimeError('Must specify path to load fit from file.')
        raw = json.load(open(self.path))

        self.data = numpy.array(raw['data']).T
        self.fit = numpy.array(raw['fit']).T

        if 'meta' in raw:
            self.meta = raw['meta']
        else:
            self.meta = {}
        for key in self.meta: self._extend_meta(self.meta[key])

    def _extend_meta(self, meta):
        if isinstance(meta, str): return None
        if isinstance(meta, dict): meta = [meta]
        for item in meta:
            if 'name' in item: key = item['name']
            if 'symbol' in item: key = item['symbol']

            if 'value' in item:
                item['disply_value'] = str(self._value_transform(key)(item['value']))

            item['tex_symbol'] = self._get_tex_symbol(key)

            if 'units' in item:
                item['siunitx'] = self._get_siunitx(item['units'])
            else:
                item['siunitx'] = ''

    def _get_siunitx(self, key):
        table = self.maps['siunitx']
        if key in table: return table[key]
        return ''

    def _get_tex_symbol(self, key):
        table = self.maps['tex_symbol']
        if key in table: return table[key]
        return key

    def _value_transform(self, key):
        table = self.maps['value_transforms']
        if key in table:
            return table[key]
        else:
            return table['__default__']
