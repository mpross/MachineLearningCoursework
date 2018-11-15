"""
**Data Fitting with SciPy**

Complete pipeline for easy data fitting with Python.

This package is registered on the Python Package Index (PyPI) at
[pypi.python.org/pypi/scipy-data_fitting](https://pypi.python.org/pypi/scipy-data_fitting).

The source is hosted on GitHub at
[github.com/razor-x/scipy-data_fitting](https://github.com/razor-x/scipy-data_fitting).

This is the pdoc generated documentation,
please see the README at either location above
for more details about this package.

There is also pdoc documentation for the [figure creation subpackage](figure).
"""

from .version import __version__
from .data import Data
from .fit import Fit
from .model import Model
from .plot import Plot

__all__ = ['Data', 'Fit', 'Model', 'Plot']
