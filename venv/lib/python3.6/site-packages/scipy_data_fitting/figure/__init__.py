"""
**Figure creation subpackage.**

_This subpackage is in beta._

Use this subpackage to create publication quality figures
with [matplotlib][1] from fits generated with [`scipy_data_fitting`][2].

This subpackage works independently from [`scipy_data_fitting`][2]:
it only assumes json fit files formatted according to
[`scipy_data_fitting.Fit.to_json`][3] with [`meta=fit.metadata`][4].

Use `figure.Fit` to load fits saved as json files into a `figure.Fit` object.
Create a [`matplotlib.figure`][5] and manage subplots with `figure.Plot`.

[1]: http://matplotlib.org/
[2]: https://pythonhosted.org/scipy-data_fitting/
[3]: https://pythonhosted.org/scipy-data_fitting/#scipy_data_fitting.Fit.to_json
[4]: https://pythonhosted.org/scipy-data_fitting/#scipy_data_fitting.Fit.metadata
[5]: http://matplotlib.org/api/figure_api.html
"""

from .fit import Fit
from .plot import Plot

__all__ = ['Fit', 'Plot']
