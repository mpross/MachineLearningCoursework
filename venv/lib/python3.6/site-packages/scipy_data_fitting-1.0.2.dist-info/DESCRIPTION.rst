Data Fitting with SciPy
=======================

|PyPI| |GitHub-license| |Requires.io| |CircleCI| |Codecov|

    Built from `makenew/python-package <https://github.com/makenew/python-package>`__.

.. |PyPI| image:: https://img.shields.io/pypi/v/scipy-data_fitting.svg
   :target: https://pypi.python.org/pypi/scipy-data_fitting
   :alt: PyPI
.. |GitHub-license| image:: https://img.shields.io/github/license/razor-x/scipy-data_fitting.svg
   :target: ./LICENSE.txt
   :alt: GitHub license
.. |Requires.io| image:: https://img.shields.io/requires/github/razor-x/scipy-data_fitting.svg
   :target: https://requires.io/github/razor-x/scipy-data_fitting/requirements/
   :alt: Requires.io
.. |CircleCI| image:: https://img.shields.io/circleci/project/razor-x/scipy-data_fitting.svg
   :target: https://circleci.com/gh/razor-x/scipy-data_fitting
   :alt: CircleCI
.. |Codecov| image:: https://img.shields.io/codecov/c/github/razor-x/scipy-data_fitting.svg
   :target: https://codecov.io/gh/razor-x/scipy-data_fitting
   :alt: Codecov

Description
-----------

|figure|

.. |figure| image:: https://raw.github.com/razor-x/scipy-data_fitting/master/plot.png

Complete pipeline for easy data fitting with Python 3.

Check out the `example fits on Fitalyzer`_.
See the `Fitalyzer README`_ for details on how to use Fitalyzer for
visualizing your fits.

.. _example fits on Fitalyzer: http://io.evansosenko.com/fitalyzer/?firebase=scipy-data-fitting
.. _Fitalyzer README: https://github.com/razor-x/fitalyzer

Installation
------------

This package is registered on the `Python Package Index (PyPI)`_
as scipy_data_fitting_.

Add this line to your application's requirements.txt

::

    scipy_data_fitting

and install it with

::

    $ pip install -r requirements.txt

If you are writing a Python package which will depend on this,
add this to your requirements in ``setup.py``.

Alternatively, install it directly using pip with

::

    $ pip install scipy_data_fitting

.. _scipy_data_fitting: https://pypi.python.org/pypi/scipy-data_fitting
.. _Python Package Index (PyPI): https://pypi.python.org/

Documentation
-------------

Documentation is generated from source with `pdoc`_.
The latest version is hosted at `pythonhosted.org/scipy-data\_fitting/`_.

To get started quickly, check out the `examples`_.

Then, refer to the source documentation for details on how to use each class.

.. _pdoc: https://pypi.python.org/pypi/pdoc/
.. _pythonhosted.org/scipy-data\_fitting/: https://pythonhosted.org/scipy-data_fitting/
.. _examples: https://github.com/razor-x/scipy-data_fitting/tree/master/examples

Basic Usage
-----------

.. code:: python

    from scipy_data_fitting import Data, Model, Fit, Plot

    # Load data from a CSV file.
    data = Data('linear')
    data.path = 'linear.csv'
    data.error = (0.5, None)

    # Create a linear model.
    model = Model('linear')
    model.add_symbols('t', 'v', 'x_0')
    t, v, x_0 = model.get_symbols('t', 'v', 'x_0')
    model.expressions['line'] = v * t + x_0

    # Create the fit using the data and model.
    fit = Fit('linear', data=data, model=model)
    fit.expression = 'line'
    fit.independent = {'symbol': 't', 'name': 'Time', 'units': 's'}
    fit.dependent = {'name': 'Distance', 'units': 'm'}
    fit.parameters = [
        {'symbol': 'v', 'guess': 1, 'units': 'm/s'},
        {'symbol': 'x_0', 'value': 1, 'units': 'm'},
    ]

    # Save the fit result to a json file.
    fit.to_json(fit.name + '.json', meta=fit.metadata)

    # Save a plot of the fit to an image file.
    plot = Plot(fit)
    plot.save(fit.name + '.svg')
    plot.close()

Controlling the fitting process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above example will fit the line using the default algorithm
``scipy.optimize.curve_fit``.

For a linear fit, it may be more desirable to use a more efficient
algorithm.

For example, to use ``numpy.polyfit``, one could set a
``fit_function`` and allow both parameters to vary,

.. code:: python

    fit.parameters = [
        {'symbol': 'v', 'guess': 1, 'units': 'm/s'},
        {'symbol': 'x_0', 'guess': 1, 'units': 'm'},
    ]
    fit.options['fit_function'] = \
        lambda f, x, y, p0, **op: (numpy.polyfit(x, y, 1), )

Controlling the fitting process this way allows, for example,
incorporating error values and computing and returning goodness of fit
information.

See ``scipy_data_fitting.Fit.options`` for further details on how to
control the fit and also how to use `lmfit`_.

.. _lmfit: http://lmfit.github.io/lmfit-py/

Development and Testing
-----------------------

Source Code
~~~~~~~~~~~

The `scipy-data_fitting source`_ is hosted on GitHub.
Clone the project with

::

    $ git clone https://github.com/razor-x/scipy-data_fitting.git

.. _scipy-data_fitting source: https://github.com/razor-x/scipy-data_fitting

Requirements
~~~~~~~~~~~~

You will need `Python 3`_ with pip_.

Install the development dependencies with

::

    $ pip install -r requirements.devel.txt

.. _pip: https://pip.pypa.io/
.. _Python 3: https://www.python.org/

Tests
~~~~~

Lint code with

::

    $ python setup.py lint


Run tests with

::

    $ python setup.py test

or

::

    $ make test

Documentation
~~~~~~~~~~~~~

Generate documentation with pdoc by running

::

    $ make docs

Examples
~~~~~~~~

Run an example with

::

    $ python examples/example_fit.py

or run all the examples with

::

    $ make examples

Contributing
------------

Please submit and comment on bug reports and feature requests.

To submit a patch:

1. Fork it (https://github.com/razor-x/scipy-data_fitting/fork).
2. Create your feature branch (``git checkout -b my-new-feature``).
3. Make changes. Write and run tests.
4. Commit your changes (``git commit -am 'Add some feature'``).
5. Push to the branch (``git push origin my-new-feature``).
6. Create a new Pull Request.

License
-------

This Python package is licensed under the MIT license.

Warranty
--------

This software is provided "as is" and without any express or implied
warranties, including, without limitation, the implied warranties of
merchantibility and fitness for a particular purpose.


