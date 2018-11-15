import functools
import sympy

class Model:
    """
    A model organizes symbols, expressions and replacements rules by name.

    Example:

        #!python
        >>> model = Model()
        >>> model.add_symbols('y', 'x', 'm', 'b')
        >>> y, m, x, b = model.get_symbols('y', 'x', 'm', 'b')
        >>> model.expressions['line'] = y
        >>> model.replacements['slope_intercept'] = (y, m * x + b)
        >>> line = model.replace('line', 'slope_intercept')
        m * x + b
        >>> function = model.lambdify(line, ('m', 'x', 'b'))
        >>> function(1, 2, 3)
        5
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
    def expressions(self):
        """
        Dictionary to store SymPy expressions by name.
        """
        if not hasattr(self, '_expressions'): self._expressions = {}
        return self._expressions

    @expressions.setter
    def expressions(self, value):
        self._expressions = value

    @property
    def replacements(self):
        """
        Dictionary to store replacement rules by name.
        Each value is a tuple of SymPy expressions: `(expression, replacement)`.
        """
        if not hasattr(self, '_replacements'): self._replacements = {}
        return self._replacements

    @replacements.setter
    def replacements(self, value):
        self._replacements = value

    @property
    def replacement_groups(self):
        """
        Dictionary to store a sequence of replacements by name.
        Each value is a list of names that will be looked up
        in `scipy_data_fitting.Model.replacements`.

        When used to make substitutions, replacements will be applied
        one at a time in the order given.
        """
        if not hasattr(self, '_replacement_groups'): self._replacement_groups = {}
        return self._replacement_groups

    @replacement_groups.setter
    def replacement_groups(self, value):
        self._replacement_groups = value

    @property
    def symbols(self):
        """
        Dictionary to store symbols by name.

        Add symbols directly, or with `scipy_data_fitting.Model.add_symbol`
        and `scipy_data_fitting.Model.add_symbols`.
        """
        if not hasattr(self, '_symbols'): self._symbols = {}
        return self._symbols

    @symbols.setter
    def symbols(self, value):
        self._symbols = value

    def symbol(self, name):
        """
        Function to provide a shorthand for `self.symbols[name]`.
        """
        return self.symbols[name]

    def add_symbol(self, name, string=None):
        """
        Add a symbol with key `name` to `scipy_data_fitting.Model.symbols`.
        Optionally, specify an alternative `string` to pass to [`sympy.Symbol`][1],
        otherwise `name` is used.

        [1]: http://docs.sympy.org/dev/modules/core.html#id4
        """
        if not string: string = name
        self.symbols[name] = sympy.Symbol(string)

    def add_symbols(self, *names):
        """
        Pass any number of strings to add symbols to `scipy_data_fitting.Model.symbols`
        using `scipy_data_fitting.Model.add_symbol`.

        Example:

            #!python
            >>> model.add_symbols('x', 'y', 'z')
        """
        for name in names:
            self.add_symbol(name)

    def get_symbols(self, *symbols):
        """
        A tuple of symbols by name.

        Example:

            #!python
            >>> x, y, z = model.get_symbols('x', 'y', 'z')
        """
        return ( self.symbol(s) for s in symbols )

    def replace(self, expression, replacements):
        """
        All purpose method to reduce an expression by applying
        successive replacement rules.

        `expression` is either a SymPy expression
        or a key in `scipy_data_fitting.Model.expressions`.

        `replacements` can be any of the following,
        or a list of any combination of the following:

        - A replacement tuple as in `scipy_data_fitting.Model.replacements`.
        - The name of a replacement in `scipy_data_fitting.Model.replacements`.
        - The name of a replacement group in `scipy_data_fitting.Model.replacement_groups`.

        Examples:

            #!python
            >>> model.replace(x + y, (x, z))
            z + y

            >>> model.replace('expression', (x, z))
            >>> model.replace('expression', 'replacement')
            >>> model.replace('expression', ['replacement_1', 'replacement_2'])
            >>> model.replace('expression', ['replacement', 'group'])
        """
        # When expression is a string,
        # get the expressions from self.expressions.
        if isinstance(expression, str):
            expression = self.expressions[expression]

        # Allow for replacements to be empty.
        if not replacements:
            return expression

        # Allow replacements to be a string.
        if isinstance(replacements, str):
            if replacements in self.replacements:
                return self.replace(expression, self.replacements[replacements])
            elif replacements in self.replacement_groups:
                return self.replace(expression, self.replacement_groups[replacements])

        # When replacements is a list of strings or tuples,
        # Use reduce to make all the replacements.
        if all(isinstance(item, str) for item in replacements) \
        or all(isinstance(item, tuple) for item in replacements):
            return functools.reduce(self.replace, replacements, expression)

        # Otherwise make the replacement.
        return expression.replace(*replacements)

    def lambdify(self, expression, symbols, **kwargs):
        """
        Converts a SymPy expression into a function using [`sympy.lambdify`][1].

        `expression` can be a SymPy expression or the name of an expression
        in `scipy_data_fitting.Model.expressions`.

        `symbols` can be any of the following,
        or a list of any combination of the following:

        - A SymPy symbol.
        - The name of a symbol in `scipy_data_fitting.Model.symbols`.

        Additional keyword arguments are passed to [`sympy.lambdify`][1].

        [1]: http://docs.sympy.org/latest/modules/utilities/lambdify.html#sympy.utilities.lambdify.lambdify
        """
        if isinstance(expression, str):
            expression = self.expressions[expression]

        if hasattr(symbols, '__iter__'):
            variables = []
            for s in symbols:
                if isinstance(s, str):
                    variables.append(self.symbol(s))
                else:
                    variables.append(s)
        else:
            if isinstance(symbols, str):
                variables = (self.symbol(symbols), )
            else:
                variables = (symbols, )

        return sympy.lambdify(tuple(variables), expression, **kwargs)
