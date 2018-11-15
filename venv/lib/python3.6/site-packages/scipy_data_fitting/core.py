import scipy.constants

def get_constant(value):
    """
    When `value` is a string, get the corresponding constant from [`scipy.constants`][1].

    [1]: http://docs.scipy.org/doc/scipy/reference/constants.html
    """
    if type(value) is str:
        if hasattr(scipy.constants, value):
            return getattr(scipy.constants, value)
        else:
            return scipy.constants.physical_constants[value][0]
    else:
        return value

def prefix_factor(dictionary):
    """
    Searches `dictionary` for the key `prefix`.
    If found, lookup the value with `scipy_data_fittings.core.get_constant`,
    else return 1.
    """
    if 'prefix' in dictionary:
        return get_constant(dictionary['prefix'])
    else:
        return 1
