#!/usr/bin/env python3

import scipy.optimize
import numpy as np
import array
import collections


def Minuit(fun, x0, args=(), bounds=None, verbose=0, step_size=0.01, n_iterations=500, delta_one_sigma=0.5, **options):
    """
    Interface to ROOT Minuit
    @param bounds list of tuple with limits for each variable
    @param verbose -1 (silent) 0 (normal) 1 (verbose)
    @param step_size float or Sequence
    @param n_iterations maximum number of iterations
    @param delta_one_sigma delta in loss function which corresponds to one sigma errors
    """
    from ROOT import TMinuit, Double, Long
    n_parameters = len(x0)

    def to_minimize(npar, deriv, f, apar, iflag):
        f[0] = fun(np.array([apar[i] for i in range(n_parameters)]), *args)

    minuit = TMinuit(n_parameters)
    minuit.SetFCN(to_minimize)

    minuit.SetPrintLevel(verbose)

    ierflg = np.array(0, dtype=np.int32)
    minuit.mnexcm("SET ERR", np.array([delta_one_sigma]), 1, ierflg)

    if ierflg > 0:
        raise RuntimeError("Error during \"SET ERR\" call")

    for i in range(n_parameters):
        low, high = 0.0, 0.0 # Zero seems to mean no limit
        if bounds is not None:
            low, high = bounds[i]
        minuit.mnparm(i, 'Parameter_{}'.format(i), x0[i], step_size[i] if isinstance(step_size, collections.Sequence) else step_size, low, high, ierflg)
        if ierflg > 0:
            raise RuntimeError("Error during \"nmparm\" call for parmameter {}".format(i))

    minuit.mnexcm("MIGRAD", np.array([n_iterations, 0.01], dtype=np.float64), 2, ierflg)
    
    if ierflg > 0:
        raise RuntimeError("Error during \"MIGRAD\" call")

    xbest = np.zeros(n_parameters)
    xbest_error = np.zeros(n_parameters)

    p, pe = Double(0), Double(0)
    for i in range(n_parameters):
        minuit.GetParameter(i, p, pe)
        xbest[i] = p
        xbest_error[i] = pe

    buf = array.array('d', [0.]*(n_parameters**2))
    minuit.mnemat(buf, n_parameters) # retrieve error matrix
    hessian = np.array(buf).reshape(n_parameters, n_parameters)

    amin = Double(0.) # value of fcn at minimum
    edm = Double(0.) # estimated distance to mimimum
    errdef = Double(0.) # delta_fcn used to define 1 sigma errors
    nvpar = np.array(0, dtype=np.int32) # number of variable parameters
    nparx = np.array(0, dtype=np.int32) # total number of parameters
    icstat = np.array(0, dtype=np.int32) # status of error matrix: 3=accurate 2=forced pos. def 1= approximative 0=not calculated
    minuit.mnstat(amin, edm, errdef, nvpar, nparx, icstat)

    return scipy.optimize.OptimizeResult(x=xbest, fun=amin, hessian=hessian, success=(ierflg == 0), status=ierflg)

