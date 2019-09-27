#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Module that specifies the leaky integration perception and decision model

Defines the Stimulator and Leaky class that generate the stimulation and
the model's response respectively.

Author: Luciano Paz
Year: 2017

"""

from __future__ import division, print_function, absolute_import, unicode_literals
from six import iteritems
import numpy as np
from scipy.special import gamma
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pprint, sympy, itertools, collections, warnings, re
try:
    from .leaky_integral_calculator import leaky_integral_calculator as _leaky_integral_calculator
    from .leaky_integral_calculator import lowergamma as lowergamma
    from .leaky_integral_calculator import uppergamma as uppergamma
    _c_api_available = True
except:
    warnings.warn("Unable to import the optimized c extension for the leaky integration! Runs will be significantly slower.", ImportWarning)
    _c_api_available = False
    from mpmath import gammainc as mpmath_gammainc
    uppergamma = lambda a, x, dtype=np.complex: np.array(np.frompyfunc(lambda aa, xx: np.complex(mpmath_gammainc(aa, xx)), 2, 1)(a, x), dtype=dtype)
    lowergamma = lambda a, x, dtype=np.complex: np.array(np.frompyfunc(lambda aa, xx: np.complex(mpmath_gammainc(aa, 0., xx)), 2, 1)(a, x), dtype=dtype)

sympy.uppergamma =  sympy.special.gamma_functions.uppergamma
sympy.lowergamma =  sympy.special.gamma_functions.lowergamma


def compatibalize_shapes(flatten=True, **kwargs):
    """
    compatibalize_shapes(**kwargs)
    
    Takes an unpacked dictionary **kwargs and returns a new dictionary
    with the same keys, but where all the values corresponding to the keys
    were broadcasted to the same shape using numpy.broadcast_array, and
    then flattened with C order.
    
    """
    broadcasted = np.broadcast_arrays(*tuple([arr for arr in kwargs.values()]))
    if flatten:
        return {k: arr.flatten(order='C') for k, arr in zip(kwargs.keys(), broadcasted)}
    else:
        return {k: arr for k, arr in zip(kwargs.keys(), broadcasted)}

class WrapperKwargs(object):
    """
    A wrapper class for functions that stores the default keyword 
    arguments dictionary (kwargs) as a member function. This allows one
    to define default argument values for a function from a dictionary.
    Then the wrapped function is called normally with the WrapperKwargs
    __call__ method.
    
    """
    def __init__(self, fun, **kwargs):
        """
        WrapperKwargs(fun, **kwargs)
        
        input:
            fun: a callable
            kwargs: an unpacked dictionary of keyword arguments, which
                will be internally stored and used when calling fun
        
        """
        self.fun = fun
        self._kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        """
        self.__call__(*args, **kwargs)
        
        Calls self.fun using the positional arguments args. The keyword
        arguments are used to update a copy of the stored kwargs given
        to self upon creation, and then inserted into the function call.
        
        """
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        return self.fun(*args, **_kwargs)

class CompatibleParameters(object):
    """
    CompatibleParameters is a container class. It holds entries almost
    like a dictionary. The differences are that its list of keys is
    immutable, and all corresponding the values are broadcasted to arrays
    with the same shape using compatibalize_shapes.
    
    The parameter values can be accessed both as attributes and items.
    i.e.
    parvalue = self.parname
    parvalue = self['parname']
    
    """
    def __init__(self, flatten=True, **kwargs):
        """
        CompatibleParameters(flatten=True, **kwargs)
        
        Initializes the instance. The parameter_names attribute, which
        act as dict keys, are the keys in kwargs. The initial values of
        the parameters attribute is an OrderedDict with parameter_names
        as keys and whose values are taken from the kwargs values.
        
        Input flatten must be a bool indicates whether the arrays must
        be flatten when calling compatibalize_shapes
        """
        self.flatten = flatten
        self.parameter_names = tuple([k for k in kwargs.keys()])
        self.parameters = collections.OrderedDict()
        for pn in self.parameter_names:
            self.parameters[pn] = np.array(kwargs[pn])
        self.compatibalize_parameters()
    
    def __setitem__(self, key, value):
        if key in self.parameter_names:
            self.parameters[key] = np.array(value)
        else:
            raise KeyError('Parameter "{0}" is not defined'.format(key))
        self.compatibalize_parameters()
    
    def __getattr__(self, name):
        if name not in self.parameter_names:
            raise AttributeError('Parameter "{0}" is not defined'.format(name))
        else:
            return self.parameters[name]
    
    def __str__(self):
        return self.parameters.__str__()
    
    def __repr__(self):
        return '{cls}(parameter_names={parameter_names}\nparameters={parameters})'.format(
                cls=self.__class__.__name__, parameter_names=self.parameter_names, parameters=self.parameters)
    
    def set_parameters(self, **kwargs):
        """
        self.set_parameters(**kwargs)
        
        Changes the values of the parameter keys provided from the input
        kwargs. If kwargs has a key that is not in self.parameter_names
        this function raise a KeyError. If this does not have any problems,
        then the updated parameters are passed through compatibalize_shapes.
        If an error occurs at any point during this process, no change is
        done self.parameters.
        
        """
        draft = self.parameters.copy()
        for pn in kwargs.keys():
            if pn not in self.parameter_names:
                raise KeyError('Parameter "{0}" is not defined'.format(key))
            draft[pn] = kwargs[pn]
        draft = compatibalize_shapes(draft)
        for pn in self.parameter_names:
            self.parameters[pn] = draft[pn]
    
    def get_parameters(self):
        """
        self.get_parameters()
        
        Returns a copy of the parameters attribute.
        
        """
        return self.parameters.copy()
    
    def get_parameter_names(self):
        return self.parameter_names
    
    def compatibalize_parameters(self):
        """
        self.compatibalize_parameters()
        
        Calls compatibalize_shapes on self.parameters and then resets
        the values from the output of that call.
        
        """
        d = compatibalize_shapes(flatten=self.flatten, **(self.parameters))
        for pn in self.parameter_names:
            self.parameters[pn] = d[pn]
    
    def get_unique_values(self, name, **kwargs):
        return np.unique(self.parameters[name], **kwargs)
    
    def as_array(self):
        """
        self.as_array()
        
        Returns a numpy structured array where each column has a different
        parameter_name, and the rows are the corresponding values. 
        
        """
        dtype = []
        arrs = []
        for k, v in iteritems(self.parameters):
            dtype.append((k, v.dtype))
            arrs.append(v)
        dtype = np.dtype(dtype)
        return np.array([np.array(tuple(a), dtype=dtype) for a in np.array(arrs).T])
    

class Stimulator(CompatibleParameters):
    """
    Stimulator class that is used to sample streams of vibration stimuli.
    It inherits from CompatibleParameters.
    
    """
    def __init__(self, slope=0., origin=1., alpha=1., adaptation_amplitude=1.,
                 adaptation_baseline=1., adaptation_tau_inv=0., background_var=0.,
                 background_mean=0.):
        """
        Stimulator(slope=0., origin=1., alpha=1., adaptation_amplitude=1.,
                 adaptation_baseline=1., adaptation_tau_inv=0., background_var=0.,
                 background_mean=0.)
        
        All input parameters of __init__ are passed to the CompatibleParameters
        super class and are hence the considered as the parameters whoes
        shape must always remain compatible and broadcasted.
        
        Input:
            slope: slope of the sigma time dependence. sigma(t) = slope * t + origin
            origin: origin of the sigma time dependence. sigma(t) = slope * t + origin
            alpha: exponent of the power law transfer function of sigma(t).
                sp(t) = sigma(t)**alpha
            adaptation_amplitude: amplitude, A, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_baseline: baseline, B, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_tau_inv: inverse time constant, Tau_inv_ad, of the
                adaptation function that multiplies the total raw input. This
                function is A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            background_var: background input's variance.
            background_mean: background input's mean value.
        
        """
        super(Stimulator, self).__init__(slope=slope, origin=origin, alpha=alpha,
                            adaptation_amplitude=adaptation_amplitude,
                            adaptation_baseline=adaptation_baseline,
                            adaptation_tau_inv=adaptation_tau_inv,
                            background_var=background_var,
                            background_mean=background_mean)
    
    def sample(self,t, nsamples, split_stochastic=True, use_gaussian_approx=False, **kwargs):
        """
        self.sample(t, nsamples, split_stochastic=True, use_gaussian_approx=False, **kwargs)
        
        Generate nsamples independent signal samples at the times provided
        by t.
        
        Input:
            t: arraylike with the times at which the samples must be collected.
            nsamples: int number of independent samples to generate at each
                time t and parameter tuples.
            split_stochastic: bool. If true, the method returns two arrays,
                the deterministic contribution and the stochastic contribution.
                If false, the sum of these two is returned.
            use_gaussian_approx: bool that indicates whether to approximate
                the stimulus generation process as a time inhomogenous weiner
                process. The default is false, and thus gets samples from
                a truncated half gaussian passed through a power law
                transfer function.
            kwargs: Can be any of the kwargs of the self.__init__ method.
                The default values are the attributes of self, but if
                a kwarg is provided to this function, its value will be
                used instead. Be aware that this does not overite self's
                attribute value.
            WARNING: t, and the optional **kwargs must be convertible to
            numpy.ndarrays, as this function will attempt to broadcast
            all inputs to ndarrays of the same shape, and the output
            will also be ndarrays with this shape.
        
        Output:
            stimulus or (deterministic_stim, stochastic_stim):
                A single output or a tuple of two values is returned
                depending on the value of split_stochastic. The outputs
                will be numpy.ndarrays with the shape of the broadcasted
                t, slope, etc, where slope, origin and the other
                parameters available at the __init__ method can be
                specified through the kwargs of this function. The default
                output shape will be equal to the broadcast of t, against
                the slope, origin, etc values used in this function. If
                these don't broadcast by default, then slope et al assumed
                to be flat and will be converted to 3D arrays where the
                last two axis are None. On the other hand, t will be
                assumed to be flat and will also be converted to a 3D
                array but with its non singleton dimension being axis=1.
                The last axis will represent the independent random samples.
                No checks are done to verify that these arrays are 1D,
                and it is possible that an exception could be raised
                if this were not true.
        
        See also Stimulator._sample
        
        """
        slope = kwargs.pop('slope', self.slope)
        origin =kwargs.pop('origin', self.origin)
        alpha = kwargs.pop('alpha', self.alpha)
        adaptation_amplitude = kwargs.pop('adaptation_amplitude', self.adaptation_amplitude)
        adaptation_baseline = kwargs.pop('adaptation_baseline', self.adaptation_baseline)
        adaptation_tau_inv = kwargs.pop('adaptation_tau_inv', self.adaptation_tau_inv)
        background_var = kwargs.pop('background_var', self.background_var)
        background_mean = kwargs.pop('background_mean', self.background_mean)
        
        try:
            if all([np.isscalar(x) or x.size==1 for x in [slope, origin, alpha, adaptation_amplitude, \
                    adaptation_baseline, adaptation_tau_inv, background_var, background_mean]]) or \
                    np.isscalar(t):
                raise Exception('Alternating to hardcoded reshaping')
            t, slope, origin, alpha, adaptation_amplitude, \
            adaptation_baseline, adaptation_tau_inv, background_var, \
            background_mean = \
                np.broadcast_arrays(t, slope, origin, alpha,
                                    adaptation_amplitude,
                                    adaptation_baseline, adaptation_tau_inv,
                                    background_var, background_mean)
            output_shape = t.shape + (nsamples,)
            t = np.broadcast_to(t, output_shape)
            slope = np.broadcast_to(slope, output_shape)
            origin = np.broadcast_to(origin, output_shape)
            alpha = np.broadcast_to(alpha, output_shape)
            adaptation_amplitude = np.broadcast_to(adaptation_amplitude, output_shape)
            adaptation_baseline = np.broadcast_to(adaptation_baseline, output_shape)
            adaptation_tau_inv = np.broadcast_to(adaptation_tau_inv, output_shape)
            background_var = np.broadcast_to(background_var, output_shape)
            background_mean = np.broadcast_to(background_mean, output_shape)
            if time_axis<0:
                time_axis+= len(output_shape)
        except:
            t = t[None, :, None]
            slope = slope[:, None, None]
            origin = origin[:, None, None]
            alpha = alpha[:, None, None]
            adaptation_amplitude = adaptation_amplitude[:, None, None]
            adaptation_baseline = adaptation_baseline[:, None, None]
            adaptation_tau_inv = adaptation_tau_inv[:, None, None]
            background_var = background_var[:, None, None]
            background_mean = background_mean[:, None, None]
            output_shape = (slope.size, t.size, nsamples)
        
        return Stimulator._sample(output_shape=output_shape,
                    t=t,
                    slope=slope,
                    origin=origin,
                    alpha=alpha,
                    adaptation_amplitude=adaptation_amplitude,
                    adaptation_baseline=adaptation_baseline,
                    adaptation_tau_inv=adaptation_tau_inv,
                    background_var=background_var,
                    background_mean=background_mean,
                    split_stochastic=split_stochastic,
                    use_gaussian_approx=use_gaussian_approx)
    
    @staticmethod
    def _sample(output_shape, t, slope, origin, alpha, adaptation_amplitude,
                adaptation_baseline, adaptation_tau_inv, background_var,
                background_mean, split_stochastic=True, use_gaussian_approx=False):
        """
        Low level sampling scheme which is called from the instance 'sample'
        method. All inputs must be broadcastable together.
        
        Syntax:
        _sample(output_shape, t, dt, slope, origin, alpha, adaptation_amplitude,
                adaptation_baseline, adaptation_tau_inv, background_var,
                background_mean, split_stochastic=True, use_gaussian_approx=False)
        
        The total stimulus that feeds the leaky integrator is:
        leaky_input(t) = sp(t) * adfun(t)
        
        where:
        sp(t) = (sigma(t) * np.abs(np.random.randn(output_shape)))**alpha
        sigma(t) = slope * t + origin
        adfun(t) = A * (B + (1 - B)*exp(-t*Tau_inv_ad))
        
        Input:
            output_shape: tuple that represents the desired shape of the output array,
                leaky_input(t).
            t: times at which to generate the random stimulus samples.
            slope: slope of the sigma time dependence. sigma(t) = slope * t + origin
            origin: origin of the sigma time dependence. sigma(t) = slope * t + origin
            alpha: exponent of the power law transfer function of sigma(t).
                sp(t) = rand_sample(t)**alpha
            adaptation_amplitude: amplitude, A, of the adaptation function, adfun.
            adaptation_baseline: baseline, B, of the adaptation function, adfun.
            adaptation_tau_inv: inverse time constant, Tau_inv_ad, of the
                adaptation function, adfun.
            split_stochastic: bool. If true, the method returns two arrays,
                the deterministic contribution and the stochastic contribution.
                If false, the sum of these two is returned.
            use_gaussian_approx: bool that indicates whether to approximate
                the stimulus generation process as a time inhomogenous weiner
                process. The default is false, and thus gets samples from
                a truncated half gaussian passed through a power law
                transfer function. If true, then leaky_input(t) is generated as
                mean(t) + var(t) * np.random.randn(output_shape)
                where
                mean(t) = np.sqrt(2**alpha/np.pi)*sigma(t)**alpha * gamma(0.5*(1+alpha))
                var(t) = (2*sigma(t)**2)**alpha/np.pi *
                  (np.sqrt(np.pi)*gamma(alpha+0.5) - gamma(0.5*(1+alpha))**2)
            
        Output:
            leaky_input(t) or (deterministic_stim(t), stochastic_stim(t))
            depending on the value of split_stochastic. If split_stochastic is
            true, then deterministic_stim(t) is equal to mean(t)*adfun(t), as written
            in use_gaussian_approx, and stochastic_stim(t) is the result of
            (sp(t)-mean(t))*adfun(t).
        
        """
        adaptation = adaptation_amplitude * (adaptation_baseline +
                (1 - adaptation_baseline) * np.exp(-t*adaptation_tau_inv))
        
        sigma = (t*slope + origin)
        mean = np.sqrt(2**alpha/np.pi)*sigma**alpha * gamma(0.5*(1+alpha))
        internal_noise = np.sqrt(background_var)*np.random.randn(*output_shape)
        if use_gaussian_approx:
            var = (2*sigma**2)**alpha/np.pi *\
                  (np.sqrt(np.pi)*gamma(alpha+0.5) - gamma(0.5*(1+alpha))**2)
            stim_noise = np.random.randn(*output_shape) * np.sqrt(var)
            if split_stochastic:
                output = (mean * adaptation + background_mean, stim_noise * adaptation + internal_noise)
            else:
                output = (mean + stim_noise) * adaptation + background_mean + internal_noise
        else:
            stim = np.abs(np.random.randn(*output_shape) * sigma)**alpha
            stim_noise = stim - mean
            if split_stochastic:
                output = (mean * adaptation + background_mean, stim_noise * adaptation + internal_noise)
            else:
                output = stim * adaptation + background_mean + internal_noise
        return output
    
    @staticmethod
    def sde_symbols():
        """
        Stimulator.sde_symbols()
        
        Returns sympy symbols asociated with the model's stimulator
        parameters: t, slope, origin, alpha, adaptation_amplitude,
        adaptation_baseline, adaptation_tau_inv, background_var,
        background_mean, mean and var. These last two parameters are
        the expected value and variance of the stochastic input and
        are dependent on all the other parameters.
        
        Output: a dictionary. The keys are the parameter names, t, slope,
            origin, alpha, adaptation_amplitude, adaptation_baseline,
            adaptation_tau_inv, background_var, background_mean, mean
            and var. The corresponding values are sympy symbols with the
            proper attributes defined such as 'real', 'nonnegative' or
            'positive'
        
        """
        x, y, t, slope, adaptation_amplitude, adaptation_baseline, background_mean = \
            sympy.symbols('x y t slope adaptation_amplitude adaptation_baseline background_mean', real=True)
        alpha, sigma, T, adaptation_tau_inv, background_var = \
            sympy.symbols('alpha sigma T adaptation_tau_inv background_var',
                          positive=True, real=True)
        origin = sympy.symbols('origin',nonnegative=True, real=True)
        
        adfun = adaptation_amplitude * (adaptation_baseline + \
                                        (1-adaptation_baseline) * sympy.exp(-t*adaptation_tau_inv))
        
        half_gauss_pdf = 2/sympy.sqrt(2*sympy.pi)/sigma * sympy.exp(-x**2/(2*sigma**2))
        dydx = sympy.diff(x**alpha, x)
        xy = x**(1/alpha)
        pdf = sympy.simplify((half_gauss_pdf/dydx).subs(x, xy))
        
        mean = sympy.simplify(sympy.integrate(x*pdf, (x, 0, sympy.oo))).subs(sigma, slope*t + origin)
        mean*= adfun
        mean+= background_mean
        mean = sympy.simplify(mean)
        var = sympy.simplify(sympy.integrate(x**2*pdf, (x, 0, sympy.oo))-mean**2).subs(sigma, slope*t + origin) 
        var*= (adfun**2)
        var+= background_var
        var = sympy.simplify(var)
        
        return {'t':t, 'slope':slope, 'origin':origin, 'alpha':alpha,
                'adaptation_amplitude':adaptation_amplitude,
                'adaptation_baseline':adaptation_baseline,
                'adaptation_tau_inv':adaptation_tau_inv,
                'background_var':background_var,
                'background_mean':background_mean,
                'mean':mean, 'var':var}

class Leaky(CompatibleParameters):
    """
    Leaky class that is used to generate sample path of the stochastic
    leaky integration and also calculate the theoretically predicted
    mean and variance if the distribution of sample paths.
    It inherits from CompatibleParameters.
    
    """
    def __init__(self, leak=1., C_inv=1., x0=0., var0=0.):
        """
        Leaky(leak=1., C_inv=1., x0=0., var0=0.)
        
        All input parameters of __init__ are passed to the CompatibleParameters
        super class and are hence the considered as the parameters whoes
        shape must always remain compatible and broadcasted.
        
        The leaky integral equation that is represented by this class is:
        
        C_inv * dx = -leak * x + leaky_input(t)
        
        Input:
            leak: leak of the leaky integrator.
            C_inv: constant that multiplies the derivative of the percept.
            x0: start value of the percept.
            var0: initial variance across x0.
        
        """
        super(Leaky, self).__init__(leak=leak, C_inv=C_inv, x0=x0, var0=var0)
    
    def sample(self, dt, T, stimulator=None, nsamples=100, time_unit_conversion=1.,
                       use_gaussian_approx=False, return_t=False, return_stim=False,
                       return_parameters=False, randomize_x0=True, **kwargs):
        """
        self.sample(dt, T, stimulator=None, nsamples=100, time_unit_conversion=1.,
                    use_gaussian_approx=False, return_t=False, return_stim=False,
                    return_parameters=False, randomize_x0=True, **kwargs)
        
        Generate sample path of the stochastic differential equation
        using the Euler Maruyama method.
        
        Input:
            dt: a float that indicates the timestep used in the sampling
                method.
            T: a float that indicates the maximum time value. The array
                of time points, t, will be set as
                numpy.arange(int((T+dt)/dt)).astype(numpy.float) * dt
            stimulator: None or a stimulator instance. If None, a 
                stimulator with default parameters will be used. This
                instance will be used to generate stimulation samples.
            nsamples: an int. The number of independent sample paths to
                generate for each parameter tuple.
            time_unit_conversion: a float that indicates how to convert
                from the assumed time units to milliseconds. This is
                necessary to get the proper variance units.
            use_gaussian_approx: a bool that indicates whether to approximate
                the stimulus generation process as a time inhomogenous weiner
                process. The default is false, and thus gets samples from
                a truncated half gaussian passed through a power law
                transfer function.
            return_t: a bool that indicates whether to return the array
                of time points, t, in which the samples were drawn.
            return_stim: a bool that indicates whether to return the
                sampled stimulation array.
            return_parameters: a bool that indicates whether to return a
                CompatibleParameters instance with the parameter values
                used for the sampling.
            randomize_x0: a bool that indicates whether to randomize the
                initial value of the stochastic variable with a gaussian
                of variance given by var0. Default is true, but it could
                be set to false in cases several calls to sample are
                chained with different parameters to get samples of the
                full process.
            kwargs: Can be any of the Leaky or Stimulator class attributes
                set in their respective __init__ methods. These are:
                leak, C_inv, x0, var0, slope, origin, alpha,
                adaptation_amplitude, adaptation_baseline,
                adaptation_tau_inv, background_var and background_mean.
                If supplied, these values will be used instead of the
                Leaky or Stimulator instance attribute's values.
                WARNING! If one of the above parameters is not present in
                kwargs, the parameters belonging to the Leaky class
                instance will be reshaped as [:, None], while the ones
                belonging to the Stimulator class will be reshaped to
                [None, :]. If the parameter was provided in kwargs, no
                reshaping is done. The final parameter values must be
                broadcastable, as they will be used to construct a
                CompatibleParameters instance.
        
        Output:
            If all the return_* input parameters are False, a single
            numpy.ndarray with the sample paths is returned. The first
            axis corresponds to indepdents parameter values, the
            second axis corresponds to time, and the third axis is for
            the indepdent random samples.
            
            If any of the return_* input parameters is True, the output
            will be a tuple. The first element will always be the sample
            paths. The following elements depend on which of the
            return_* parameters were True.
            
            t: numpy.ndarray, optional
                The array of time points, t, which is equal to
                numpy.arange(int((T+dt)/dt)).astype(numpy.float) * dt
            stim: numpy.ndarray, optional
                Full stimulatus array sampled from the Stimulator instance
            parameters: CompatibleParameters, optional
                The parameter values used. It is constructed as
                CompatibleParameters(
                    slope  =kwargs.pop('slope',  stimulator.slope[None, :]),
                    origin =kwargs.pop('origin', stimulator.origin[None, :]),
                    alpha  =kwargs.pop('alpha',  stimulator.alpha[None, :]),
                    adaptation_amplitude = kwargs.pop('adaptation_amplitude', stimulator.adaptation_amplitude[None, :]),
                    adaptation_baseline  = kwargs.pop('adaptation_baseline',  stimulator.adaptation_baseline[None, :]),
                    adaptation_tau_inv   = kwargs.pop('adaptation_tau_inv',   stimulator.adaptation_tau_inv[None, :]),
                    background_var  = kwargs.pop('background_var',  stimulator.background_var[None, :]),
                    background_mean = kwargs.pop('background_mean', stimulator.background_mean[None, :]),
                    leak  = kwargs.pop('leak',  self.leak[:, None]),
                    C_inv = kwargs.pop('C_inv', self.C_inv[:, None]),
                    x0    = kwargs.pop('x0',    self.x0[:, None]),
                    var0  = kwargs.pop('var0',  self.var0[:, None]),
                )
        
        """
        if stimulator is None:
            stimulator = Stimulator()
        T = float(T)
        dt = float(dt)
        nsamples = int(nsamples)
        t = np.arange(int((T+dt)/dt)).astype(np.float) * dt
        sqdt = np.sqrt(dt*time_unit_conversion)
        
        parameters = CompatibleParameters(slope=kwargs.pop('slope', stimulator.slope[None, :]),
                        origin=kwargs.pop('origin', stimulator.origin[None, :]),
                        alpha=kwargs.pop('alpha', stimulator.alpha[None, :]),
                        adaptation_amplitude = kwargs.pop('adaptation_amplitude', stimulator.adaptation_amplitude[None, :]),
                        adaptation_baseline = kwargs.pop('adaptation_baseline', stimulator.adaptation_baseline[None, :]),
                        adaptation_tau_inv = kwargs.pop('adaptation_tau_inv', stimulator.adaptation_tau_inv[None, :]),
                        background_var = kwargs.pop('background_var', stimulator.background_var[None, :]),
                        background_mean = kwargs.pop('background_mean', stimulator.background_mean[None, :]),
                        leak = kwargs.pop('leak', self.leak[:, None]),
                        C_inv = kwargs.pop('C_inv', self.C_inv[:, None]),
                        x0 = kwargs.pop('x0', self.x0[:, None]),
                        var0 = kwargs.pop('var0', self.var0[:, None]),
                    )
        
        x0 = parameters.x0[:, None]
        var0 = parameters.var0[:, None]
        leak = parameters.leak[:, None]
        C_inv = parameters.C_inv[:, None]
        
        
        deterministic_stimulus, stochastic_stimulus = \
                stimulator.sample(t[1:], nsamples, split_stochastic=True,
                        use_gaussian_approx=use_gaussian_approx, **parameters.get_parameters())
        
        output_shape = list(stochastic_stimulus.shape)
        output_shape[1]+=1
        x = np.zeros(tuple(output_shape))
        
        x[:, 0] = x0
        
        if randomize_x0:
            x[:, 0]+= np.random.randn(*(x[:, 0].shape)) * np.sqrt(var0)
        for index in range(len(t)-1):
            x[:, index + 1] = x[:, index] + C_inv *\
                           ((-leak * x[:, index] +\
                             deterministic_stimulus[:, index]) * dt +\
                            stochastic_stimulus[:, index] * sqdt)
        output = (x,)
        if return_t:
            output+= (t,)
        if return_stim:
            output+= (deterministic_stimulus + stochastic_stimulus,)
        if return_parameters:
            output+= (parameters,)
        if len(output)==1:
            output = output[0]
        return output
    
    @staticmethod
    def _sample(output_shape, t, x0, var0, slope, origin, leak,
                C_inv, alpha, adaptation_amplitude, adaptation_baseline,
                adaptation_tau_inv, background_var, background_mean,
                time_axis, time_unit_conversion=1.,
                randomize_x0=True, use_gaussian_approx=False, return_stim=False):
        """
        Leaky._sample(output_shape, t, x0, var0, slope, origin, leak,
                C_inv, alpha, adaptation_amplitude, adaptation_baseline,
                adaptation_tau_inv, background_var, background_mean, dt,
                time_axis, time_unit_conversion=1.,
                randomize_x0=True, use_gaussian_approx=False, return_stim=False)
        
        Low level static method to generate sample paths with the Euler
        Maruyama numerical method. Like a Leaky instance's sample method
        doing very few shape checks, and not invoquing a CompatibleParameters
        instance.
        
        Input:
            output_shape: desired shape of the output array of sample paths.
                All the following parameters must be broadcastable to this
                shape.
            t: numpy.ndarray of time points in which to get samples.
            x0: numpy.ndarray start value of the percept.
            var0: numpy.ndarray initial variance across x0.
            slope: numpy.ndarray slope of the sigma time dependence. sigma(t) = slope * t + origin
            origin: numpy.ndarray origin of the sigma time dependence. sigma(t) = slope * t + origin
            leak: numpy.ndarray leak of the leaky integrator.
            C_inv: numpy.ndarray constant that multiplies the derivative of the percept.
            alpha: numpy.ndarray exponent of the power law transfer function of sigma(t).
                sp(t) = sigma(t)**alpha
            adaptation_amplitude: numpy.ndarray amplitude, A, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_baseline: numpy.ndarray baseline, B, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_tau_inv: numpy.ndarray inverse time constant, Tau_inv_ad, of the
                adaptation function that multiplies the total raw input. This
                function is A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            background_var: numpy.ndarray background input's variance.
            background_mean: numpy.ndarray background input's mean value.
            time_axis: axis of output_shape in which t is changing
            time_unit_conversion: a float that indicates how to convert
                from the assumed time units to milliseconds. This is
                necessary to get the proper variance units.
            randomize_x0: a bool that indicates whether to randomize the
                initial value of the stochastic variable with a gaussian
                of variance given by var0. Default is true, but it could
                be set to false in cases several calls to sample are
                chained with different parameters to get samples of the
                full process.
            use_gaussian_approx: a bool that indicates whether to approximate
                the stimulus generation process as a time inhomogenous weiner
                process. The default is false, and thus gets samples from
                a truncated half gaussian passed through a power law
                transfer function.
            return_stim: a bool that indicates whether to return the
                sampled stimulation array.
        
        Output:
            x: numpy.ndarray with the generated sample paths.
            stim: numpy.ndarray, optional
                Full stimulatus array sampled from the Stimulator instance
        
        """
        x = np.empty(output_shape)
        
        det, sto = Stimulator._sample(output_shape, t, slope, origin, alpha, adaptation_amplitude,
                                    adaptation_baseline, adaptation_tau_inv, background_var, background_mean,
                                    use_gaussian_approx=use_gaussian_approx)
        
        ind0 = [slice(None) if axis!=time_axis else 0 for axis in range(len(output_shape))]
        ind1 = [slice(None) if axis!=time_axis else 1 for axis in range(len(output_shape))]
        x[tuple(ind0)] = x0
        if randomize_x0 and (np.any(var0>0) if isinstance(var0, np.ndarray) else var0>0):
            x[tuple(ind0)]+= np.random.randn(*(x[tuple(ind0)].shape)) * np.sqrt(var0)
        
        dt = np.diff(t, axis=time_axis)
        sqdt = np.sqrt(time_unit_conversion*dt)
        
        if isinstance(C_inv, np.ndarray):
            if C_inv.shape[time_axis]>=x.shape[time_axis]:
                C_t = C_inv
            else:
                C_t = C_inv[ind0]
        else:
            C_t = C_inv
        if isinstance(leak, np.ndarray):
            if leak.shape[time_axis]>=x.shape[time_axis]:
                leak_t = leak
            else:
                leak_t = leak[ind0]
        else:
            leak_t = leak
        for t_ind in range(x.shape[time_axis]-1):
            x[tuple(ind1)] = x[tuple(ind0)] + C_t * \
                               ((-leak_t * x[tuple(ind0)] + det[tuple(ind0)]) * \
                               dt[tuple(ind0)] + \
                             sto[tuple(ind0)] * sqdt[tuple(ind0)])
            ind0[time_axis]+=1
            ind1[time_axis]+=1
        output = (x,)
        if return_stim:
            output+= (det + sto,)
        if len(output)==1:
            output = output[0]
        return output
    
    @staticmethod
    def sde_symbols():
        """
        Leaky.sde_symbols()
        
        Returns sympy symbols asociated with the leaky integration model's
        parameters: t, x0, var0, leak and C_inv.
        
        Output: a dictionary. The keys are the parameter names, t, x0,
            var0, leak and C_inv. The corresponding values are sympy
            symbols with the proper attributes defined such as 'real',
            'nonnegative' or 'positive'
        
        """
        t, x0 = sympy.symbols('t x0', real=True)
        C_inv = sympy.symbols('C_inv',positive=True, real=True)
        leak, var0 = sympy.symbols('leak var0',nonnegative=True, real=True)
        return {'t':t, 'x0':x0, 'var0':var0, 'leak':leak, 'C_inv':C_inv}
    
    def theoretical_symbolic(self, stimulator=None, symbols={}, return_symbolic=True,
                    set_defaults=False, replace_scalars=False,
                    return_internal_symbols=False):
        """
        WARNING, implementation is not finished nor working properly.
        The intention is to work with sympy to get the analytical expressions
        of the stochastic integrals and then return the theoretical
        mean and variance of the probability density function asociated
        to the stochastic dynamical variable.
        
        """
        if stimulator is None:
            stimulator = Stimulator()
        s = Stimulator.sde_symbols()
        s.update(Leaky.sde_symbols())
        s.update(symbols)
        
        adaptation = s['adaptation_amplitude'] * (s['adaptation_baseline'] +
                (1 - s['adaptation_baseline']) * sympy.exp(-s['t']*s['adaptation_tau_inv']))
        
        """
        The following is the sympy implementation to programmatically
        get the symbolic representations of the process' mean and variance.
        In the tested version sympy-1.1, sympy was unable to solve the
        integrals and left them hanging so it is was necessary to solve
        them by hand. Hopefully in future versions, sympy will be able
        to solve the integrals by itself.
        
        x = sympy.symbols('x', real=True)
        phi = sympy.exp(-s['leak'] * s['C_inv'] * s['t'])
        at  = sympy.simplify(s['mean'] * s['C_inv'] * adaptation)
        ft  = sympy.simplify(s['var'] * s['C_inv']**2 * adaptation**2)
        
        mean = phi* (s['x0']   + sympy.integrate(
                        sympy.expand(sympy.simplify((at/phi).subs(s['t'],x))),
                        (x, 0, s['t']))
                    )
        var  = phi* (s['x0']   + sympy.integrate(
                        sympy.expand(sympy.simplify((ft/phi**2).subs(s['t'],x))),
                        (x, 0, s['t']))
                    )
        mean = sympy.simplify(mean)
        var  = sympy.simplify(var)
        """
        # Done by hand because sympy cannot resolve the integrals by itself!
        
        A1 = s['adaptation_amplitude']*s['adaptation_baseline']
        A2 = s['adaptation_amplitude']*(1-s['adaptation_baseline'])
        k1 = s['leak']*s['C_inv']
        k2 = s['leak']*s['C_inv'] - s['adaptation_tau_inv']
        k3 = k1+k2
        
        integral = A1 * sympy.Piecewise(
                    (s['origin']**s['alpha'] * s['t'], sympy.And(s['slope']==0, k1==0)),
                    (s['origin']**s['alpha']/k1 * (sympy.exp(k1*s['t'])-1), sympy.And(s['slope']==0, k1!=0)),
                    (((s['slope']*s['t'] + s['origin'])**(s['alpha']+1) - \
                       s['origin']**(s['alpha']+1))/\
                     (s['slope']*(s['alpha']+1)), sympy.And(s['slope']!=0,k1==0)),
                    ((-s['slope']/k1)**(s['alpha'])/k1 * sympy.exp(-k1*s['origin']/s['slope']) * \
                     (sympy.uppergamma(s['alpha']+1,-k1*(s['t']+s['origin']/s['slope'])) - \
                     sympy.uppergamma(s['alpha']+1,-k1/s['slope']*s['origin'])), True)
                   ) + \
                   A2 * sympy.Piecewise(
                    (s['origin']**s['alpha'] * s['t'], sympy.And(s['slope']==0, k1==0)),
                    (s['origin']**s['alpha']/k2 * (sympy.exp(k2*s['t'])-1), sympy.And(s['slope']==0, k2!=0)),
                    (((s['slope']*s['t'] + s['origin'])**(s['alpha']+1) - \
                       s['origin']**(s['alpha']+1))/\
                     (s['slope']*(s['alpha']+1)), sympy.And(s['slope']!=0,k2==0)),
                    ((-s['slope']/k2)**(s['alpha'])/k2 * sympy.exp(-k2*s['origin']/s['slope']) * \
                     (sympy.uppergamma(s['alpha']+1,-k2*(s['t']+s['origin']/s['slope'])) - \
                     sympy.uppergamma(s['alpha']+1,-k2/s['slope']*s['origin'])), True)
                   )
        integral2 = A1**2 * sympy.Piecewise(
                    (s['origin']**(2*s['alpha']) * s['t'], sympy.And(s['slope']==0, k1==0)),
                    (s['origin']**(2*s['alpha'])/2/k1 * (sympy.exp(2*k1*s['t'])-1), sympy.And(s['slope']==0, k1!=0)),
                    (((s['slope']*s['t'] + s['origin'])**(2*s['alpha']+1) - \
                       s['origin']**(2*s['alpha']+1))/\
                     (s['slope']*(2*s['alpha']+1)), sympy.And(s['slope']!=0,k1==0)),
                    ((-s['slope']/2/k1)**(2*s['alpha'])/(2*k1) * sympy.exp(-2*k1*s['origin']/s['slope']) * \
                     (sympy.uppergamma(2*s['alpha']+1,-2*k1*(s['t']+s['origin']/s['slope'])) - \
                     sympy.uppergamma(2*s['alpha']+1,-2*k1/s['slope']*s['origin'])), True)
                   ) + \
                   A2**2 * sympy.Piecewise(
                    (s['origin']**s['alpha'] * s['t'], sympy.And(s['slope']==0, k1==0)),
                    (s['origin']**s['alpha']/k2 * (sympy.exp(k2*s['t'])-1), sympy.And(s['slope']==0, k2!=0)),
                    (((s['slope']*s['t'] + s['origin'])**(s['alpha']+1) - \
                       s['origin']**(s['alpha']+1))/\
                     (s['slope']*(s['alpha']+1)), sympy.And(s['slope']!=0,k2==0)),
                    ((-s['slope']/2/k2)**(2*s['alpha'])/(2*k2) * sympy.exp(-2*k2*s['origin']/s['slope']) * \
                     (sympy.uppergamma(2*s['alpha']+1,-2*k2*(s['t']+s['origin']/s['slope'])) - \
                     sympy.uppergamma(2*s['alpha']+1,-2*k2/s['slope']*s['origin'])), True)
                   ) + \
                   2 * A1 * A2 * sympy.Piecewise(
                    (s['origin']**(2*s['alpha']) * s['t'], sympy.And(s['slope']==0, k3==0)),
                    (s['origin']**(2*s['alpha'])/k3 * (sympy.exp(k3*s['t'])-1), sympy.And(s['slope']==0, k3!=0)),
                    (((s['slope']*s['t'] + s['origin'])**(2*s['alpha']+1) - \
                       s['origin']**(2*s['alpha']+1))/\
                     (s['slope']*(2*s['alpha']+1)), sympy.And(s['slope']!=0,k3==0)),
                    ((-s['slope']/k3)**(2*s['alpha'])/(k3) * sympy.exp(-k3*s['origin']/s['slope']) * \
                     (sympy.uppergamma(2*s['alpha']+1,-k3*(s['t']+s['origin']/s['slope'])) - \
                     sympy.uppergamma(2*s['alpha']+1,-k3/s['slope']*s['origin'])), True)
                   )
        
        mean = sympy.simplify(sympy.exp(-s['leak']*s['t']*s['C_inv']) * 
                (s['x0'] + sympy.sqrt(2**s['alpha']/sympy.pi)*s['C_inv'] *
                 sympy.gamma((1+s['alpha'])/2) * integral))
        var  = sympy.simplify(sympy.exp(-2*s['leak']*s['t']*s['C_inv']) * 
                (s['var0'] + 2**s['alpha']/sympy.pi*s['C_inv']**2 *
                (sympy.sqrt(sympy.pi)*sympy.gamma(1/2+s['alpha']) -
                 sympy.gamma((1+s['alpha'])/2)**2) *
                integral2))
        
        independent_parameters = {name: symbol for name, symbol in iteritems(s) if name not in ('sigma', 'mean', 'var')}
        
        if set_defaults or replace_scalars:
            stimulator_parameters = {k: v[:, None] for k, v in iteritems(stimulator.get_parameters())}
            model_parameters = {k: v[None, :] for k, v in iteritems(self.get_parameters())}
            stimulator_parameters.update(model_parameters)
            defaults = compatibalize_shapes(**stimulator_parameters)
            defaults = {k: v[:, None] if len(np.unique(v))>1 else np.unique(v)[0] for k, v in iteritems(defaults)}
            scalar_defaults = {k: v for k, v in iteritems(defaults) if isinstance(v, float)}
        
        if replace_scalars:
            for symbol, value in iteritems(scalar_defaults):
                if np.isinf(value):
                    mean = sympy.limit(mean, s[symbol], np.sign(value)*sympy.oo)
                    var = sympy.limit(var, s[symbol], np.sign(value)*sympy.oo)
                else:
                    mean = mean.subs(s[symbol], value)
                    var = var.subs(s[symbol], value)
                independent_parameters.pop(symbol)
            mean = sympy.simplify(mean)
            var = sympy.simplify(var)
        
        if return_symbolic:
            if return_internal_symbols:
                return (mean, var, independent_parameters)
            else:
                return (mean, var)
        else:
            modules = [{'gamma': gamma, 'uppergamma':uppergamma, 'lowergamma': lowergamma}, 'numpy']
            mean_call = sympy.lambdify(
                args = (independent_parameters.values()),
                expr = mean, modules = modules, dummify = False)
            var_call = sympy.lambdify(
                args = (independent_parameters.values()),
                expr = var, modules = modules, dummify = False)
            if set_defaults:
                mean_call = WrapperKwargs(mean_call, **{k: v for k, v in iteritems(defaults) if k in independent_parameters})
                std_call = WrapperKwargs(mean_call, **{k: v for k, v in iteritems(defaults) if k in independent_parameters})
            return (mean_call, var_call)

    def theoretical(self, t, time_unit_conversion=1., stimulator=None, return_parameters=False, **kwargs):
        """
        self.theoretical(t, time_unit_conversion=1., stimulator=None, return_parameters=False, **kwargs)
        
        Compute the theoretical expected value and variance of the
        stochastic dynamical variable in the time points t.
        
        Input:
            t: numpy.ndarray of time points in which to get samples.
            time_unit_conversion: a float that indicates how to convert
                from the assumed time units to milliseconds. This is
                necessary to get the proper variance units.
            stimulator: None or a stimulator instance. If None, a 
                stimulator with default parameters will be used. This
                instance will be used to generate stimulation samples.
            return_parameters: a bool that indicates whether to return a
                CompatibleParameters instance with the parameter values
                used for the sampling.
            kwargs: Can be any of the Leaky or Stimulator class attributes
                set in their respective __init__ methods. These are:
                leak, C_inv, x0, var0, slope, origin, alpha,
                adaptation_amplitude, adaptation_baseline,
                adaptation_tau_inv, background_var and background_mean.
                If supplied, these values will be used instead of the
                Leaky or Stimulator instance attribute's values.
                WARNING! If one of the above parameters is not present in
                kwargs, the parameters belonging to the Leaky class
                instance will be reshaped as [:, None], while the ones
                belonging to the Stimulator class will be reshaped to
                [None, :]. If the parameter was provided in kwargs, no
                reshaping is done. The final parameter values must be
                broadcastable, as they will be used to construct a
                CompatibleParameters instance.
        
        Output:
            mean: numpy.ndarray
                The theoretical expected value of the dynamical variable
                at the time points t and parameter values given in
                parameters.as_array().
            var: numpy.ndarray
                The theoretical variance of the dynamical variable
                at the time points t and parameter values given in
                parameters.as_array().
            parameters: CompatibleParameters, optional
                The parameter values used. It is constructed as
                CompatibleParameters(
                    slope  = kwargs.pop('slope',  stimulator.slope[None, :]),
                    origin = kwargs.pop('origin', stimulator.origin[None, :]),
                    alpha  = kwargs.pop('alpha',  stimulator.alpha[None, :]),
                    adaptation_amplitude = kwargs.pop('adaptation_amplitude', stimulator.adaptation_amplitude[None, :]),
                    adaptation_baseline  = kwargs.pop('adaptation_baseline',  stimulator.adaptation_baseline[None, :]),
                    adaptation_tau_inv   = kwargs.pop('adaptation_tau_inv',   stimulator.adaptation_tau_inv[None, :]),
                    background_var  = kwargs.pop('background_var',  stimulator.background_var[None, :]),
                    background_mean = kwargs.pop('background_mean', stimulator.background_mean[None, :]),
                    leak  = kwargs.pop('leak',  self.leak[:, None]),
                    C_inv = kwargs.pop('C_inv', self.C_inv[:, None]),
                    x0    = kwargs.pop('x0',    self.x0[:, None]),
                    var0  = kwargs.pop('var0',  self.var0[:, None]),
                    flatten=False,
                )
                WARNING: flatten can be called on parameters if the
                supplied t array does not broadcast with the parameter
                arrays in this CompatibleParameters instance.
        
        """
        if stimulator is None:
            stimulator = Stimulator()
        parameters = CompatibleParameters(
                        slope  = kwargs.pop('slope',  stimulator.slope[None, :]),
                        origin = kwargs.pop('origin', stimulator.origin[None, :]),
                        alpha  = kwargs.pop('alpha',  stimulator.alpha[None, :]),
                        adaptation_amplitude = kwargs.pop('adaptation_amplitude', stimulator.adaptation_amplitude[None, :]),
                        adaptation_baseline  = kwargs.pop('adaptation_baseline',  stimulator.adaptation_baseline[None, :]),
                        adaptation_tau_inv   = kwargs.pop('adaptation_tau_inv',   stimulator.adaptation_tau_inv[None, :]),
                        background_var  = kwargs.pop('background_var',  stimulator.background_var[None, :]),
                        background_mean = kwargs.pop('background_mean', stimulator.background_mean[None, :]),
                        leak  = kwargs.pop('leak',  self.leak[:, None]),
                        C_inv = kwargs.pop('C_inv', self.C_inv[:, None]),
                        x0    = kwargs.pop('x0',    self.x0[:, None]),
                        var0  = kwargs.pop('var0',  self.var0[:, None]),
                        flatten=False,
                    )
        slope = parameters.slope
        origin = parameters.origin
        alpha = parameters.alpha
        adaptation_amplitude = parameters.adaptation_amplitude
        adaptation_baseline = parameters.adaptation_baseline
        adaptation_tau_inv = parameters.adaptation_tau_inv
        background_var = parameters.background_var
        background_mean = parameters.background_mean
        leak = parameters.leak
        C_inv = parameters.C_inv
        x0 = parameters.x0
        var0 = parameters.var0
        if t.ndim==1 and len(t)!=parameters.slope.shape[-1]:
            # If t is a 1D vector that does not broadcast with the compatibalized shaped
            # parameters, then flatten the parameters and add a new_axis
            # at the end to enable broadcasting with t
            parameters.flatten = True
            parameters.compatibalize_parameters()
            
            slope = slope.flatten()[...,None]
            origin = origin.flatten()[...,None]
            alpha = alpha.flatten()[...,None]
            adaptation_amplitude = adaptation_amplitude.flatten()[...,None]
            adaptation_baseline = adaptation_baseline.flatten()[...,None]
            adaptation_tau_inv = adaptation_tau_inv.flatten()[...,None]
            background_var = background_var.flatten()[...,None]
            background_mean = background_mean.flatten()[...,None]
            leak = leak.flatten()[...,None]
            C_inv = C_inv.flatten()[...,None]
            x0 = x0.flatten()[...,None]
            var0 = var0.flatten()[...,None]
        if _c_api_available or np.any(parameters.slope!=0):
            output = Leaky._theoretical_generalstim(t=t,
                                            x0=x0,
                                            var0=var0,
                                            leak=leak,
                                            C_inv=C_inv,
                                            origin=origin,
                                            slope=slope,
                                            alpha=alpha,
                                            adaptation_amplitude=adaptation_amplitude,
                                            adaptation_baseline=adaptation_baseline,
                                            adaptation_tau_inv=adaptation_tau_inv,
                                            background_var=background_var,
                                            background_mean=background_mean,
                                            time_unit_conversion=time_unit_conversion)
        else:
            output = Leaky._theoretical_flatstim(t=t,
                                            x0=x0,
                                            var0=var0,
                                            leak=leak,
                                            C_inv=C_inv,
                                            origin=origin,
                                            alpha=alpha,
                                            adaptation_amplitude=adaptation_amplitude,
                                            adaptation_baseline=adaptation_baseline,
                                            adaptation_tau_inv=adaptation_tau_inv,
                                            background_var=background_var,
                                            background_mean=background_mean,
                                            time_unit_conversion=time_unit_conversion)
        if return_parameters:
            output+= (parameters,)
        return output
    
    @staticmethod
    def _theoretical_generalstim(t, x0, var0, leak, C_inv, origin, slope, alpha,
                    adaptation_amplitude, adaptation_baseline, adaptation_tau_inv,
                    background_var, background_mean, time_unit_conversion=1.):
        """
        Leaky._theoretical_generalstim(t, x0, var0, leak, C_inv, origin, slope, alpha,
                    adaptation_amplitude, adaptation_baseline, adaptation_tau_inv,
                    background_var, background_mean, time_unit_conversion=1.)
        
        Compute the theoretical expected value and variance of the
        stochastic dynamical variable in the time points t for a general
        stimulation profile.
        
        Input:
            t: numpy.ndarray of time points in which to get samples.
            x0: numpy.ndarray start value of the percept.
            var0: numpy.ndarray initial variance across x0.
            leak: numpy.ndarray leak of the leaky integrator.
            C_inv: numpy.ndarray constant that multiplies the derivative of the percept.
            origin: numpy.ndarray origin of the sigma time dependence. sigma(t) = slope * t + origin
            slope: numpy.ndarray slope of the sigma time dependence. sigma(t) = slope * t + origin
            alpha: numpy.ndarray exponent of the power law transfer function of sigma(t).
                sp(t) = sigma(t)**alpha
            adaptation_amplitude: numpy.ndarray amplitude, A, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_baseline: numpy.ndarray baseline, B, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_tau_inv: numpy.ndarray inverse time constant, Tau_inv_ad, of the
                adaptation function that multiplies the total raw input. This
                function is A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            background_var: numpy.ndarray background input's variance.
            background_mean: numpy.ndarray background input's mean value.
            time_unit_conversion: a float that indicates how to convert
                from the assumed time units to milliseconds. This is
                necessary to get the proper variance units.
        
        Output:
            mean: numpy.ndarray
                The theoretical expected value of the dynamical variable
                at the time points t and parameter values. Its shape is
                equal to the broadcast shape of all the array input
                provided to this function.
            var: numpy.ndarray
                The theoretical variance of the dynamical variable
                at the time points t and parameter values. Its shape is
                equal to the broadcast shape of all the array input
                provided to this function.
        
        """
        tau_inv = leak*C_inv
        
        c1 = 2**alpha*C_inv**2/np.pi
        c2 = gamma(0.5*(1+alpha))
        c3 = np.sqrt(np.pi)*gamma(0.5+alpha) - c2**2
        
        integral_mean, integral_var = Leaky._integral_calculator(t,
                                                                slope,
                                                                origin,
                                                                alpha,
                                                                tau_inv,
                                                                adaptation_amplitude,
                                                                adaptation_baseline,
                                                                adaptation_tau_inv)
        internal_det = background_mean * np.where(tau_inv==0, t * np.exp(-tau_inv*t), (1-np.exp(-tau_inv*t))/tau_inv)
        internal_stoch = background_var * np.where(tau_inv==0, t * np.exp(-2*tau_inv*t), 0.5*(1-np.exp(-2*tau_inv*t))/tau_inv)
        mean = x0 * np.exp(-t*tau_inv) + np.sqrt(c1) * c2 * integral_mean +\
               C_inv * internal_det
        var = var0 * np.exp(-2*t*tau_inv) + c1 * c3 * time_unit_conversion * integral_var +\
              time_unit_conversion * C_inv**2 * internal_stoch
        return mean, var
    
    @staticmethod
    def _theoretical_flatstim(t, x0, var0, leak, C_inv, origin, alpha,
                    adaptation_amplitude, adaptation_baseline, adaptation_tau_inv,
                    background_var, background_mean, time_unit_conversion=1.):
        """
        Leaky._theoretical_flatstim(t, x0, var0, leak, C_inv, origin, alpha,
                    adaptation_amplitude, adaptation_baseline, adaptation_tau_inv,
                    background_var, background_mean, time_unit_conversion=1.)
        
        Compute the theoretical expected value and variance of the
        stochastic dynamical variable in the time points t for a flat
        stimulation profile.
        
        Input:
            t: numpy.ndarray of time points in which to get samples.
            x0: numpy.ndarray start value of the percept.
            var0: numpy.ndarray initial variance across x0.
            leak: numpy.ndarray leak of the leaky integrator.
            C_inv: numpy.ndarray constant that multiplies the derivative of the percept.
            origin: numpy.ndarray constant value of the stimulation sigma.
            alpha: numpy.ndarray exponent of the power law transfer function of sigma(t).
                sp(t) = sigma(t)**alpha
            adaptation_amplitude: numpy.ndarray amplitude, A, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_baseline: numpy.ndarray baseline, B, of the adaptation function that
                multiplies the total raw input sp(t). This function is
                A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            adaptation_tau_inv: numpy.ndarray inverse time constant, Tau_inv_ad, of the
                adaptation function that multiplies the total raw input. This
                function is A * (B + (1 - B)*exp(-t*Tau_inv_ad)).
            background_var: numpy.ndarray background input's variance.
            background_mean: numpy.ndarray background input's mean value.
            time_unit_conversion: a float that indicates how to convert
                from the assumed time units to milliseconds. This is
                necessary to get the proper variance units.
        
        Output:
            mean: numpy.ndarray
                The theoretical expected value of the dynamical variable
                at the time points t and parameter values. Its shape is
                equal to the broadcast shape of all the array input
                provided to this function.
            var: numpy.ndarray
                The theoretical variance of the dynamical variable
                at the time points t and parameter values. Its shape is
                equal to the broadcast shape of all the array input
                provided to this function.
        
        """
        c1 = 2**alpha*C_inv**2/np.pi
        c2 = gamma(0.5*(1+alpha))
        c3 = np.sqrt(np.pi)*gamma(0.5+alpha) - c2**2
        
        tau = leak * C_inv
        A1 = adaptation_amplitude * adaptation_baseline
        A2 = adaptation_amplitude * (1-adaptation_baseline)
        A3 = 2 * A1 * A2
        k1 = tau
        k2 = tau - adaptation_tau_inv
        k3 = 2 * tau - adaptation_tau_inv
        integral_mean = np.where(k1!=0,
                                 A1 * np.abs(origin)**alpha/k1 * (np.exp((k1-tau)*t)-np.exp(-tau*t)),
                                 A1 * np.abs(origin)**alpha * t * np.exp(-tau*t))
        integral_mean+= np.where(k2!=0,
                                 A2 * np.abs(origin)**alpha/k2 * (np.exp((k2-tau)*t)-np.exp(-tau*t)),
                                 A2 * np.abs(origin)**alpha * t * np.exp(-tau*t))
        integral_var  = np.where(k1!=0,
                                 A1**2 * 0.5*(origin**2)**alpha/k1 * (np.exp(2*(k1-tau)*t)-np.exp(-2*tau*t)),
                                 A1**2 * (origin**2)**alpha * t * np.exp(-2*tau*t))
        integral_var += np.where(k2!=0,
                                 A2**2 * 0.5*(origin**2)**alpha/k2 * (np.exp(2*(k2-tau)*t)-np.exp(-2*tau*t)),
                                 A2**2 * (origin**2)**alpha * t * np.exp(-2*tau*t))
        integral_var += np.where(k3!=0,
                                 A3 * (origin**2)**alpha/k3 * (np.exp((k3-2*tau)*t)-np.exp(-2*tau*t)),
                                 A3 * (origin**2)**alpha * t * np.exp(-2*tau*t))
        
        mean = x0 * np.exp(-t*tau) + np.sqrt(c1) * c2 * integral_mean + \
               background_mean*C_inv * np.where(tau!=0, 1./tau * (1-np.exp(-tau*t)), t * np.exp(-tau*t))
        var = var0 * np.exp(-2*t*tau) + time_unit_conversion * (c1 * c3 * integral_var + \
              background_var*C_inv**2 * np.where(tau!=0, 0.5/tau * (1-np.exp(-2*tau*t)), t * np.exp(-2*tau*t)))
        return mean, var
    
    @staticmethod
    def _integral_calculator_py(T, slope, origin, alpha, Tau_inv,
                                adaptation_amplitude, adaptation_baseline,
                                adaptation_tau_inv):
        """
        Syntax:
        
        _integral_calculator_py(T, slope, origin, alpha, Tau_inv, 
            adaptation_amplitude, adaptation_baseline, adaptation_tau_inv)
        
        Calculate the stochastic integrals necessary to compute the 
        theoretical mean and variance from the stochastic process
        
        Input:
            T: time at which the to compute the integrals
            slope: slope of the sigma time dependence. sigma(t) = slope * T + origin.
            origin: origin of the sigma time dependence. sigma(t) = slope * T + origin.
            alpha: power law transformation of raw sigma to sigma(t)**alpha.
            Tau_inv: inverse of the time constant, tau, of the leaky integration process.
            adaptation_amplitude: amplitude, A, of the adaptation function that
                multiplies the total raw input. This function is
                A * (B + (1 - B)*exp(-T*Tau_inv_ad)).
            adaptation_baseline: baseline, B, of the adaptation function that
                multiplies the total raw input. This function is
                A * (B + (1 - B)*exp(-T*Tau_inv_ad)).
            adaptation_tau_inv: inverse time constant, Tau_inv_ad, of the
                adaptation function that multiplies the total raw input. This
                function is A * (B + (1 - B)*exp(-T*Tau_inv_ad)).
        
        Output:
            integral_mean: The mean of the stochastic integral.
            integral_var: The variance of the stochastic integral.
        
        """
        it = np.nditer([T, slope, origin, alpha, Tau_inv,
                        adaptation_amplitude, adaptation_baseline,
                        adaptation_tau_inv, None, None])
        for t, m, b, a, tau_inv, A, B, ad_tau_inv, integral_mean, integral_var in it:
            A1 = A * B
            A2 = A * (1-B)
            A3 = 2 * A1 * A2
            k1 = tau_inv
            k2 = tau_inv - ad_tau_inv
            k3 = 2*tau_inv - ad_tau_inv
            
            temp1 = 0.
            temp2 = 0.
            if A1!=0:
                if m==0:
                    if k1==0:
                        # slope==0 and k1==0
                        temp1+= A1 * np.abs(b)**a * t * np.exp(-tau_inv*t)
                        temp2+= A1**2 * (b**2)**a * t * np.exp(-2*tau_inv*t)
                    else:
                        # slope==0 and k1!=0
                        temp1+= A1 * np.abs(b)**a/k1 * (np.exp((k1-tau_inv)*t)-np.exp(-tau_inv*t))
                        temp2+= A1**2 * 0.5*(b**2)**a/k1 * (np.exp(2*(k1-tau_inv)*t)-np.exp(-2*tau_inv*t))
                else:
                    if k1==0:
                        # slope!=0 and k1==0
                        temp1+= A1/(a+1)/m * (np.abs(m*t+b)**(a+1) - np.abs(b)**(a+1)) * np.exp(-tau_inv*t)
                        temp2+= A1**2/(2*a+1)/m * (np.abs(m*t+b)**(2*a+1) - np.abs(b)**(2*a+1)) * np.exp(-2*tau_inv*t)
                    else:
                        # slope!=0 and k1!=0
                        temp1+= A1*np.abs(np.complex(-m/k1)**(a)/k1 * \
                                np.exp(-k1*b/m) * (uppergamma(a+1,-k1*b/m) - \
                                 uppergamma(a+1,-k1*(t+b/m))) * np.exp(-tau_inv*t)).astype(np.float)  
                        temp2+= A1**2*np.abs(np.complex(0.5*m/k1)**(2*a+1)/m * \
                                np.exp(-2*k1*b/m) * (uppergamma(2*a+1,-2*k1*(t+b/m)) - \
                                 uppergamma(2*a+1,-2*k1*b/m)) * np.exp(-2*tau_inv*t)).astype(np.float)
            if A2!=0:
                if m==0:
                    if k2==0:
                        # slope==0 and k2==0
                        temp1+= A2 * b**a * t * np.exp(-tau_inv*t)
                        temp2+= A2**2 * b**(2*a) * t * np.exp(-2*tau_inv*t)
                    else:
                        # slope==0 and k2!=0
                        temp1+= A2 * b**a/k2 * (np.exp((k2-tau_inv)*t)-np.exp(-tau_inv*t))
                        temp2+= A2**2 * 0.5*b**(2*a)/k2 * (np.exp(2*(k2-tau_inv)*t)-np.exp(-2*tau_inv*t))
                else:
                    if k2==0:
                        # slope!=0 and k2==0
                        temp1+= A2/(a+1)/m * ((m*t+b)**(a+1) - b**(a+1)) * np.exp(-tau_inv*t)
                        temp2+= A2**2/(2*a+1)/m * ((m*t+b)**(2*a+1) - b**(2*a+1)) * np.exp(-2*tau_inv*t)
                    else:
                        # slope!=0 and k2!=0
                        temp1+= A2*np.abs(np.complex(-m/k2)**(a+1)/m * \
                                np.exp(-k2*b/m) * (uppergamma(a+1,-k2*b/m) - \
                                 uppergamma(a+1,-k2*(t+b/m))) * np.exp(-tau_inv*t)).astype(np.float)  
                        temp2+= A2**2*np.abs(np.complex(0.5*m/k2)**(2*a+1)/m * \
                                np.exp(-2*k2*b/m) * (uppergamma(2*a+1,-2*k2*(t+b/m)) - \
                                 uppergamma(2*a+1,-2*k2*b/m)) * np.exp(-2*tau_inv*t)).astype(np.float)
            if A3!=0:
                if m==0:
                    if k3==0:
                        # slope==0 and k3==0
                        temp2+= A3 * b**(2*a) * t * np.exp(-2*tau_inv*t)
                    else:
                        # slope==0 and k3!=0
                        temp2+= A3 * b**(2*a)/k3 * (np.exp((k3-2*tau_inv)*t)-np.exp(-2*tau_inv*t))
                else:
                    if k3==0:
                        # slope!=0 and k3==0
                        temp2+= A3/(2*a+1)/m * ((m*t+b)**(2*a+1) - b**(2*a+1)) * np.exp(-2*tau_inv*t)
                    else:
                        # slope!=0 and k3!=0
                        temp2+= A3*np.abs(np.complex(m/k3)**(2*a)/k3 * \
                                np.exp(-k3*b/m) * (uppergamma(2*a+1,-k3*(t+b/m)) - \
                                 uppergamma(2*a+1,-k3*b/m)) * np.exp(-2*tau_inv*t)).astype(np.float)
            
            integral_mean[...] = temp1
            integral_var[...] = temp2
        return it.operands[8:]
    
    if _c_api_available:
        def _integral_calculator(t, slope, origin, alpha, tau_inv,
                                adaptation_amplitude, adaptation_baseline,
                                adaptation_tau_inv, *args, **kwargs):
            return _leaky_integral_calculator(t, slope, origin, alpha, tau_inv,
                            adaptation_amplitude, adaptation_baseline,
                            adaptation_tau_inv, *args, **kwargs)
        _integral_calculator.__doc__ = re.sub('leaky_integral_calculator',
                          '_integral_calculator',
                          _leaky_integral_calculator.__doc__)
    else:
        def _integral_calculator(t, slope, origin, alpha, tau_inv,
                                adaptation_amplitude, adaptation_baseline,
                                adaptation_tau_inv):
            return Leaky._integral_calculator_py(t, slope, origin, alpha, tau_inv,
                            adaptation_amplitude, adaptation_baseline,
                            adaptation_tau_inv)
        _integral_calculator.__doc__ = re.sub('_integral_calculator_py',
                          '_integral_calculator',
                          _integral_calculator_py.__doc__)
    
    _integral_calculator = staticmethod(_integral_calculator)

def prob2AFC(mean1, var1, mean2, var2, low_perror=0., high_perror=0.):
    """
    prob2AFC(mean1, var1, mean2, var2, low_perror=0., high_perror=0.)
    
    Calculate the probability of deciding that the second stimulus has
    a higher value given the theoretically calculated expected values
    and variances of both stimuli
    
    Input:
        mean1: expected value of the stochastic dynamical variable of
            the first stimulus
        var1: variance of the stochastic dynamical variable of the
            first stimulus
        mean2: expected value of the stochastic dynamical variable of
            the second stimulus
        var2: variance of the stochastic dynamical variable of the
            second stimulus
        low_perror: probability of making an error for very low mean2
            values due to the asymmetric lapse rates.
        high_perror: probability of making an error for very high mean2
            values due to the asymmetric lapse rates.
    
    """
    return 1-high_perror - (1-low_perror-high_perror)*stats.norm.cdf(0., loc=mean2-mean1, scale=np.sqrt(var1+var2))

def tests(show=False):
    time_unit_conversion = 1.
    T = 1000.*time_unit_conversion
    dt = 1.*time_unit_conversion
    nsamples = 1000
    
    origin = np.array([0., 1., 0.5]).reshape((-1, 1))
    slope  = np.array([1./T, -1./T, 0.]).reshape((-1, 1))
    alpha  = np.array([0.1, 0.5, 1.]).reshape((1, -1))
    #~ alpha  = np.array([1.]).reshape((1, -1))
    adaptation_amplitude = 1.
    adaptation_baseline = 0.6
    adaptation_tau_inv = 1./(200.*time_unit_conversion)
    background_var = 4.
    background_mean=2.
    stimulator = Stimulator(slope=slope, origin=origin, alpha=alpha,
                            adaptation_amplitude=adaptation_amplitude,
                            adaptation_baseline=adaptation_baseline,
                            adaptation_tau_inv=adaptation_tau_inv,
                            background_var=background_var,
                            background_mean=background_mean)
    
    stimulator.sde_symbols()
    
    leak = np.array([0., 1.]).reshape((-1, 1, 1))
    C_inv = 1./np.array([100., 500., 1000.]).reshape((1, -1, 1))/ time_unit_conversion
    x0 = np.array([0., 0.5, 1.]).reshape((1, 1, -1))
    var0 = 0.
    leaky = Leaky(leak=leak, C_inv=C_inv, x0=x0, var0=var0)
    
    """
    The following takes forever to do and is not simplified in a
    satisfying way, so it is commented out
    
    mean, var, s = leaky.theoretical_symbolic(return_internal_symbols=True)
    mean = sympy.simplify(mean.subs(s['alpha'], 1)).doit()
    var = sympy.simplify(var.subs(s['alpha'], 1)).doit()
    
    replacements = {s['leak']: 'lambda', s['C_inv']: r'1/C', s['x0']: 'x_0', s['var0']: 'var_0',
                    s['slope']: 'm', s['origin']: 'b', s['alpha']: 'alpha',
                    s['adaptation_amplitude']: 'A',
                    s['adaptation_baseline']: 'B',
                    s['adaptation_tau_inv']: r'1/{\tau}_a'}
    for symb, replacement in iteritems(replacements):
        mean = mean.subs(symb, replacement)
        var = var.subs(symb, replacement)
    mean = sympy.simplify(mean)
    var = sympy.simplify(var)
    print(sympy.pretty(mean))
    print(sympy.pretty(var))
    """
    
    x, t, parameters = leaky.sample(dt, T, stimulator, nsamples, time_unit_conversion,
                             return_t=True, return_parameters=True)
    means, var, parameters = leaky.theoretical(stimulator=stimulator, t=t,
                             time_unit_conversion=time_unit_conversion,
                             return_parameters=True)
    stds = np.sqrt(var)
    
    parameters = parameters.as_array()
    
    # Plot simulation
    
    slope  = slope.flatten()
    origin = origin.flatten()
    alpha  = alpha.flatten()
    nlines = np.min([3, x.shape[-1]])
    plt.figure(figsize=(10,9))
    colors = {key: c for key, c in zip(C_inv.flatten(),['r', 'g', 'b'])}
    linestyles = {key: l for key, l in zip(leak.flatten(),['--', '-'])}
    alphas = {key: a for key, a in zip(x0.flatten(),[0.3, 0.6, 1.])}
    gs = gridspec.GridSpec(len(alpha), len(slope))
    is_first = True
    axs = []
    for row in range(len(alpha)):
        temp = []
        for col in range(len(slope)):
            if is_first:
                ax = plt.subplot(gs[row, col])
                is_first = False
            else:
                ax = plt.subplot(gs[row, col], sharex=ax, sharey=ax)
            temp.append(ax)
        axs.append(temp)
    for ind, pars in enumerate(parameters):
        row = list(alpha).index(pars['alpha'])
        col = list(slope).index(pars['slope'])
        ax = axs[row][col]
        tt = pars['C_inv']
        ll = pars['leak']
        xx = pars['x0']
        ax.plot(t, x[ind,:,:nlines], color=colors[tt],
                 linestyle=linestyles[ll], alpha=alphas[xx])
        ax.set_title(r'slope={0} origin={1} $\alpha$={2}'.format(pars['slope'], pars['origin'], pars['alpha']))
        artists = []
        if row==0 and col==0:
            for tt in np.unique(parameters['C_inv']):
                artists.append(plt.Line2D([0], [0], color=colors[tt],
                                label=r'$C_inv={0}$'.format(tt)))
        elif row==0 and col==1:
            for ll in np.unique(parameters['leak']):
                artists.append(plt.Line2D([0], [0], color='k',
                                linestyle=linestyles[ll], label=r'$\lambda={0}$'.format(ll)))
        elif row==0 and col==2:
            for xx in np.unique(parameters['x0']):
                artists.append(plt.Line2D([0], [0], color='k',
                                alpha=alphas[xx], label=r'$x_{{0}}={0}$'.format(xx)))
        if artists:
            ax.legend(handles=artists)
    plt.suptitle('Simulation')
    
    # Plot theoretical
    
    plt.figure(figsize=(10,9))
    gs = gridspec.GridSpec(len(alpha), len(slope))
    is_first = True
    axs = []
    for row in range(len(alpha)):
        temp = []
        for col in range(len(slope)):
            if is_first:
                ax = plt.subplot(gs[row, col])
                is_first = False
            else:
                ax = plt.subplot(gs[row, col], sharex=ax, sharey=ax)
            temp.append(ax)
        axs.append(temp)
    for ind, pars in enumerate(parameters):
        row = list(alpha).index(pars['alpha'])
        col = list(slope).index(pars['slope'])
        ax = axs[row][col]
        tt = pars['C_inv']
        ll = pars['leak']
        xx = pars['x0']
        mean = means[ind]
        std  = stds[ind]
        ax.plot(t, mean, color=colors[tt],
                 linestyle=linestyles[ll], alpha=alphas[xx], linewidth=3)
        ax.fill_between(t, (mean-std), (mean+std), color=colors[tt],
                 alpha=0.1)
        ax.set_title(r'slope={0} origin={1} $\alpha$={2}'.format(pars['slope'], pars['origin'], pars['alpha']))
        artists = []
        if row==0 and col==0:
            for tt in np.unique(parameters['C_inv']):
                artists.append(plt.Line2D([0], [0], color=colors[tt],
                                label=r'$C_inv={0}$'.format(tt)))
        elif row==0 and col==1:
            for ll in np.unique(parameters['leak']):
                artists.append(plt.Line2D([0], [0], color='k',
                                linestyle=linestyles[ll], label=r'$\lambda={0}$'.format(ll)))
        elif row==0 and col==2:
            for xx in np.unique(parameters['x0']):
                artists.append(plt.Line2D([0], [0], color='k',
                                alpha=alphas[xx], label=r'$x_{{0}}={0}$'.format(xx)))
        if artists:
            ax.legend(handles=artists)
    plt.suptitle('Theoretical')
    
    # Plot relative error in the mean
    
    plt.figure(figsize=(10,9))
    gs = gridspec.GridSpec(len(alpha), len(slope))
    is_first = True
    axs = []
    for row in range(len(alpha)):
        temp = []
        for col in range(len(slope)):
            if is_first:
                ax = plt.subplot(gs[row, col])
                is_first = False
            else:
                ax = plt.subplot(gs[row, col], sharex=ax, sharey=ax)
            temp.append(ax)
        axs.append(temp)
    for ind, pars in enumerate(parameters):
        row = list(alpha).index(pars['alpha'])
        col = list(slope).index(pars['slope'])
        ax = axs[row][col]
        tt = pars['C_inv']
        ll = pars['leak']
        xx = pars['x0']
        ax.plot(t, np.mean(x[ind],axis=-1)/1 - means[ind], color=colors[tt],
                 linestyle=linestyles[ll], alpha=alphas[xx])
        ax.set_title(r'slope={0} origin={1} $\alpha$={2}'.format(pars['slope'], pars['origin'], pars['alpha']))
        artists = []
        if row==0 and col==0:
            for tt in np.unique(parameters['C_inv']):
                artists.append(plt.Line2D([0], [0], color=colors[tt],
                                label=r'$C_inv={0}$'.format(tt)))
        elif row==0 and col==1:
            for ll in np.unique(parameters['leak']):
                artists.append(plt.Line2D([0], [0], color='k',
                                linestyle=linestyles[ll], label=r'$\lambda={0}$'.format(ll)))
        elif row==0 and col==2:
            for xx in np.unique(parameters['x0']):
                artists.append(plt.Line2D([0], [0], color='k',
                                alpha=alphas[xx], label=r'$x_{{0}}={0}$'.format(xx)))
        if artists:
            ax.legend(handles=artists)
    plt.suptitle('Mean difference')
    
    # Plot relative error in var
    
    plt.figure(figsize=(10,9))
    gs = gridspec.GridSpec(len(alpha), len(slope))
    is_first = True
    axs = []
    for row in range(len(alpha)):
        temp = []
        for col in range(len(slope)):
            if is_first:
                ax = plt.subplot(gs[row, col])
                is_first = False
            else:
                ax = plt.subplot(gs[row, col], sharex=ax, sharey=ax)
            temp.append(ax)
        axs.append(temp)
    for ind, pars in enumerate(parameters):
        row = list(alpha).index(pars['alpha'])
        col = list(slope).index(pars['slope'])
        ax = axs[row][col]
        tt = pars['C_inv']
        ll = pars['leak']
        xx = pars['x0']
        mean = means[ind]
        std  = stds[ind]
        ax.plot(t, np.var(x[ind], axis=-1)/1 - var[ind], color=colors[tt],
                 linestyle=linestyles[ll], alpha=alphas[xx], linewidth=3)
        ax.set_title(r'slope={0} origin={1} $\alpha$={2}'.format(pars['slope'], pars['origin'], pars['alpha']))
        artists = []
        if row==0 and col==0:
            for tt in np.unique(parameters['C_inv']):
                artists.append(plt.Line2D([0], [0], color=colors[tt],
                                label=r'$C_inv={0}$'.format(tt)))
        elif row==0 and col==1:
            for ll in np.unique(parameters['leak']):
                artists.append(plt.Line2D([0], [0], color='k',
                                linestyle=linestyles[ll], label=r'$\lambda={0}$'.format(ll)))
        elif row==0 and col==2:
            for xx in np.unique(parameters['x0']):
                artists.append(plt.Line2D([0], [0], color='k',
                                alpha=alphas[xx], label=r'$x_{{0}}={0}$'.format(xx)))
        if artists:
            ax.legend(handles=artists)
    plt.suptitle('Var difference')
    
    #~ plt.show(True)
    
    leak = np.array([0., 0.5, 1.])
    C_inv = 1./np.array([100., 500., 1000., 2000.])/time_unit_conversion
    origin = np.arange(3, 23, 2).astype(np.float)
    background_var = 0.
    background_mean = 40.
    
    mean, var = leaky.theoretical(t=t.reshape((1, 1, 1, -1)), x0=0., var0=0.,
                                  leak=leak.reshape((-1, 1, 1, 1)),
                                  C_inv=C_inv.reshape((1, -1, 1, 1)),
                                  origin=origin.reshape((1, 1, -1, 1)),
                                  alpha=1.,
                                  adaptation_amplitude=1.,
                                  adaptation_baseline=0.5,
                                  adaptation_tau_inv=0.6,
                                  background_var=background_var,
                                  background_mean=background_mean,
                                  time_unit_conversion=time_unit_conversion)
    sim_x = Leaky._sample(output_shape=mean.shape+(10,),
                          t=t.reshape((1, 1, 1, -1, 1)),
                          x0=0., var0=0., slope=0.,
                          origin=origin.reshape((1, 1, -1, 1, 1)),
                          leak=leak.reshape((-1, 1, 1, 1, 1)),
                          C_inv=C_inv.reshape((1, -1, 1, 1, 1)),
                          alpha=1.,
                          adaptation_amplitude=1.,
                          adaptation_baseline=0.5,
                          adaptation_tau_inv=600.*time_unit_conversion,
                          background_var=background_var,
                          background_mean=background_mean,
                          time_axis = 3,
                          time_unit_conversion=time_unit_conversion,
                          use_gaussian_approx=False)
    colors = [plt.get_cmap('hot')(xx) for xx in np.linspace(0.25, 0.75, len(origin))]
    
    plt.figure()
    for row, l in enumerate(leak):
        for col, c in enumerate(C_inv):
            ax = plt.subplot(len(leak), len(C_inv), row*len(C_inv)+col+1)
            for b, color in enumerate(colors):
                m = mean[row, col, b]
                v = var[row, col, b]
                xx = sim_x[row, col, b]
                ax.plot(t, m, color=color)
                ax.fill_between(t, m+np.sqrt(v), m-np.sqrt(v), alpha=0.2, color=color)
                ax.plot(t, xx, color=color, alpha=0.1)
            if row==0:
                ax.set_title(r'$1/C={0}$'.format(c))
            if col==0:
                ax.set_ylabel(r'$\lambda={0}$'.format(l))
    plt.suptitle(r'$\alpha=1$ and $slope=0$')
    if show:
        plt.show(True)


if __name__=='__main__':
    tests()
