#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Module for fitting the decision model parameters to the behavioral
dataset

Defines the Fitter class that provides an interface to fit the
experimental data provided by data_io.py.

Author: Luciano Paz
Year: 2017

"""

from __future__ import division, print_function, absolute_import, unicode_literals

try:
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    can_plot = True
except:
    can_plot = False

import os, sys, scipy, pickle, warnings, json, logging, copy, re, pprint, six, time
from six import iteritems
import scipy.signal
import numpy as np
import pandas as pd
from .utils import Bootstraper
from . import data_io as io
from .model import Leaky, Stimulator, prob2AFC
import cma

prettyprinter = pprint.PrettyPrinter(indent=4, width=80, depth=None, stream=None)
prettyformat = prettyprinter.pformat
package_logger = logging.getLogger("fits_module")

PICKLE_PROTOCOL = 2  # For compatibility between python 2 and python 3

__is_py3__ = sys.version_info[0]==3

#~ plt.ion()
#~ crap_fig = plt.figure()
#~ crap_axs = [[plt.subplot(221), plt.subplot(222)],[plt.subplot(223), plt.subplot(224)]]
#~ crap_counter = 0
#~ crap_plotevery = 100

def cmaes_fmin(objective_function, x0, sigma0,
         options=None,
         args=(),
         gradf=None,
         restarts=0,
         restart_from_best='False',
         incpopsize=2,
         eval_initial_x=False,
         noise_handler=None,
         noise_change_sigma_exponent=1,
         noise_kappa_exponent=0,
         bipop=False):
    """
    This cmaes_fmin is just a copy of cmaes.fmin but removing the data
    file logging. The following is a verbatim copy of the cmaes.fmin docstring.
    
    functional interface to the stochastic optimizer CMA-ES
    for non-convex function minimization.
    
    Calling Sequences
    =================
        ``cmaes_fmin(objective_function, x0, sigma0)``
            minimizes `objective_function` starting at `x0` and with standard deviation
            `sigma0` (step-size)
        ``cmaes_fmin(objective_function, x0, sigma0, options={'ftarget': 1e-5})``
            minimizes `objective_function` up to target function value 1e-5, which
            is typically useful for benchmarking.
        ``cmaes_fmin(objective_function, x0, sigma0, args=('f',))``
            minimizes `objective_function` called with an additional argument ``'f'``.
        ``cmaes_fmin(objective_function, x0, sigma0, options={'ftarget':1e-5, 'popsize':40})``
            uses additional options ``ftarget`` and ``popsize``
        ``cmaes_fmin(objective_function, esobj, None, options={'maxfevals': 1e5})``
            uses the `CMAEvolutionStrategy` object instance `esobj` to optimize
            `objective_function`, similar to `esobj.optimize()`.
    
    Arguments
    =========
        `objective_function`
            function to be minimized. Called as ``objective_function(x,
            *args)``. `x` is a one-dimensional `numpy.ndarray`.
            `objective_function` can return `numpy.NaN`,
            which is interpreted as outright rejection of solution `x`
            and invokes an immediate resampling and (re-)evaluation
            of a new solution not counting as function evaluation.
        `x0`
            list or `numpy.ndarray`, initial guess of minimum solution
            before the application of the geno-phenotype transformation
            according to the ``transformation`` option.  It can also be
            a string holding a Python expression that is evaluated
            to yield the initial guess - this is important in case
            restarts are performed so that they start from different
            places.  Otherwise `x0` can also be a `cma.CMAEvolutionStrategy`
            object instance, in that case `sigma0` can be ``None``.
        `sigma0`
            scalar, initial standard deviation in each coordinate.
            `sigma0` should be about 1/4th of the search domain width
            (where the optimum is to be expected). The variables in
            `objective_function` should be scaled such that they
            presumably have similar sensitivity.
            See also option `scaling_of_variables`.
        `options`
            a dictionary with additional options passed to the constructor
            of class ``CMAEvolutionStrategy``, see ``cma.CMAOptions()``
            for a list of available options.
        ``args=()``
            arguments to be used to call the `objective_function`
        ``gradf``
            gradient of f, where ``len(gradf(x, *args)) == len(x)``.
            `gradf` is called once in each iteration if
            ``gradf is not None``.
        ``restarts=0``
            number of restarts with increasing population size, see also
            parameter `incpopsize`, implementing the IPOP-CMA-ES restart
            strategy, see also parameter `bipop`; to restart from
            different points (recommended), pass `x0` as a string.
        ``restart_from_best=False``
            which point to restart from
        ``incpopsize=2``
            multiplier for increasing the population size `popsize` before
            each restart
        ``eval_initial_x=None``
            evaluate initial solution, for `None` only with elitist option
        ``noise_handler=None``
            a ``NoiseHandler`` instance or ``None``, a simple usecase is
            ``cma.cmaes_fmin(f, 6 * [1], 1, noise_handler=cma.NoiseHandler(6))``
            see ``help(cma.NoiseHandler)``.
        ``noise_change_sigma_exponent=1``
            exponent for sigma increment for additional noise treatment
        ``noise_evaluations_as_kappa``
            instead of applying reevaluations, the "number of evaluations"
            is (ab)used as scaling factor kappa (experimental).
        ``bipop``
            if True, run as BIPOP-CMA-ES; BIPOP is a special restart
            strategy switching between two population sizings - small
            (like the default CMA, but with more focused search) and
            large (progressively increased as in IPOP). This makes the
            algorithm perform well both on functions with many regularly
            or irregularly arranged local optima (the latter by frequently
            restarting with small populations).  For the `bipop` parameter
            to actually take effect, also select non-zero number of
            (IPOP) restarts; the recommended setting is ``restarts<=9``
            and `x0` passed as a string.  Note that small-population
            restarts do not count into the total restart count.
    
    Optional Arguments
    ==================
    All values in the `options` dictionary are evaluated if they are of
    type `str`, besides `verb_filenameprefix`, see class `CMAOptions` for
    details. The full list is available via ``cma.CMAOptions()``.
    
    >>> import cma
    >>> cma.CMAOptions()
    
    Subsets of options can be displayed, for example like
    ``cma.CMAOptions('tol')``, or ``cma.CMAOptions('bound')``,
    see also class `CMAOptions`.
    
    Return
    ======
    Return the list provided by `CMAEvolutionStrategy.result()` appended
    with termination conditions, an `OOOptimizer` and a `BaseDataLogger`::
    
        res = es.result() + (es.stop(), es, logger)
    
    where
        - ``res[0]`` (``xopt``) -- best evaluated solution
        - ``res[1]`` (``fopt``) -- respective function value
        - ``res[2]`` (``evalsopt``) -- respective number of function evaluations
        - ``res[3]`` (``evals``) -- number of overall conducted objective function evaluations
        - ``res[4]`` (``iterations``) -- number of overall conducted iterations
        - ``res[5]`` (``xmean``) -- mean of the final sample distribution
        - ``res[6]`` (``stds``) -- effective stds of the final sample distribution
        - ``res[-3]`` (``stop``) -- termination condition(s) in a dictionary
        - ``res[-2]`` (``cmaes``) -- class `CMAEvolutionStrategy` instance
        - ``res[-1]`` (``logger``) -- class `CMADataLogger` instance # EDIT THIS IS ALWAYS SET TO None TO REMOVE FILE LOGGING
    
    Details
    =======
    This function is an interface to the class `CMAEvolutionStrategy`. The
    latter class should be used when full control over the iteration loop
    of the optimizer is desired.
    
    """
    if 1 < 3:  # try: # pass on KeyboardInterrupt    if 1 < 3:  # try: # pass on KeyboardInterrupt
        if not objective_function:  # cma.fmin(0, 0, 0)
            return cma.CMAOptions()  # these opts are by definition valid

        fmin_options = locals().copy()  # archive original options
        del fmin_options['objective_function']
        del fmin_options['x0']
        del fmin_options['sigma0']
        del fmin_options['options']
        del fmin_options['args']

        if options is None:
            options = cma_default_options
        cma.CMAOptions().check_attributes(options)  # might modify options
        # checked that no options.ftarget =
        opts = cma.CMAOptions(options.copy()).complement()

        callback = []

        # BIPOP-related variables:
        runs_with_small = 0
        small_i = []
        large_i = []
        popsize0 = None  # to be evaluated after the first iteration
        maxiter0 = None  # to be evaluated after the first iteration
        base_evals = 0

        irun = 0
        best = cma.optimization_tools.BestSolution()
        while True:  # restart loop
            sigma_factor = 1

            # Adjust the population according to BIPOP after a restart.
            if not bipop:
                # BIPOP not in use, simply double the previous population
                # on restart.
                if irun > 0:
                    popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                    opts['popsize'] = popsize0 * popsize_multiplier

            elif irun == 0:
                # Initial run is with "normal" population size; it is
                # the large population before first doubling, but its
                # budget accounting is the same as in case of small
                # population.
                poptype = 'small'

            elif sum(small_i) < sum(large_i):
                # An interweaved run with small population size
                poptype = 'small'
                if 11 < 3:  # not needed when compared to irun - runs_with_small
                    restarts += 1  # A small restart doesn't count in the total
                runs_with_small += 1  # _Before_ it's used in popsize_lastlarge

                sigma_factor = 0.01**np.random.uniform()  # Local search
                popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                opts['popsize'] = np.floor(popsize0 * popsize_multiplier**(np.random.uniform()**2))
                opts['maxiter'] = min(maxiter0, 0.5 * sum(large_i) / opts['popsize'])
                # print('small basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            else:
                # A run with large population size; the population
                # doubling is implicit with incpopsize.
                poptype = 'large'

                popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                opts['popsize'] = popsize0 * popsize_multiplier
                opts['maxiter'] = maxiter0
                # print('large basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            # recover from a CMA object
            if irun == 0 and isinstance(x0, cma.CMAEvolutionStrategy):
                es = x0
                x0 = es.inputargs['x0']  # for the next restarts
                if np.isscalar(sigma0) and np.isfinite(sigma0) and sigma0 > 0:
                    es.sigma = sigma0
                # debatable whether this makes sense:
                sigma0 = es.inputargs['sigma0']  # for the next restarts
                if options is not None:
                    es.opts.set(options)
                # ignore further input args and keep original options
            else:  # default case
                if irun and eval(str(fmin_options['restart_from_best'])):
                    utils.print_warning('CAVE: restart_from_best is often not useful',
                                        verbose=opts['verbose'])
                    es = cma.CMAEvolutionStrategy(best.x, sigma_factor * sigma0, opts)
                else:
                    es = cma.CMAEvolutionStrategy(x0, sigma_factor * sigma0, opts)
                if eval_initial_x or es.opts['CMA_elitist'] == 'initial' \
                   or (es.opts['CMA_elitist'] and eval_initial_x is None):
                    x = es.gp.pheno(es.mean,
                                    into_bounds=es.boundary_handler.repair,
                                    archive=es.sent_solutions)
                    es.f0 = objective_function(x, *args)
                    es.best.update([x], es.sent_solutions,
                                   [es.f0], 1)
                    es.countevals += 1

            opts = es.opts  # processed options, unambiguous
            # a hack:
            fmin_opts = cma.CMAOptions("unchecked", **fmin_options.copy())
            for k in fmin_opts:
                # locals() cannot be modified directly, exec won't work
                # in 3.x, therefore
                fmin_opts.eval(k, loc={'N': es.N,
                                       'popsize': opts['popsize']},
                               correct_key=False)

            #~ es.logger.append = opts['verb_append'] or es.countiter > 0 or irun > 0
            # es.logger is "the same" logger, because the "identity"
            # is only determined by the `verb_filenameprefix` option
            #~ logger = es.logger  # shortcut
            #~ try:
                #~ logger.persistent_communication_dict.update(
                    #~ {'variable_annotations':
                    #~ objective_function.variable_annotations})
            #~ except AttributeError:
                #~ pass

            #~ if 11 < 3:
                #~ if es.countiter == 0 and es.opts['verb_log'] > 0 and \
                        #~ not es.opts['verb_append']:
                   #~ logger = CMADataLogger(es.opts['verb_filenameprefix']
                                            #~ ).register(es)
                   #~ logger.add()
                #~ es.writeOutput()  # initial values for sigma etc

            if noise_handler:
                if isinstance(noise_handler, type):
                    noisehandler = noise_handler(es.N)
                else:
                    noisehandler = noise_handler
                noise_handling = True
                if fmin_opts['noise_change_sigma_exponent'] > 0:
                    es.opts['tolfacupx'] = inf
            else:
                noisehandler = cma.optimization_tools.NoiseHandler(es.N, 0)  # switched off
                noise_handling = False
            es.noise_handler = noisehandler

            # the problem: this assumes that good solutions cannot take longer than bad ones:
            # with EvalInParallel(objective_function, 2, is_feasible=opts['is_feasible']) as eval_in_parallel:
            if 1 < 3:
                while not es.stop():  # iteration loop
                    # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
                    X, fit = es.ask_and_eval(objective_function, args, gradf=gradf,
                                             evaluations=noisehandler.evaluations,
                                             aggregation=np.median)  # treats NaN with resampling
                    # TODO: check args and in case use args=(noisehandler.evaluations, )

                    if 11 < 3 and opts['vv']:  # inject a solution
                        # use option check_point = [0]
                        if 0 * np.random.randn() >= 0:
                            X[0] = 0 + opts['vv'] * es.sigma**0 * np.random.randn(es.N)
                            fit[0] = objective_function(X[0], *args)
                            # print fit[0]
                    if es.opts['verbose'] > 4:
                        if es.countiter > 1 and min(fit) > es.best.last.f:
                            unsuccessful_iterations_count += 1
                            if unsuccessful_iterations_count > 4:
                                utils.print_message('%d unsuccessful iterations'
                                                    % unsuccessful_iterations_count,
                                                    iteration=es.countiter)
                        else:
                            unsuccessful_iterations_count = 0
                    es.tell(X, fit)  # prepare for next iteration
                    if noise_handling:  # it would be better to also use these f-evaluations in tell
                        es.sigma *= noisehandler(X, fit, objective_function, es.ask,
                                                 args=args)**fmin_opts['noise_change_sigma_exponent']

                        es.countevals += noisehandler.evaluations_just_done  # TODO: this is a hack, not important though
                        # es.more_to_write.append(noisehandler.evaluations_just_done)
                        if noisehandler.maxevals > noisehandler.minevals:
                            es.more_to_write.append(noisehandler.evaluations)
                        if 1 < 3:
                            es.sp.cmean *= np.exp(-noise_kappa_exponent * np.tanh(noisehandler.noiseS))
                            if es.sp.cmean > 1:
                                es.sp.cmean = 1
                    for f in callback:
                        f is None or f(es)
                    es.disp()
                    #~ logger.add(# more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                               #~ modulo=1 if es.stop() and logger.modulo else None)
                    #~ if (opts['verb_log'] and opts['verb_plot'] and
                          #~ (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                        #~ logger.plot(324)

            # end while not es.stop
            mean_pheno = es.gp.pheno(es.mean,
                                     into_bounds=es.boundary_handler.repair,
                                     archive=es.sent_solutions)
            fmean = objective_function(mean_pheno, *args)
            es.countevals += 1

            es.best.update([mean_pheno], es.sent_solutions, [fmean], es.countevals)
            best.update(es.best, es.sent_solutions)  # in restarted case
            # es.best.update(best)

            this_evals = es.countevals - base_evals
            base_evals = es.countevals

            # BIPOP stats update

            if irun == 0:
                popsize0 = opts['popsize']
                maxiter0 = opts['maxiter']
                # XXX: This might be a bug? Reproduced from Matlab
                # small_i.append(this_evals)

            if bipop:
                if poptype == 'small':
                    small_i.append(this_evals)
                else:  # poptype == 'large'
                    large_i.append(this_evals)

            # final message
            if opts['verb_disp']:
                es.result_pretty(irun, time.asctime(time.localtime()),
                                 best.f)

            irun += 1
            # if irun > fmin_opts['restarts'] or 'ftarget' in es.stop() \
            # if irun > restarts or 'ftarget' in es.stop() \
            if irun - runs_with_small > fmin_opts['restarts'] or 'ftarget' in es.stop() \
                    or 'maxfevals' in es.stop(check=False) or 'callback' in es.stop(check=False):
                break
            opts['verb_append'] = es.countevals
            opts['popsize'] = fmin_opts['incpopsize'] * es.sp.popsize  # TODO: use rather options?
            opts['seed'] += 1

        # while irun

        # es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
        if irun:
            es.best.update(best)
            # TODO: there should be a better way to communicate the overall best
        return es.result + (es.stop(), es, None)
        ### 4560
        # TODO refine output, can #args be flexible?
        # is this well usable as it is now?
    else:  # except KeyboardInterrupt:  # Exception, e:
        if eval(str(options['verb_disp'])) > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit

def load_Fitter_from_file(fname):
    """
    load_Fitter_from_file(fname)
    
    Return the Fitter instance that is stored in the file with name fname.
    
    """
    with open(fname,'rb') as f:
        if __is_py3__:
            fitter = pickle.load(f, encoding='latin1')
        else:
            fitter = pickle.load(f)
    return fitter

def Fitter_filename(name, session, method, optimizer, suffix, fits_path='fits'):
    """
    Fitter_filename(name, session, method, optimizer, suffix, fits_path='fits')
    
    Returns a string. Returns the formated filename for the supplied
    experiment, method, name, session, optimizer, suffix and
    confidence_map_method strings.
    
    The output is
    os.path.join('{fits_path}','fit_subject_{name}_session_{session}_{method}_{optimizer}{suffix}.pkl')
    
    """
    name = '[{}]'.format('-'.join([str(s).strip() for s in sorted(name)]))
    session = '[{}]'.format(' '.join([str(s).strip() for s in sorted(session)]))
    return os.path.join(fits_path,'fit_subject_{name}_session_{session}_{method}_{optimizer}{suffix}.pkl'.format(
                name=name, session=session, method=method, optimizer=optimizer, suffix=suffix))

def stringify_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in iteritems(obj):
            out[str(k)] = stringify_keys(v)
        return out
    elif isinstance(obj, list):
        return [stringify_keys(o) for o in obj]
    elif isinstance(obj, six.string_types):
        return str(obj)
    else:
        return obj

class Fitter(object):
    duration_parameters = {'leak': 'duration_leak',
                           'tau': 'duration_tau',
                           'x0': 'duration_x0',
                           'var0': 'duration_var0',
                           'background_var': 'duration_background_var',
                           'background_mean': 'duration_background_mean',
                           'alpha': 'duration_alpha',
                           'low_perror': 'duration_low_perror',
                           'high_perror': 'duration_high_perror',
                           'adaptation_amplitude': 'adaptation_amplitude',
                           'adaptation_baseline': 'adaptation_baseline',
                           'adaptation_tau_inv': 'adaptation_tau_inv'}
    intensity_parameters = {'leak': 'intensity_leak',
                            'tau': 'intensity_tau',
                            'x0': 'intensity_x0',
                            'var0': 'intensity_var0',
                            'background_var': 'intensity_background_var',
                            'background_mean': 'intensity_background_mean',
                            'alpha': 'intensity_alpha',
                            'low_perror': 'intensity_low_perror',
                            'high_perror': 'intensity_high_perror',
                            'adaptation_amplitude': 'adaptation_amplitude',
                            'adaptation_baseline': 'adaptation_baseline',
                            'adaptation_tau_inv': 'adaptation_tau_inv'}
    
    human_duration_slider_query  = "experiment=='slider' and task==1"
    human_intensity_slider_query = "experiment=='slider' and task==0"
    
    human_duration_discrimination_query  = "experiment=='human_dur_disc'"
    human_intensity_discrimination_query = "experiment=='human_int_disc'"
    
    #~ human_duration_psychometric_query  = "experiment=='human_dur_disc' and "\
                                         #~ "intensity1==32. and duration1==300"
    #~ human_intensity_psychometric_query = "experiment=='human_int_disc' and "\
                                         #~ "intensity1==32. and duration1==300"
    human_duration_psychometric_query  = "experiment=='human_dur_disc' and "\
                                         "duration1==300"
    human_intensity_psychometric_query = "experiment=='human_int_disc' and "\
                                         "intensity1==32."
    
    rat_duration_discrimination_query  = "experiment=='rat_disc' and task==1"
    rat_intensity_discrimination_query = "experiment=='rat_disc' and task==0"
    
    rat_duration_psychometric_query  = "experiment=='rat_disc' and "\
                                       "task==1 and intensity1==64. and "\
                                       "duration1==334 and abs(NSD)<0.35"
    rat_intensity_psychometric_query = "experiment=='rat_disc' and "\
                                       "task==0 and intensity1==64. and "\
                                       "duration1==334 and abs(NSD)<0.35"
    # Initer
    def __init__(self, dataframe, method=None, optimizer='cma',
                 suffix='',
                 fits_path='fits'):
        """
        Fitter(self, dataframe, method=None, optimizer='cma',
                suffix='',
                fits_path='fits')
        
        Construct a Fitter instance that interfaces the fitting procedure
        of the model's likelihood of the observed subjectSession data.
        Input:
            dataframe: A pandas dataframe with the experimental data
            method: None or a string that determines the merit function
                that will be used. If None, the merit function will try
                to be inferred from the dataframe structure, and raise
                an exception if this fails. Possible method values are
                'slider_disc', 'disc', 'slider_disc_2alpha' and
                'disc_2alpha'.
            optimizer: The optimizer used for the fitting procedure.
                Available optimizers are 'cma', scipy.optimize.basinhopping
                and scipy.optimize methods called from minimize and
                minimize_scalar.
            decisionModelKwArgs: A dict of kwargs to use for the construction
                of the used DecisionModel instance
            suffix: A string suffix to append to the saved filenames
            fits_path: The path to the directory where the fit results
                should be saved.
        
        Output: A Fitter instance that can be used as an interface to
            fit the model's parameters to the subjectSession data
        
        Example assuming that subjectSession is a data_io_cognition.SubjectSession
        instance:
        fitter = Fitter(subjectSession)
        fitter.fit()
        fitter.save()
        print(fitter.stats)
        fitter.plot()
        
        """
        
        self.__initlogger__()
        self.logger.debug("Entered Fitter.__init__(...)")
        self.fits_path = fits_path
        self.logger.debug('Setted fits_path = %s',self.fits_path)
        self.set_data(dataframe)
        self.set_method(method)
        self.optimizer = str(optimizer)
        self.logger.debug('Setted Fitter optimizer = %s',self.optimizer)
        self.suffix = str(suffix)
        self.logger.debug('Setted Fitter suffix = %s',self.suffix)
        self.__fit_internals__ = None
    
    def __initlogger__(self):
        self.logger = logging.getLogger("fits_module.Fitter")
    
    def __str__(self):
        if hasattr(self,'_fit_arguments'):
            _fit_arguments = self._fit_arguments
        else:
            _fit_arguments = None
        if hasattr(self,'_fit_output'):
            _fit_output = self._fit_output
        else:
            _fit_output = None
        string = """
<{class_module}.{class_name} object at {address}>
fits_path = {fits_path},
experiments = {experiments},
subjects = {subjects},
sessions = {sessions},
method = {method},
optimizer = {optimizer},
suffix = {suffix},
_fit_arguments = {_fit_arguments},
_fit_output = {_fit_output}
        """.format(class_module=self.__class__.__module__,
                    class_name=self.__class__.__name__,
                    address=hex(id(self)),
                    fits_path=self.fits_path,
                    experiments = self.experiments,
                    subjects = self.subjects,
                    sessions = self.sessions,
                    method = self.method,
                    optimizer = self.optimizer,
                    suffix = self.suffix,
                    _fit_arguments = prettyformat(self._fit_arguments),
                    _fit_output = prettyformat(self._fit_output))
        return string
    
    # Setters
    def set_data(self, dataframe):
        """
        self.set_data(dataframe)
        
        This function takes a pandas dataframe of raw data to be fitted.
        It parses the raw data to extract the experiment, subject id and 
        session numbers, along with all data needed for the fits.
        
        """
        
        self.experiments = sorted(dataframe.experiment.unique())
        self.logger.debug('Setted Fitter experiments = {0}'.format(prettyformat(self.experiments)))
        self.subjects = sorted(dataframe.subject.unique())
        self.logger.debug('Setted Fitter subjects = {0}'.format(prettyformat(self.subjects)))
        self.sessions = sorted(dataframe.session.unique())
        self.logger.debug('Setted Fitter sessions = {0}'.format(prettyformat(self.sessions)))
        
        self.data = dataframe
        self.logger.debug('Setted Fitter data of shape = {0}'.format(self.data.shape))
        
        self.__has_duration_slider = False
        self.__has_intensity_slider = False
        
        # Duration slider
        d = self.data.query(self.human_duration_slider_query)
        if len(d)>0:
            self.__has_duration_slider = True
            
            # Get the responses of the trials
            self.slider_duration_response = d['answer'].values
            
            # Get the stimuli pairs of duration and intensity
            stims = np.array([d.duration1, d.intensity1]).T
            
            # Find the unique pairs and the mapping to the true stim train
            unique_dur_stims, self.inv_dur_stims_inds = np.unique(stims, return_inverse=True, axis=0)
            
            assert np.all(stims==unique_dur_stims[self.inv_dur_stims_inds]), "Unique stim problems"
            self.slider_duration_t = unique_dur_stims[:,0]
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.slider_duration_sp = unique_dur_stims[:,1]
            self.slider_duration_origins = self.slider_duration_sp * np.sqrt(0.5*np.pi)
            self.slider_duration_slopes = np.zeros_like(self.slider_duration_origins)
            
            # Compute the median response across the T/sp pairs
            d = self.data.query(self.human_duration_slider_query)\
                .groupby(['duration1', 'intensity1'])['answer']\
                .apply(lambda x: Bootstraper.Median(x, 1000, 0.5, return_std=True))
            self.slider_duration_curves = d.xs('mean', level=2).values.flatten()
            self.slider_duration_curves_std = d.xs('std', level=2).values.flatten()
            
            # Correct the std for zero variance cases (normally due to few trials)
            if np.all(self.slider_duration_curves_std==0):
                inds = self.slider_duration_curves_std==0
                self.slider_duration_curves_std[inds] = 5.
            elif np.any(self.slider_duration_curves_std==0):
                inds = self.slider_duration_curves_std==0
                self.slider_duration_curves_std[inds] = np.min(self.slider_duration_curves_std[np.logical_not(inds)])
            
            self.logger.debug('Stored {0} trials of slider duration experiment trials'.format(stims.shape[0]))
        else:
            self.logger.debug('No duration slider trials available')
        
        # Intensity slider
        d = self.data.query(self.human_intensity_slider_query)
        if len(d)>0:
            self.__has_intensity_slider = True
            
            # Get the responses of the trials
            self.slider_intensity_response = d['answer'].values
            
            # Get the stimuli pairs of duration and intensity
            stims = np.array([d.duration1, d.intensity1]).T
            
            # Find the unique pairs and the mapping to the true stim train
            unique_int_stims, self.inv_int_stims_inds = np.unique(stims, return_inverse=True, axis=0)
            self.slider_intensity_t = unique_int_stims[:,0]
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.slider_intensity_sp = unique_int_stims[:,1]
            self.slider_intensity_origins = self.slider_intensity_sp * np.sqrt(0.5*np.pi)
            self.slider_intensity_slopes = np.zeros_like(self.slider_intensity_origins)
            
            # Compute the median response across the T/sp pairs
            d = self.data.query(self.human_intensity_slider_query)\
                .groupby(['duration1', 'intensity1'])['answer']\
                .apply(lambda x: Bootstraper.Median(x, 1000, 0.5, return_std=True))
            self.slider_intensity_curves = d.xs('mean', level=2).values.flatten()
            self.slider_intensity_curves_std = d.xs('std', level=2).values.flatten()
            
            # Correct the std for zero variance cases (normally due to few trials)
            if np.all(self.slider_intensity_curves_std==0):
                inds = self.slider_intensity_curves_std==0
                self.slider_intensity_curves_std[inds] = 5.
            elif np.any(self.slider_intensity_curves_std==0):
                inds = self.slider_intensity_curves_std==0
                self.slider_intensity_curves_std[inds] = np.min(self.slider_intensity_curves_std[np.logical_not(inds)])
            
            self.logger.debug('Stored {0} trials of slider intensity experiment trials'.format(stims.shape[0]))
        else:
            self.logger.debug('No intensity slider trials available')
        
        self.__has_duration_disc = False
        
        # Human duration discrimination
#        d = self.data.query(self.human_duration_discrimination_query)
        d = self.data.query(self.human_duration_psychometric_query)
        if len(d)>0:
            self.__has_duration_disc = True
            
            # Group by stim pairs T1/sp1 T2/sp2, and compute number of decisions in favor of each option
            d = d.groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])
            n_dec2 = d['answer'].sum()
            n_dec1 = d['correct'].count() - n_dec2
            self.n_dur_dec1 = n_dec1.values
            self.n_dur_dec2 = n_dec2.values
            
            # Compute the term of the binomial likelihood due to the combinatorial number of responses
            self.loglike_scale_dur = scipy.special.gammaln(self.n_dur_dec1+self.n_dur_dec2)
            self.loglike_scale_dur[self.n_dur_dec1!=0]-= scipy.special.gammaln(self.n_dur_dec1[self.n_dur_dec1!=0])
            self.loglike_scale_dur[self.n_dur_dec2!=0]-= scipy.special.gammaln(self.n_dur_dec2[self.n_dur_dec2!=0])
            self.loglike_scale_dur = np.sum(self.loglike_scale_dur)
            
            # Get unique stim pair values
            stims = np.array(sorted(d.groups.keys()))
            self.disc_duration_t = np.concatenate((stims[:,0],stims[:,2]), axis=0)
            self.disc_duration_sp = np.concatenate((stims[:,1],stims[:,3]), axis=0)
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.disc_duration_origins = self.disc_duration_sp * np.sqrt(0.5*np.pi)
            self.disc_duration_slopes = np.zeros_like(self.disc_duration_origins)
            
            # Get the data of the vertical psychometric in humans
            d = self.data.query(self.human_duration_psychometric_query)
            
            # Compute the mean of responding in favor of stim2, for each stim pair
            bla = d.assign(prob2=(d['answer']).values)\
                .groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])['prob2']\
                .apply(lambda x: Bootstraper.Mean(x, 1000, 0.5, return_std=True))
            means = bla.xs('mean', level=4)
            stds = bla.xs('std', level=4)
            
            # Convert the stim features and the data of the psychometrics from the means and stds pandas series to numpy arrays
            self.psych_duration_t = []
            self.psych_duration_sp = []
            self.psych_duration = []
            self.psych_duration_std = []
            for (t1, sp1, t2, sp2), psych in iteritems(means):
                self.psych_duration_t.append([t1, t2])
                self.psych_duration_sp.append([sp1, sp2])
                self.psych_duration.append(psych)
                self.psych_duration_std.append(stds[(t1, sp1, t2, sp2)])
            self.psych_duration_t = np.array(self.psych_duration_t)
            self.psych_duration_sp = np.array(self.psych_duration_sp)
            self.psych_duration = np.array(self.psych_duration)
            self.psych_duration_std = np.array(self.psych_duration_std)
            
            # Correct the std for zero variance cases (normally due to few trials)
            if np.all(self.psych_duration_std==0):
                inds = self.psych_duration_std==0
                self.psych_duration_std[inds] = 0.5
            elif np.any(self.psych_duration_std==0):
                inds = self.psych_duration_std==0
                self.psych_duration_std[inds] = np.min(self.psych_duration_std[np.logical_not(inds)])
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.psych_duration_origins = self.psych_duration_sp * np.sqrt(0.5*np.pi)
            self.psych_duration_slopes = np.zeros_like(self.psych_duration_origins)
            
            self.logger.debug('Stored {0} trials of discrimination duration experiment trials'.format(stims.shape[0]))
        else:
            self.logger.debug('No human duration discrimination trials available')
        
        self.__has_intensity_disc = False
        # Human intensity
#        d = self.data.query(self.human_intensity_discrimination_query)
        d = self.data.query(self.human_intensity_psychometric_query)
        if len(d)>0:
            self.__has_intensity_disc = True
            
            # Group by stim pairs T1/sp1 T2/sp2, and compute number of decisions in favor of each option
            d = d.groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])
            n_dec2 = d['answer'].sum()
            n_dec1 = d['correct'].count() - n_dec2
            self.n_int_dec1 = n_dec1.values
            self.n_int_dec2 = n_dec2.values
            
            # Compute the term of the binomial likelihood due to the combinatorial number of responses
            self.loglike_scale_int = scipy.special.gammaln(self.n_int_dec1+self.n_int_dec2)
            self.loglike_scale_int[self.n_int_dec1!=0]-= scipy.special.gammaln(self.n_int_dec1[self.n_int_dec1!=0])
            self.loglike_scale_int[self.n_int_dec2!=0]-= scipy.special.gammaln(self.n_int_dec2[self.n_int_dec2!=0])
            self.loglike_scale_int = np.sum(self.loglike_scale_int)
            
            # Get unique stim pair values
            stims = np.array(sorted(d.groups.keys()))
            self.disc_intensity_t = np.concatenate((stims[:,0],stims[:,2]), axis=0)
            self.disc_intensity_sp = np.concatenate((stims[:,1],stims[:,3]), axis=0)
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.disc_intensity_origins = self.disc_intensity_sp * np.sqrt(0.5*np.pi)
            self.disc_intensity_slopes = np.zeros_like(self.disc_intensity_origins)
            
            # Get the data of the vertical psychometric in humans
            d = self.data.query(self.human_intensity_psychometric_query)
            
            # Compute the mean of responding in favor of stim2, for each stim pair
            bla = d.assign(prob2=(d['answer']).values)\
                .groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])['prob2']\
                .apply(lambda x: Bootstraper.Mean(x, 1000, 0.5, return_std=True))
            means = bla.xs('mean', level=4)
            stds = bla.xs('std', level=4)
            
            # Convert the stim features and the data of the psychometrics from the means and stds pandas series to numpy arrays
            self.psych_intensity_t = []
            self.psych_intensity_sp = []
            self.psych_intensity = []
            self.psych_intensity_std = []
            for (t1, sp1, t2, sp2), psych in iteritems(means):
                self.psych_intensity_t.append([t1, t2])
                self.psych_intensity_sp.append([sp1, sp2])
                self.psych_intensity.append(psych)
                self.psych_intensity_std.append(stds[(t1, sp1, t2, sp2)])
            self.psych_intensity_t = np.array(self.psych_intensity_t)
            self.psych_intensity_sp = np.array(self.psych_intensity_sp)
            self.psych_intensity = np.array(self.psych_intensity)
            self.psych_intensity_std = np.array(self.psych_intensity_std)
            
            # Correct the std for zero variance cases (normally due to few trials)
            if np.all(self.psych_intensity_std==0):
                inds = self.psych_intensity_std==0
                self.psych_intensity_std[inds] = 0.5
            elif np.any(self.psych_intensity_std==0):
                inds = self.psych_intensity_std==0
                self.psych_intensity_std[inds] = np.min(self.psych_intensity_std[np.logical_not(inds)])
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.psych_intensity_origins = self.psych_intensity_sp * np.sqrt(0.5*np.pi)
            self.psych_intensity_slopes = np.zeros_like(self.psych_intensity_origins)
            
            self.logger.debug('Stored {0} trials of discrimination intensity experiment trials'.format(stims.shape[0]))
        else:
            self.logger.debug('No human intensity discrimination trials available')
        
        # Rat duration
#        d = self.data.query(self.rat_duration_discrimination_query)
        d = self.data.query(self.rat_duration_psychometric_query)
        if len(d)>0:
            self.__has_duration_disc = True
            
            # Group by stim pairs T1/sp1 T2/sp2, and compute number of decisions in favor of each option
            d = d.groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])
            n_dec2 = d['answer'].sum()
            n_dec1 = d['correct'].count() - n_dec2
            self.n_dur_dec1 = n_dec1.values
            self.n_dur_dec2 = n_dec2.values
            
            # Compute the term of the binomial likelihood due to the combinatorial number of responses
            self.loglike_scale_dur = scipy.special.gammaln(self.n_dur_dec1+self.n_dur_dec2)
            self.loglike_scale_dur[self.n_dur_dec1!=0]-= scipy.special.gammaln(self.n_dur_dec1[self.n_dur_dec1!=0])
            self.loglike_scale_dur[self.n_dur_dec2!=0]-= scipy.special.gammaln(self.n_dur_dec2[self.n_dur_dec2!=0])
            self.loglike_scale_dur = np.sum(self.loglike_scale_dur)
            
            # Get unique stim pair values
            stims = np.array(sorted(d.groups.keys()))
            self.disc_duration_t = np.concatenate((stims[:,0],stims[:,2]), axis=0)
            self.disc_duration_sp = np.concatenate((stims[:,1],stims[:,3]), axis=0)
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.disc_duration_origins = self.disc_duration_sp * np.sqrt(0.5*np.pi)
            self.disc_duration_slopes = np.zeros_like(self.disc_duration_origins)
            
            # Get the data of the vertical psychometric in rats
            d = self.data.query(self.rat_duration_psychometric_query)
            
            # Compute the mean of responding in favor of stim2, for each stim pair
            bla = d.assign(prob2=(d['answer']).values)\
                .groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])['prob2']\
                .apply(lambda x: Bootstraper.Mean(x, 1000, 0.5, return_std=True))
            means = bla.xs('mean', level=4)
            stds = bla.xs('std', level=4)
            
            # Convert the stim features and the data of the psychometrics from the means and stds pandas series to numpy arrays
            self.psych_duration_t = []
            self.psych_duration_sp = []
            self.psych_duration = []
            self.psych_duration_std = []
            for (t1, sp1, t2, sp2), psych in iteritems(means):
                self.psych_duration_t.append([t1, t2])
                self.psych_duration_sp.append([sp1, sp2])
                self.psych_duration.append(psych)
                self.psych_duration_std.append(stds[(t1, sp1, t2, sp2)])
            self.psych_duration_t = np.array(self.psych_duration_t)
            self.psych_duration_sp = np.array(self.psych_duration_sp)
            self.psych_duration = np.array(self.psych_duration)
            self.psych_duration_std = np.array(self.psych_duration_std)
            
            # Correct the std for zero variance cases (normally due to few trials)
            if np.all(self.psych_duration_std==0):
                inds = self.psych_duration_std==0
                self.psych_duration_std[inds] = 0.5
            elif np.any(self.psych_duration_std==0):
                inds = self.psych_duration_std==0
                self.psych_duration_std[inds] = np.min(self.psych_duration_std[np.logical_not(inds)])
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.psych_duration_origins = self.psych_duration_sp * np.sqrt(0.5*np.pi)
            self.psych_duration_slopes = np.zeros_like(self.psych_duration_origins)
            
            self.logger.debug('Stored {0} trials of discrimination duration experiment trials'.format(stims.shape[0]))
            
        else:
            self.logger.debug('No rat discrimination trials available')
        
        # Rat intensity
#        d = self.data.query(self.rat_intensity_discrimination_query)
        d = self.data.query(self.rat_intensity_psychometric_query)
        if len(d)>0:
            self.__has_intensity_disc = True
            
            # Group by stim pairs T1/sp1 T2/sp2, and compute number of decisions in favor of each option
            d = d.groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])
            n_dec2 = d['answer'].sum()
            n_dec1 = d['correct'].count() - n_dec2
            self.n_int_dec1 = n_dec1.values
            self.n_int_dec2 = n_dec2.values
            
            # Compute the term of the binomial likelihood due to the combinatorial number of responses
            self.loglike_scale_int = scipy.special.gammaln(self.n_int_dec1+self.n_int_dec2)
            self.loglike_scale_int[self.n_int_dec1!=0]-= scipy.special.gammaln(self.n_int_dec1[self.n_int_dec1!=0])
            self.loglike_scale_int[self.n_int_dec2!=0]-= scipy.special.gammaln(self.n_int_dec2[self.n_int_dec2!=0])
            self.loglike_scale_int = np.sum(self.loglike_scale_int)
            
            # Get unique stim pair values
            stims = np.array(sorted(d.groups.keys()))
            self.disc_intensity_t = np.concatenate((stims[:,0],stims[:,2]), axis=0)
            self.disc_intensity_sp = np.concatenate((stims[:,1],stims[:,3]), axis=0)
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.disc_intensity_origins = self.disc_intensity_sp * np.sqrt(0.5*np.pi)
            self.disc_intensity_slopes = np.zeros_like(self.disc_intensity_origins)
            
            # Get the data of the vertical psychometric in rats
            d = self.data.query(self.rat_intensity_psychometric_query)
            
            # Compute the mean of responding in favor of stim2, for each stim pair
            bla = d.assign(prob2=(d['answer']).values)\
                .groupby(['duration1', 'intensity1', 'duration2', 'intensity2'])['prob2']\
                .apply(lambda x: Bootstraper.Mean(x, 1000, 0.5, return_std=True))
            means = bla.xs('mean', level=4)
            stds = bla.xs('std', level=4)
            
            # Convert the stim features and the data of the psychometrics from the means and stds pandas series to numpy arrays
            self.psych_intensity_t = []
            self.psych_intensity_sp = []
            self.psych_intensity = []
            self.psych_intensity_std = []
            for (t1, sp1, t2, sp2), psych in iteritems(means):
                self.psych_intensity_t.append([t1, t2])
                self.psych_intensity_sp.append([sp1, sp2])
                self.psych_intensity.append(psych)
                self.psych_intensity_std.append(stds[(t1, sp1, t2, sp2)])
            self.psych_intensity_t = np.array(self.psych_intensity_t)
            self.psych_intensity_sp = np.array(self.psych_intensity_sp)
            self.psych_intensity = np.array(self.psych_intensity)
            self.psych_intensity_std = np.array(self.psych_intensity_std)
            
            # Correct the std for zero variance cases (normally due to few trials)
            if np.all(self.psych_intensity_std==0):
                inds = self.psych_intensity_std==0
                self.psych_intensity_std[inds] = 0.5
            elif np.any(self.psych_intensity_std==0):
                inds = self.psych_intensity_std==0
                self.psych_intensity_std[inds] = np.min(self.psych_intensity_std[np.logical_not(inds)])
            
            # IMPORTANT NOTE, sp was defined as the mean of the half gaussian,
            # while the model expects the sigma of the half gaussian
            # this means that the 'origins' must be equal to sp * sqrt(0.5*pi)
            self.psych_intensity_origins = self.psych_intensity_sp * np.sqrt(0.5*np.pi)
            self.psych_intensity_slopes = np.zeros_like(self.psych_intensity_origins)
            
            self.logger.debug('Stored {0} trials of discrimination intensity experiment trials'.format(stims.shape[0]))
        else:
            self.logger.debug('No rat intensity discrimination trials available')
        
        self.__has_duration_data = self.__has_duration_slider or self.__has_duration_disc
        self.__has_intensity_data = self.__has_intensity_slider or self.__has_intensity_disc
    
    def set_method(self, method):
        valid_methods = self.get_valid_methods()
        if method is None:
            if all([e in self.experiments for e in ['slider', 'human_int_disc', 'human_dur_disc']])\
                and 'rat_disc' not in self.experiments:
                method = 'slider_disc_ls'
            elif not any([e in self.experiments for e in ['slider', 'human_int_disc', 'human_dur_disc']])\
                and 'rat_disc' in self.experiments:
                method = 'disc_ls'
            else:
                raise RuntimeError('Could not infer the fit method from the dataframe structure. Please explicitly provide the desired method.')
        method = method.lower()
        if not method in valid_methods:
            raise ValueError('Provided method "{0}" not in the valid methods list: {1}'.format(method, valid_methods))
        self.method = method
        self.__is_2alpha_method = '_2alpha' in self.method
        self.__is_nll_method = self.method.endswith('_nll')
        self.__is_ls_method = self.method.endswith('_ls')
        self.logger.debug('Setted self.method={0}'.format(self.method))
        self.logger.debug('Is a 2 alpha method? {0}'.format(self.__is_2alpha_method))
    
    def __setstate__(self, state):
        """
        self.__setstate__(state)
        
        Only used when loading from a pickle file to init the Fitter
        instance. Could also be used to copy one Fitter instance to
        another one.
        
        """
        self.__initlogger__()
        dataframe = pd.DataFrame()
        dataframe.__setstate__(state['dataframe'])
        self.set_data(dataframe)
        self.set_method(state['method'])
        if 'fits_path' in state.keys():
            self.fits_path = state['fits_path']
        else:
            self.fits_path = 'fits'
        self.optimizer = state['optimizer']
        self.suffix = state['suffix']
        
        if '_start_point' in state.keys():
            self._start_point = state['_start_point']
        if '_bounds' in state.keys():
            self._bounds = state['_bounds']
        if '_fitted_parameters' in state.keys():
            self._fitted_parameters = state['_fitted_parameters']
        if '_fixed_parameters' in state.keys():
            self._fixed_parameters = state['_fixed_parameters']
        if 'fit_arguments' in state.keys():
            self.set_fit_arguments(state['fit_arguments'])
        if 'fit_output' in state.keys():
            self._fit_output = state['fit_output']
        self.__fit_internals__ = None
    
    def set_fixed_parameters(self, fixed_parameters=None):
        """
        self.set_fixed_parameters(fixed_parameters={})
        
        Set the fixed_parameters by merging the supplied fixed_parameters
        input dict with the default fixed_parameters. Note that these
        fixed_parameters need to be sanitized before being used to init
        the minimizer. Also note that this method sets the unsanitized fitted
        parameters as the complement of the fixed_parameters keys.
        
        Input:
            fixed_parameters: A dict whose keys are the parameter names and
                the values are the corresponding parameter fixed values
        
        """
        self.logger.debug('Setting fixed parameters with input {0}'.format(prettyformat(fixed_parameters)))
        if fixed_parameters is None:
            fixed_parameters = self.default_fixed_parameters()
        fittable_parameters = self.get_fittable_parameters()
        self._fixed_parameters = fixed_parameters.copy()
        self._fitted_parameters = []
        for par in fittable_parameters:
            if par not in self._fixed_parameters.keys():
                self._fitted_parameters.append(par)
        self.logger.debug('Setted Fitter fixed_parameters = %s', prettyformat(self._fixed_parameters))
        self.logger.debug('Setted Fitter fitted_parameters = %s', prettyformat(self._fitted_parameters))
    
    def set_start_point(self, start_point={}):
        """
        self.set_start_point(start_point={})
        
        Set the start_point by merging the supplied start_point input dict with
        the default start_point. Note that this start_point need to be sanitized
        before being used to init the minimizer.
        
        Input:
            start_point: A dict whose keys are the parameter names and
                the values are the corresponding parameter starting value
        
        """
        defaults = self.default_start_point()
        defaults.update(start_point)
        self._start_point = defaults
        self.logger.debug('Setted Fitter start_point = %s', prettyformat(self._start_point))
    
    def set_bounds(self,bounds={}):
        """
        self.set_bounds(bounds={})
        
        Set the bounds by merging the supplied bounds input dict with
        the default bounds. Note that these bounds need to be sanitized
        before being used to init the minimizer.
        
        Input:
            bounds: A dict whose keys are the parameter names and
                the values are a list with the [low_bound,up_bound] values.
        
        """
        defaults = self.default_bounds()
        defaults.update(bounds)
        self._bounds = defaults
        self.logger.debug('Setted Fitter bounds = %s', prettyformat(self._bounds))
    
    def set_optimizer_kwargs(self, optimizer_kwargs={}):
        """
        self.set_optimizer_kwargs(optimizer_kwargs={})
        
        Set the optimizer_kwargs by merging the supplied optimizer_kwargs
        input dict with the default optimizer_kwargs. Note that these
        kwargs do not need any further sanitation that could depend on
        the Fitter's method.
        
        """
        defaults = self.default_optimizer_kwargs()
        defaults.update(optimizer_kwargs)
        self.optimizer_kwargs = defaults
        self.logger.debug('Setted Fitter optimizer_kwargs = %s', prettyformat(self.optimizer_kwargs))
    
    def set_fit_arguments(self, fit_arguments):
        """
        self.set_fit_arguments(fit_arguments)
        
        Set the instance's fit_arguments and sanitized: start point, bounds
        optimizer_kwargs, fitted parameters and fixed parameters.
        
        Input:
        fit_argument: A dict with keys: start_point, bounds, optimizer_kwargs
            fitted_parameters and fixed_parameters
        
        """
        self._fit_arguments = fit_arguments
        self.start_point = fit_arguments['start_point']
        self.bounds = fit_arguments['bounds']
        self.optimizer_kwargs = fit_arguments['optimizer_kwargs']
        self.fitted_parameters = fit_arguments['fitted_parameters']
        self.fixed_parameters = fit_arguments['fixed_parameters']
    
    # Getters
    def get_valid_methods(self):
        valid_methods = ['slider_disc', 'disc', 'slider']
        valid_methods.extend([m+'_2alpha' for m in valid_methods])
        valid_methods = [m+'_nll' for m in valid_methods] + [m+'_ls' for m in valid_methods]
        return valid_methods
    
    def transform_parameters(self, parameters, transform_taus=False, split_duration_intensity=False):
        if not self.__is_2alpha_method:
            if not parameters['intensity_alpha'] is None:
                parameters['duration_alpha'] = parameters['intensity_alpha']
        #~ dback_var = parameters.pop('duration_background_var_eff', None)
        #~ if dback_var is not None:
            #~ if dback_var>0:
                #~ parameters['duration_background_var'] = dback_var * parameters['duration_tau']
            #~ else:
                #~ parameters['duration_background_var'] = 0.
        #~ iback_var = parameters.pop('intensity_background_var_eff', None)
        #~ if iback_var is not None:
            #~ if iback_var>0:
                #~ parameters['intensity_background_var'] = iback_var * parameters['intensity_tau']
            #~ else:
                #~ parameters['intensity_background_var'] = 0.
        if transform_taus:
            if self.__has_duration_data:
                duration_tau_inv = 1./parameters.pop('duration_tau')
                parameters['duration_C_inv'] = duration_tau_inv/parameters['duration_leak'] if parameters['duration_leak']!=0 else duration_tau_inv
            else:
                parameters['duration_C_inv'] = None
            if self.__has_intensity_data:
                intensity_tau_inv = 1./parameters.pop('intensity_tau')
                parameters['intensity_C_inv'] = intensity_tau_inv/parameters['intensity_leak'] if parameters['intensity_leak']!=0 else intensity_tau_inv
            else:
                parameters['intensity_C_inv'] = None
            if split_duration_intensity:
                duration_parameters = {k: parameters[v] for k, v in iteritems(self.duration_parameters) if (not 'perror' in k) and not (v=='duration_tau' or v=='intensity_tau')}
                intensity_parameters = {k: parameters[v] for k, v in iteritems(self.intensity_parameters) if (not 'perror' in k) and not (v=='duration_tau' or v=='intensity_tau')}
                duration_parameters['C_inv'] = parameters['duration_C_inv']
                intensity_parameters['C_inv'] = parameters['intensity_C_inv']
                duration_error_rates = {k: parameters[v] for k, v in iteritems(self.duration_parameters) if 'perror' in k}
                intensity_error_rates = {k: parameters[v] for k, v in iteritems(self.intensity_parameters) if 'perror' in k}
                parameters = (duration_parameters, intensity_parameters, duration_error_rates, intensity_error_rates)
        elif split_duration_intensity:
            duration_parameters = {k: parameters[v] for k, v in iteritems(self.duration_parameters) if not 'perror' in k}
            intensity_parameters = {k: parameters[v] for k, v in iteritems(self.intensity_parameters) if not 'perror' in k}
            duration_error_rates = {k: parameters[v] for k, v in iteritems(self.duration_parameters) if 'perror' in k}
            intensity_error_rates = {k: parameters[v] for k, v in iteritems(self.intensity_parameters) if 'perror' in k}
            parameters = (duration_parameters, intensity_parameters, duration_error_rates, intensity_error_rates)
        
        #~ print(prettyformat(parameters))
        return parameters
    
    def get_parameters_dict(self, transform_taus=False, split_duration_intensity=False):
        """
        self.get_parameters_dict():
        
        Get a dict with all the model's parameter names as keys. The
        the values of the fixed parameters are taken from 
        self.get_fixed_parameters() and the rest of the parameter values
        are taken from the self.get_start_point() method.
        
        """
        parameters = self.get_fixed_parameters().copy()
        start_point = self.get_start_point()
        for fp in self.get_fitted_parameters():
            parameters[fp] = start_point[fp]
        return self.transform_parameters(parameters, transform_taus=transform_taus, split_duration_intensity=split_duration_intensity)
    
    def get_parameters_dict_from_array(self, x, transform_taus=False, split_duration_intensity=False):
        """
        self.get_parameters_dict_from_array(x):
        
        Get a dict with all the model's parameter names as keys. The
        the values of the fixed parameters are taken from the 
        sanitized fixed parameters and the fitted parameters' values
        are taken from the supplied array. The array elements must have
        the same order as the sanitized fitted parameters.
        
        """
        parameters = self.fixed_parameters.copy()
        try:
            for index,key in enumerate(self.fitted_parameters):
                parameters[key] = x[index]
        except IndexError:
            parameters[self.fitted_parameters[0]] = x
        return self.transform_parameters(parameters, transform_taus=transform_taus, split_duration_intensity=split_duration_intensity)
    
    def get_parameters_dict_from_fit_output(self,fit_output=None, transform_taus=False, split_duration_intensity=False):
        """
        self.get_parameters_dict_from_fit_output(fit_output=None):
        
        Get a dict with all the model's parameter names as keys. The
        the values of the fixed parameters are taken from the 
        sanitized fixed parameters and the fitted parameters' values
        are taken from the fit_output. If fit_output is None, the
        instance's fit_output is used instead.
        
        """
        if fit_output is None:
            fit_output = self._fit_output
        parameters = self.fixed_parameters.copy()
        parameters.update(fit_output[0])
        return self.transform_parameters(parameters, transform_taus=transform_taus, split_duration_intensity=split_duration_intensity)
    
    def get_fixed_parameters(self):
        """
        self.get_fixed_parameters()
        
        This function first attempts to return the sanitized fixed parameters.
        If this fails (most likely because the fixed parameters were not sanitized)
        it attempts to return the setted fixed parameters.
        If this fails, because the fixed parameters were not set, it returns the
        default fixed parameters.
        This function always returns a dict that has the parameter names
        as keys and as values floats with the parameter's fixed value.
        
        """
        try:
            return self.fixed_parameters
        except:
            try:
                return self._fixed_parameters
            except:
                return self.default_fixed_parameters()
    
    def get_fitted_parameters(self):
        """
        self.get_fitted_parameters()
        
        This function first attempts to return the sanitized fitted parameters.
        If this fails (most likely because the fitted parameters were not sanitized)
        it attempts to return the setted fitted parameters.
        If this fails, because the fitted parameters were not set, it returns the
        default fitted parameters.
        This function always returns a list of parameter names.
        
        """
        try:
            return self.fitted_parameters
        except:
            try:
                return self._fitted_parameters
            except:
                return [p for p in self.get_fittable_parameters() if p not in self.default_fixed_parameters().keys()]
    
    def get_start_point(self):
        """
        self.get_start_point()
        
        This function first attempts to return the sanitized start point, which
        is an array of shape (2,len(self.fitted_parameters)).
        If this fails (most likely because the start point was not sanitized)
        it attempts to return the setted parameter start point dictionary.
        This dict has the parameter names as keys and as values floats
        with the parameter's value.
        If this fails, because the start point was not set, it returns the
        default start point.
        
        """
        try:
            return self.start_point
        except:
            try:
                return self._start_point
            except:
                return self.default_start_point()
    
    def get_bounds(self):
        """
        self.get_bounds()
        
        This function first attempts to return the sanitized bounds, which
        is an array of shape (2,len(self.fitted_parameters)).
        If this fails (most likely because the bounds were not sanitized)
        it attempts to return the setted parameter bound dictionary.
        This dict has the parameter names as keys and as values lists
        with [low_bound,high_bound] values.
        If this fails, because the bounds were not set, it returns the
        default bounds.
        
        """
        try:
            return self.bounds
        except:
            try:
                return self._bounds
            except:
                return self.default_bounds()
    
    def __getstate__(self):
        """
        self.__getstate__()
        
        Get the Fitter instance's state dictionary. This function is used
        when pickling the Fitter.
        """
        state = {'dataframe':self.data.__getstate__(),
                 'method':self.method,
                 'optimizer':self.optimizer,
                 'suffix':self.suffix,
                 'fits_path':self.fits_path}
        if hasattr(self,'_start_point'):
            state['_start_point'] = self._start_point
        if hasattr(self,'_bounds'):
            state['_bounds'] = self._bounds
        if hasattr(self,'_fitted_parameters'):
            state['_fitted_parameters'] = self._fitted_parameters
        if hasattr(self,'_fixed_parameters'):
            state['_fixed_parameters'] = self._fixed_parameters
        if hasattr(self,'_fit_arguments'):
            state['fit_arguments'] = self._fit_arguments
        if hasattr(self,'_fit_output'):
            state['fit_output'] = self._fit_output
        return state
    
    def get_fittable_parameters(self):
        """
        self.get_fittable_parameters()
        
        Returns Stimulator().get_parameter_names()+Leaky().get_parameter_names()
        """
        
        fittable_parameters = ['duration_leak',
                               'duration_tau',
                               'duration_x0',
                               'duration_var0',
                               'duration_background_var',
                               'duration_background_mean',
                               'duration_alpha',
                               'duration_low_perror',
                               'duration_high_perror',
                               'intensity_leak',
                               'intensity_tau',
                               'intensity_x0',
                               'intensity_var0',
                               'intensity_background_var',
                               'intensity_background_mean',
                               'intensity_alpha',
                               'intensity_low_perror',
                               'intensity_high_perror',
                               'adaptation_amplitude',
                               'adaptation_baseline',
                               'adaptation_tau_inv']
        return fittable_parameters
    
    def get_key(self,merge=None):
        """
        self.get_key(merge=None)
        
        This function returns a string that can be used as a Fitter_plot_handler's
        key.
        The returned key depends on the merge input as follows.
        If merge=None: key={subject_type}_subject_{self.subjects}_session_{self.sessions}
        If merge='subjects': key={subject_type}_session_{self.sessions}
        If merge='sessions': key={subject_type}_subject_{self.subjects}
        If merge='all': key={subject_type}
        
        """
        subject_type = 'Rats' if 'rat_disc' in self.experiments else 'Humans'
        subject = '[{}]'.format('-'.join([str(s).strip() for s in self.subjects]))
        session = '[{}]'.format(' '.join([str(s).strip() for s in sorted(self.sessions)]))
        if merge is None:
            key = "{subject_type}_subject_{subject}_session_{session}".format(
                    subject_type=subject_type, subject=subject, session=session)
        elif merge=='subjects':
            key = "{subject_type}_session_{session}".format(
                    subject_type=subject_type, session=session)
        elif merge=='sessions':
            key = "{subject_type}_subject_{subject}".format(
                    subject_type=subject_type, subject=subject)
        elif merge=='all':
            key = "{subject_type}".format(subject_type=subject_type)
        else:
            raise ValueError('Unknown merge option {0}. Available values are None, "subjects", "sessions" and "all"'.format(merge))
        return key
    
    def get_save_file_name(self):
        """
        self.get_save_file_name()
        
        An alias for the package's function Fitter_filename. This method
        simply returns the output of the call:
        Fitter_filename(name=self.subjectSession.get_name(),
                session=self.subjectSession.get_session(),
                optimizer=self.optimizer,suffix=self.suffix)
        
        """
        return Fitter_filename(name=self.subjects,
                session=self.sessions, method=self.method,optimizer=self.optimizer,suffix=self.suffix,
                fits_path=self.fits_path)
    
    def get_jacobian_dx(self):
        """
        self.get_jacobian_dx()
        
        This function returns a dict where the keys are fittable parameter
        names and the values are the parameter displacements that should
        be used to compute the numerical jacobian.
        Regretably, we cannot compute an analytical form of the derivative
        of the first passage time probability density in decision_model.DecisionModel.rt
        and this forces us to use a numerical approximation of the
        jacobian. The values returned by this function are only used
        with the scipy's optimize methods: CG, BFGS, Newton-CG, L-BFGS-B,
        TNC, SLSQP, dogleg and trust-ncg.
        
        """
        jac_dx = {par: 1e-11 for par in self.get_fittable_parameters()}
        return jac_dx
    
    def _get_Fitter_plot_handler_init_data(self, fit_output=None):
        duration_parameters, intensity_parameters, duration_error_rates, intensity_error_rates = \
            self.get_parameters_dict_from_fit_output(fit_output, transform_taus=True, split_duration_intensity=True)
        
        slider = {'duration_mean': None,
                  'duration_var': None,
                  'duration_n': None,
                  'duration_t': None,
                  'duration_sp': None,
                  'intensity_mean': None,
                  'intensity_var': None,
                  'intensity_n': None,
                  'intensity_t': None,
                  'intensity_sp': None}
        if self.__has_duration_slider:
            slider_duration_mean, slider_duration_var = \
                Leaky._theoretical_generalstim(t=self.slider_duration_t,
                                               origin=self.slider_duration_origins,
                                               slope=self.slider_duration_slopes,
                                               **duration_parameters)
            
            slider_duration_n = np.empty_like(self.slider_duration_t)
            for ind in range(len(self.slider_duration_t)):
                slider_duration_n[ind] = np.sum((self.inv_dur_stims_inds==ind).astype(int))
            slider.update({'duration_mean': slider_duration_mean,
                           'duration_var': slider_duration_var,
                           'duration_n': slider_duration_n,
                           'duration_t': self.slider_duration_t,
                           'duration_sp': self.slider_duration_sp
                          })
        if self.__has_intensity_slider:
            slider_intensity_mean, slider_intensity_var = \
                Leaky._theoretical_generalstim(t=self.slider_intensity_t,
                                               origin=self.slider_intensity_origins,
                                               slope=self.slider_intensity_slopes,
                                               **intensity_parameters)
            slider_intensity_n = np.empty_like(self.slider_intensity_t)
            for ind in range(len(self.slider_intensity_t)):
                slider_intensity_n[ind] = np.sum((self.inv_int_stims_inds==ind).astype(int))
            
            slider.update({'intensity_mean': slider_intensity_mean,
                           'intensity_var': slider_intensity_var,
                           'intensity_n': slider_intensity_n,
                           'intensity_t': self.slider_intensity_t,
                           'intensity_sp': self.slider_intensity_sp
                          })
        
        if all([v is None for v in slider.values()]):
            slider = None
        
        disc = {'prob_duration': None,
                'psych_duration_t': None,
                'psych_duration_sp': None,
                'psych_duration_n': None,
                'prob_intensity': None,
                'psych_intensity_t': None,
                'psych_intensity_sp': None,
                'psych_intensity_n': None,
               }
        
        if self.__has_duration_disc:
            disc_duration_mean, disc_duration_var = \
                Leaky._theoretical_generalstim(t=self.psych_duration_t,
                                               origin=self.psych_duration_origins,
                                               slope=self.psych_duration_slopes,
                                               **duration_parameters)
            
            mean1_dur = disc_duration_mean[:, 0]
            mean2_dur = disc_duration_mean[:, 1]
            var1_dur = disc_duration_var[:, 0]
            var2_dur = disc_duration_var[:, 1]
            prob_duration = prob2AFC(mean1_dur, var1_dur, mean2_dur, var2_dur, **duration_error_rates)
            if 'rat_disc' in self.experiments:
                psych_duration_n = self.data.query(self.rat_duration_psychometric_query)\
                                        .groupby(['duration1', 'intensity1', 'duration2', 'intensity2']).agg('count')['subject'].values.flatten()
            else:
                psych_duration_n = self.data.query(self.human_duration_psychometric_query)\
                                        .groupby(['duration1', 'intensity1', 'duration2', 'intensity2']).agg('count')['subject'].values.flatten()
            
            disc.update({'prob_duration': prob_duration,
                         'psych_duration_t': self.psych_duration_t,
                         'psych_duration_sp': self.psych_duration_sp,
                         'psych_duration_n': psych_duration_n,
                        })
        
        if self.__has_intensity_disc:
            disc_intensity_mean, disc_intensity_var = \
                Leaky._theoretical_generalstim(t=self.psych_intensity_t,
                                               origin=self.psych_intensity_origins,
                                               slope=self.psych_intensity_slopes,
                                               **intensity_parameters)
            
            mean1_int = disc_intensity_mean[:, 0]
            mean2_int = disc_intensity_mean[:, 1]
            var1_int = disc_intensity_var[:, 0]
            var2_int = disc_intensity_var[:, 1]
            prob_intensity = prob2AFC(mean1_int, var1_int, mean2_int, var2_int, **intensity_error_rates)
            if 'rat_disc' in self.experiments:
                psych_intensity_n = self.data.query(self.rat_intensity_psychometric_query)\
                                        .groupby(['duration1', 'intensity1', 'duration2', 'intensity2']).agg('count')['subject'].values.flatten()
            else:
                psych_intensity_n = self.data.query(self.human_intensity_psychometric_query)\
                                        .groupby(['duration1', 'intensity1', 'duration2', 'intensity2']).agg('count')['subject'].values.flatten()
            
            disc.update({'prob_intensity': prob_intensity,
                         'psych_intensity_t': self.psych_intensity_t,
                         'psych_intensity_sp': self.psych_intensity_sp,
                         'psych_intensity_n': psych_intensity_n,
                        })
        
        if all([v is None for v in disc.values()]):
            disc = None
        
        obj = {'dataframe': self.data,
               'slider': slider,
               'disc': disc
              }
        return obj
    
    def get_Fitter_plot_handler(self, fit_output=None):
        return Fitter_plot_handler({self.get_key(): self._get_Fitter_plot_handler_init_data(fit_output)})
    
    def as_matlab_format(self, get_fit_result=True, get_fixed_parameters=True,
                         get_list_of_fitted_parameters=True, get_merit_value=True,
                         get_fit_predictions=True, get_subject_data=True):
        output = {}
        if get_fit_result:
            temp = self.get_parameters_dict_from_fit_output(transform_taus=False, split_duration_intensity=False)
            params = {}
            for k, v in iteritems(temp):
                if isinstance(v, np.ndarray):
                    v = v.astype(np.float64)
                elif v is None:
                    v = np.nan
                else:
                    v = float(v)
                params[str(k)] = v
            output[str('parameter_values')] = params
            output[str('stop_condition')] = stringify_keys(self._fit_output[-1])
        if get_fixed_parameters:
            temp = self.get_fixed_parameters()
            fixed = {}
            for k, v in iteritems(temp):
                if isinstance(v, np.ndarray):
                    v = v.astype(np.float64)
                elif v is None:
                    v = np.nan
                else:
                    v = float(v)
                fixed[str(k)] = v
            output[str('fixed_parameters')] = fixed
        if get_list_of_fitted_parameters:
            fitted = [str(p) for p in self.get_fitted_parameters()]
            output[str('fitted_parameters')] = fitted
        if get_merit_value:
            try:
                temp = float(self._fit_output[1])
            except:
                temp = np.nan
            merit_type = 'Negative Log Likelihood' if '_nll' in self.method else 'Least Squared Difference'
            output[str('merit')] = {str('value'): temp, str('merit_type'): merit_type}
        if get_fit_predictions:
            obj = self._get_Fitter_plot_handler_init_data()
            del obj['dataframe']
            temp = {}
            for k1, v1 in iteritems(obj):
                if v1 is not None:
                    temp[str(k1)] = {}
                    for k2, v2 in iteritems(v1):
                        if v2 is None:
                            v2 = np.nan
                        elif isinstance(v2, np.ndarray):
                            v2 = v2.astype(np.float64)
                        else:
                            v2 = float(v2)
                        temp[str(k1)][str(k2)] = v2
            output[str('predictions')] = temp
        if get_subject_data:
            subj_data = {}
            # Duration slider
            temp = self.data.query('experiment=="slider" and task==1').drop(columns=['experiment', 'task'])
            if not temp.empty:
                subj_data[str('duration_slider')] = {str('data'): temp.values.astype(np.float64),
                                                     str('colnames'): list(temp.columns)}
            
            # Intensity slider
            temp = self.data.query('experiment=="slider" and task==0').drop(columns=['experiment', 'task'])
            if not temp.empty:
                subj_data[str('intensity_slider')] = {str('data'): temp.values.astype(np.float64),
                                                str('colnames'): list(temp.columns)}
            
            # Duration discrimination
            temp = self.data.query('experiment=="human_dur_disc" or (experiment=="rat_disc" and task==1)').drop(columns=['experiment', 'task'])
            if not temp.empty:
                subj_data[str('duration_discrimination')] = {str('data'): temp.values.astype(np.float64),
                                                        str('colnames'): list(temp.columns)}
            
            # Intensity discrimination
            temp = self.data.query('experiment=="human_int_disc" or (experiment=="rat_disc" and task==0)').drop(columns=['experiment', 'task'])
            if not temp.empty:
                subj_data[str('intensity_discrimination')] = {str('data'): temp.values.astype(np.float64),
                                                         str('colnames'): list(temp.columns)}
            output[str('measured_data')] = subj_data
        return {str(self.get_key()): output}
    
    # Defaults
    
    def default_fixed_parameters(self):
        """
        self.default_fixed_parameters()
        
        Returns the default parameter fixed_parameters. By default no
        parameter is fixed so it simply returns an empty dict.
        
        """
        return {'adaptation_amplitude': 1., 'adaptation_baseline': 1., 'adaptation_tau_inv': 0.,
                'duration_var0': 0., 'intensity_var0': 0.,
                'intensity_background_mean': 0.}
    
    def default_start_point(self):
        """
        self.default_start_point()
        
        Returns the default parameter start_point that depend on
        the subjectSession's response time distribution and
        performance. This function is very fine tuned to get a good
        starting point for every parameter automatically.
        This function returns a dict with the parameter names as keys and
        the corresponding start_point floating point as values.
        
        """
        default_sp = {'duration_leak': 2.,
                      'duration_tau': 520.40,
                      'duration_x0': 0.,
                      'duration_var0': 0.,
                      'duration_background_var': 1425.,
                      'duration_background_mean': 22.,
                      'duration_alpha': 0.8,
                      'duration_low_perror': 0.,
                      'duration_high_perror': 0.,
                      'intensity_leak': 0.98,
                      'intensity_tau': 68.03,
                      'intensity_x0': 0.,
                      'intensity_var0': 0.,
                      'intensity_background_var': 55.,
                      'intensity_background_mean': 0.,
                      'intensity_alpha': 0.8,
                      'intensity_low_perror': 0.,
                      'intensity_high_perror': 0.,
                      'adaptation_amplitude': 1.,
                      'adaptation_baseline': 1.,
                      'adaptation_tau_inv': 0.}
        
        return default_sp
    
    def default_bounds(self):
        """
        self.default_bounds()
        
        Returns the default parameter bounds that depend on
        whether the decision bounds are invariant or not.
        This function returns a dict with the parameter names as keys and
        the corresponding [lower_bound, upper_bound] list as values.
        
        """
        default_bs = {'duration_leak': [1e-6, 100.],
                      'duration_tau': [2e-4, 1e4],
                      'duration_x0': [-1e3, 1e3],
                      'duration_var0': [0., 100.],
                      'duration_background_var': [0., 1e5],
                      'duration_background_mean': [0., 100.],
                      'duration_alpha': [0.05, 1.],
                      'duration_low_perror': [0., 0.5],
                      'duration_high_perror': [0., 0.5],
                      'intensity_leak': [1e-6, 100.],
                      'intensity_tau': [2e-4, 1e4],
                      'intensity_x0': [-1e3, 1e3],
                      'intensity_var0': [0., 100.],
                      'intensity_background_var': [0., 1e5],
                      'intensity_background_mean': [0., 100.],
                      'intensity_alpha': [0.05, 1.],
                      'intensity_low_perror': [0., 0.5],
                      'intensity_high_perror': [0., 0.5],
                      'adaptation_amplitude': [0., 1.],
                      'adaptation_baseline': [0., 1.],
                      'adaptation_tau_inv': [0., 1.]}
        return default_bs
    
    def default_optimizer_kwargs(self):
        """
        self.default_optimizer_kwargs()
        
        Returns the default optimizer_kwargs that depend on the optimizer
        attribute
        
        """
        if self.optimizer=='cma':
            return {'restarts':4,'restart_from_best':'False'}
        elif self.optimizer=='basinhopping':
            return {'stepsize':0.25, 'minimizer_kwargs':{'method':'Nelder-Mead'},'T':10.,'niter':100,'interval':10}
        elif self.optimizer in ['Brent','Bounded','Golden']: # Scalar minimize
            return {'disp': False, 'maxiter': 1000, 'repetitions': 10}
        else: # Multivariate minimize
            return {'disp': False, 'maxiter': 1000, 'maxfev': 10000, 'repetitions': 1}
    
    # Main fit method
    def fit(self, fixed_parameters=None, start_point={}, bounds={}, optimizer_kwargs={}, fit_arguments=None):
        """
        self.fit(fixed_parameters=None,start_point={},bounds={},optimizer_kwargs={},fit_arguments=None)
        
        Main Fitter function that executes the optimization procedure
        specified by the Fitter instance's optimizer attribute.
        This methods sets the fixed_parameters, start_point, bounds and
        optimizer_kwargs using the ones supplied in the input or using
        another Fitter instance's fit_arguments attribute.
        This method also sanitizes the fixed_parameters depending on
        the selected fitting method, init's the minimizer and returns
        the sanitized minimization output.
        
        For the detailed output form refer to the method sanitize_fmin_output
        
        """
        self.logger.debug('Entered fit method with input:\nfixed_parameters={0},\nstart_point={1},\nbounds={2},\noptimizer_kwargs={3},\nfit_arguments={4}'\
                          .format(prettyformat(fixed_parameters),
                                  prettyformat(start_point),
                                  prettyformat(bounds),
                                  prettyformat(optimizer_kwargs),
                                  prettyformat(fit_arguments)))
        if fit_arguments is None:
            self.set_fixed_parameters(fixed_parameters)
            self.set_start_point(start_point)
            self.set_bounds(bounds)
            self.fixed_parameters, self.fitted_parameters, self.start_point, self.bounds = \
                self.sanitize_parameters_x0_bounds()
            
            self.set_optimizer_kwargs(optimizer_kwargs)
            self._fit_arguments = {'fixed_parameters': self.fixed_parameters,
                                   'fitted_parameters': self.fitted_parameters,
                                   'start_point': self.start_point,
                                   'bounds': self.bounds,
                                   'optimizer_kwargs': self.optimizer_kwargs}
        else:
            self.set_fit_arguments(fit_arguments)
        
        minimizer = self.init_minimizer(self.start_point, self.bounds, self.optimizer_kwargs)
        self.logger.debug('Setting merit_function to self.merit_{0}'.format(self.method))
        merit_function = eval('self.merit_{}'.format(self.method.replace('_2alpha', '')))
        self.__fit_internals__ = None
        self._fit_output = minimizer(merit_function)
        self.__fit_internals__ = None
        self.logger.info('Finished fitting round with output:\n{0}'.format(prettyformat(self._fit_output)))
        return self._fit_output
    
    # Savers
    def save(self):
        """
        self.save()
        
        Dumps the Fitter instances state to a pkl file.
        The Fitter's state is return by method self.__getstate__().
        The used pkl's file name is given by self.get_save_file_name().
        
        """
        self.logger.debug('Fitter state that will be saved = "{0}"'.format(prettyformat(self.__getstate__())))
        if not hasattr(self,'_fit_output'):
            raise ValueError('The Fitter instance has not performed any fit and still has no _fit_output attribute set')
        self.logger.info('Saving Fitter state to file "{0}"'.format(self.get_save_file_name()))
        f = open(self.get_save_file_name(),'wb')
        pickle.dump(self, f, PICKLE_PROTOCOL)
        f.close()
    
    def savemat(self, fname, varname=None, **kwargs):
        state = self.as_matlab_format(**kwargs)
        if varname is not None:
            state = {str(varname): state}
        self.logger.debug('Fitter state that will be saved = {0}'.format(prettyformat(state)))
        self.logger.info('Saving Fitter state to file "%s" in matlab format', fname)
        self.logger.info('Data will be saved with the variable name = {0}'.format(varname))
        scipy.io.savemat(fname, state)
    
    # Sanitizers
    def sanitize_parameters_x0_bounds(self):
        """
        fitter.sanitize_parameters_x0_bounds()
        
        Some of the methods used to compute the merit assume certain
        parameters are fixed while others assume they are not. Furthermore
        some merit functions do not use all the parameters.
        This function allows the users to keep the flexibility of defining
        a single set of fixed parameters without worrying about the
        method specificities. The function takes the fixed_parameters,
        start_point and bounds specified by the user, and arranges them
        correctly for the specified merit method.
        
        Output:
        fixed_parameters,fitted_parameters,sanitized_start_point,sanitized_bounds
        
        fixed_parameters: A dict of parameter names as keys and their fixed values
        fitted_parameters: A list of the fitted parameters
        sanitized_start_point: A numpy ndarray with the fitted_parameters
                               starting point
        sanitized_bounds: The fitted parameter's bounds. The specific
                          format depends on the optimizer.
        
        """
        # Get fixed parameters that were set by the user and replace the None's
        _start_point = self._start_point.copy()
        _bounds = self._bounds.copy()
        fixed_parameters = self._fixed_parameters.copy()
        for par in fixed_parameters.keys():
            if fixed_parameters[par] is None:
                fixed_parameters[par] = self._start_point[par]
        
        # Get method dependent fixed parameters
        method_dependent_parameters = {}
        if self.__is_2alpha_method:
            method_dependent_parameters['duration_background_mean'] = 0.
        else:
            method_dependent_parameters['duration_alpha'] = None
        if 'slider' not in self.method or not self.__has_duration_slider:
            method_dependent_parameters['duration_leak'] = 1.
        if 'slider' not in self.method or not self.__has_intensity_slider:
            method_dependent_parameters['intensity_leak'] = 1.
        if 'disc' not in self.method or not self.__has_duration_disc:
            method_dependent_parameters['duration_background_var'] = 0.
            method_dependent_parameters['duration_low_perror'] = 0.
            method_dependent_parameters['duration_high_perror'] = 0.
        if 'disc' not in self.method or not self.__has_intensity_disc:
            method_dependent_parameters['intensity_background_var'] = 0.
            method_dependent_parameters['intensity_low_perror'] = 0.
            method_dependent_parameters['intensity_high_perror'] = 0.
        
        # Update the method dependent fixed parameters by taking into account if there are experiments in duration or intensity modalities
        fittable_parameters = self.get_fittable_parameters()
        # Must look at intensity experiment data first because of the way in which 2alpha and single_alpha methods handle the intensity_alpha
        if not (self.__has_intensity_slider or self.__has_intensity_disc):
            self.logger.debug("No intensity experimental data, will set intensity parameters to None")
            for par in [par for par in fittable_parameters if par.startswith('intensity_')]:
                if par not in method_dependent_parameters.keys():
                    if par=='intensity_alpha' and not self.__is_2alpha_method:
                        self.logger.debug("Method assumes shared alpha! "
                        "Special care is taken in handling if intensity_alpha and duration_alpha are fixed and the start point value")
                        method_dependent_parameters.pop('duration_alpha', None)
                        if 'duration_alpha' not in fixed_parameters.keys():
                            if not _start_point['intensity_alpha'] is None:
                                _start_point['duration_alpha'] = _start_point['intensity_alpha']
                            if not _bounds['intensity_alpha'] is None:
                                _bounds['duration_alpha'] = _bounds['intensity_alpha']
                    method_dependent_parameters[par] = None
        if not (self.__has_duration_slider or self.__has_duration_disc):
            self.logger.debug("No duration experimental data, will set duration parameters to None")
            for par in [par for par in fittable_parameters if par.startswith('duration_')]:
                if par not in method_dependent_parameters.keys():
                    method_dependent_parameters[par] = None
        
        # Get the sanitized fixed_parameters, and infer the fitted_parameters, x0 and bounds
        fitted_parameters = []
        sp = []
        bs = []
        for par in self._fitted_parameters:
            if par in method_dependent_parameters.keys():
                fixed_parameters[par] = method_dependent_parameters[par]
                continue
            fitted_parameters.append(par)
            sp.append(_start_point[par])
            bs.append(_bounds[par])
            if par in fixed_parameters.keys():
                del fixed_parameters[par]
        sanitized_start_point = np.array(sp)
        sanitized_bounds = list(np.array(bs).T)
        
        # Handle optimizer conflicts
        if len(fitted_parameters)==1 and self.optimizer=='cma':
            warnings.warn('CMA is unsuited for optimization of single dimensional parameter spaces. Optimizer was changed to Nelder-Mead')
            self.optimizer = 'Nelder-Mead'
        elif len(fitted_parameters)>1 and self.optimizer in ['Brent','Bounded','Golden']:
            raise ValueError(('Brent, Bounded and Golden optimizers are only available for scalar '
                              'functions. However, {0} parameters are being fitted. Please '
                              'review the optimizer').format(len(fitted_parameters)))
        if not (self.optimizer=='cma' or self.optimizer=='basinhopping'):
            sanitized_bounds = [(lb,ub) for lb,ub in zip(sanitized_bounds[0],sanitized_bounds[1])]
        
        self.logger.debug('Sanitized fixed parameters = {0}'.format(prettyformat(fixed_parameters)))
        self.logger.debug('Sanitized fitted parameters = {0}'.format(prettyformat(fitted_parameters)))
        self.logger.debug('Sanitized start_point = {0}'.format(prettyformat(sanitized_start_point)))
        self.logger.debug('Sanitized bounds = {0}'.format(prettyformat(sanitized_bounds)))
        if len(fitted_parameters)==0:
            raise RuntimeError('No parameters left to fit!!')
        
        return (fixed_parameters,fitted_parameters,sanitized_start_point,sanitized_bounds)
    
    def sanitize_fmin_output(self, output, package='cma'):
        """
        self.sanitize_fmin_output(output, package='cma')
        
        The cma package returns the fit output in one format, the
        scipy package returns it in a completely different way, and the
        repeat_minimize method has a slightly different output format.
        This method returns the fit output in a common format:
        It returns a tuple out
        
        out[0]: A dictionary with the fitted parameter names as keys and
                the values being the best fitting parameter value.
        out[1]: Merit function value
        out[2]: Number of function evaluations
        out[3]: Overall number of function evaluations (in the cma
                package, these can be more if there is noise handling)
        out[4]: Number of iterations
        out[5]: Mean of the sample of solutions
        out[6]: Std of the sample of solutions
        out[7]: Termination condition dictionary. The contents depend on the
                package used during the optimization.
        
        """
        self.logger.debug('Sanitizing minizer output with package: {0}'.format(package))
        self.logger.debug('Output to sanitize: {0}'.format(prettyformat(output)))
        if package=='cma':
            fitted_x = {}
            for index,par in enumerate(self.fitted_parameters):
                fitted_x[par] = output[0][index]
                if fitted_x[par]<self._bounds[par][0]:
                    fitted_x[par] = self._bounds[par][0]
                elif fitted_x[par]>self._bounds[par][1]:
                    fitted_x[par] = self._bounds[par][1]
            return (fitted_x,) + output[1:7] + (dict(output[-3]),)
        elif package=='scipy':
            fitted_x = {}
            for index,par in enumerate(self.fitted_parameters):
                fitted_x[par] = output.x[index]
                if fitted_x[par]<self._bounds[par][0]:
                    fitted_x[par] = self._bounds[par][0]
                elif fitted_x[par]>self._bounds[par][1]:
                    fitted_x[par] = self._bounds[par][1]
            stopcond = {'success': output.success, 'status': output.status,
                        'message': output.message}
            return (fitted_x, output.fun, output.nfev, output.nfev, output.nit,
                    output.x, np.nan*np.ones_like(output.x), stopcond)
        elif package=='repeat_minimize':
            fitted_x = {}
            for index,par in enumerate(self.fitted_parameters):
                fitted_x[par] = output['xbest'][index]
                if fitted_x[par]<self._bounds[par][0]:
                    fitted_x[par] = self._bounds[par][0]
                elif fitted_x[par]>self._bounds[par][1]:
                    fitted_x[par] = self._bounds[par][1]
            return (fitted_x, output['funbest'], output['nfev'], output['nfev'],
                    output['nit'], output['xmean'], output['xstd'],
                    output['stopcond'])
        else:
            raise ValueError('Unknown package used for optimization. Unable to sanitize the fmin output')
    
    # Minimizer related methods
    def init_minimizer(self, start_point, bounds, optimizer_kwargs):
        """
        self.init_minimizer(start_point, bounds, optimizer_kwargs)
        
        This method returns a callable to the minimization procedure.
        Said callable takes a single input argument, the minimization
        objective function, and returns a tuple with the sanitized
        minimization output. For more details on the output of the callable
        refer to the method sanitize_fmin_output
        
        Input:
        start_point: An array with the start point for the fitted parameters
        bounds: An array of shape (2,len(start_point)) that holds the
                lower and upper bounds for each fitted parameter. Please
                note that some of scipy's optimization methods ignore
                the parameter bounds.
        optimizer_kwargs: A dict of options passed to each optimizer.
                          Refer to the script's help for details.
        
        """
        self.logger.debug('init_minimizer args: start_point={start_point}, bounds={bounds}, optimizer_kwargs={optimizer_kwargs}'\
                          .format(start_point=start_point, bounds=bounds, optimizer_kwargs=optimizer_kwargs))
        if self.optimizer=='cma':
            scaling_factor = bounds[1]-bounds[0]
            self.logger.debug('scaling_factor = {0}'.format(scaling_factor))
            options = {'bounds':bounds,'CMA_stds':scaling_factor,'verbose':1 if optimizer_kwargs['disp'] else -1}
            options.update(optimizer_kwargs)
            restarts = options['restarts']
            del options['restarts']
            restart_from_best = options['restart_from_best']
            del options['restart_from_best']
            del options['disp']
            options = cma.CMAOptions(options)
            minimizer = lambda x: self.sanitize_fmin_output(cmaes_fmin(x,start_point,1./3.,options,restarts=restarts,restart_from_best=restart_from_best),package='cma')
        elif self.optimizer=='basinhopping':
            options = optimizer_kwargs.copy()
            for k in options.keys():
                if k not in ['niter', 'T', 'stepsize', 'minimizer_kwargs', 'take_step',\
                             'accept_test', 'callback', 'interval', 'disp', 'niter_success']:
                    del options[k]
                elif k in ['take_step', 'accept_test', 'callback']:
                    options[k] = eval(options[k])
            if 'take_step' not in options.keys():
                class Step_taker:
                    def __init__(self,stepsize=options['stepsize']):
                        self.stepsize = stepsize
                        self.scaling_factor = bounds[1]-bounds[0]
                    def __call__(self,x):
                        x+= np.random.randn(*x.shape)*self.scaling_factor*self.stepsize
                        return x
                options['take_step'] = Step_taker()
            if 'accept_test' not in options.keys():
                class Test_accepter:
                    def __init__(self,bounds=bounds):
                        self.bounds = bounds
                    def __call__(self,**kwargs):
                        return bool(np.all(np.logical_and(kwargs["x_new"]>=self.bounds[0],kwargs["x_new"]<=self.bounds[1])))
                options['accept_test'] = Test_accepter()
            if options['minimizer_kwargs']['method'] in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg']:
                jac_dx = self.get_jacobian_dx()
                epsilon = []
                for par in self.fitted_parameters:
                    epsilon.append(jac_dx[par])
                def aux_function(f):
                    options['minimizer_kwargs']['jac'] = lambda x: scipy.optimize.approx_fprime(x, f, epsilon)
                    return self.sanitize_fmin_output(scipy.optimize.basinhopping(f, start_point, **options),package='scipy')
                minimizer = aux_function
            else:
                minimizer = lambda x: self.sanitize_fmin_output(scipy.optimize.basinhopping(x, start_point, **options),package='scipy')
        else:
            repetitions = optimizer_kwargs['repetitions']
            _start_points = [start_point]
            for rsp in np.random.rand(repetitions-1,len(start_point)):
                temp = []
                for val,(lb,ub) in zip(rsp,bounds):
                    temp.append(val*(ub-lb)+lb)
                _start_points.append(np.array(temp))
            self.logger.debug('Array of start_points = {0}',_start_points)
            start_point_generator = iter(_start_points)
            if self.optimizer in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg']:
                jac_dx = self.get_jacobian_dx()
                epsilon = []
                for par in self.fitted_parameters:
                    epsilon.append(jac_dx[par])
                jac = lambda x,f: scipy.optimize.approx_fprime(x, f, epsilon)
                minimizer = lambda f: self.sanitize_fmin_output(self.repeat_minimize(f,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs,jac=lambda x:jac(x,f)),package='repeat_minimize')
            else:
                minimizer = lambda f: self.sanitize_fmin_output(self.repeat_minimize(f,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs),package='repeat_minimize')
        return minimizer
    
    def repeat_minimize(self, merit, start_point_generator, bounds, optimizer_kwargs, jac=None):
        """
        self.repeat_minimize(merit, start_point_generator, bounds, optimizer_kwargs, jac=None)
        
        A wrapper to repeat various rounds of minimization with the
        scipy.optimize.minimize or minimize_scalar method specified.
        
        Input:
            merit: The objective function used for the fits
            start_point_generator: A generator or iterable with the
                starting points that should be used for each fitting
                round
            bounds: The sanitized parameter bounds
            optimizer_kwargs: Additional kwargs to pass to the variable
                options of scipy.optimize.minimize or minimize_scalar
            jac: Can be None or a callable that computes the jacobian
                of the merit function
        
        Output:
            A dictionary with keys:
            xbest: Best solution
            funbest: Best solution function value
            nfev: Total number of function evaluations
            nit: Total number of iterations
            xs: The list of solutions for each round
            funs: The list of solutions' function values for each round
            xmean: The mean of the solutions across rounds
            xstd: The std of the solutions across rounds
            funmean: The mean of the solutions' function values across rounds
            funstd: The std of the solutions' function values across rounds
            stopcond: A dictionary with the stop condtion
        
        """
        output = {'xs':[],
                  'funs':[],
                  'nfev':0,
                  'nit':0,
                  'xbest':None,
                  'funbest':None,
                  'xmean':None,
                  'xstd':None,
                  'funmean':None,
                  'funstd':None}
        repetitions = 0
        stopcond = {'iteration_success': [], 'iteration_status': [],
                    'iteration_message': [],
                    'success': None, 'status': None, 'message': None}
        for start_point in start_point_generator:
            repetitions+=1
            self.logger.debug('Round {2} with start_point={0} and bounds={1}'.format(start_point, bounds,repetitions))
            if self.optimizer in ['Brent','Bounded','Golden']:
                res = scipy.optimize.minimize_scalar(merit,start_point,method=self.optimizer,\
                                bounds=bounds[0], options=optimizer_kwargs)
            else:
                res = scipy.optimize.minimize(merit,start_point, method=self.optimizer,bounds=bounds,\
                                options=optimizer_kwargs,jac=jac)
            self.logger.debug('New round with start_point={0} and bounds={0}'.format(prettyformat(start_point), prettyformat(bounds)))
            self.logger.debug('Round {0} ended. Fun val: {1}. x={2}'.format(repetitions,res.fun,res.x))
            self.logger.debug('OptimizeResult: {0}'.format(res))
            try:
                nit = res.rit
            except:
                nit = 1
            if isinstance(res.x,float):
                x = np.array([res.x])
            else:
                x = res.x
            output['xs'].append(x)
            output['funs'].append(res.fun)
            output['nfev']+=res.nfev
            output['nit']+=nit
            stopcond['iteration_success'].append(res.success)
            stopcond['iteration_status'].append(res.status)
            stopcond['iteration_message'].append(res.message)
            if output['funbest'] is None or res.fun<output['funbest']:
                output['funbest'] = res.fun
                output['xbest'] = x
                stopcond['success'] = res.success
                stopcond['status'] = res.status
                stopcond['message'] = res.message
            self.logger.debug('Best so far: {0} at point {1}'.format(output['funbest'],output['xbest']))
        arr_xs = np.array(output['xs'])
        arr_funs = np.array(output['funs'])
        output['xmean'] = np.mean(arr_xs)
        output['xstd'] = np.std(arr_xs)
        output['funmean'] = np.mean(arr_funs)
        output['funstd'] = np.std(arr_funs)
        output['stopcond'] = stopcond
        return output
    
    # Auxiliary functions
    def theoretical_slider_predictions(self, duration_parameters, intensity_parameters,
                                       adapt_to_merit_type=True, adapt_to_available_experiments=True):
        compute_dursl = adapt_to_available_experiments and self.__has_duration_slider
        compute_intsl = adapt_to_available_experiments and self.__has_intensity_slider
        if compute_dursl:
            slider_duration_mean, slider_duration_var = \
                Leaky._theoretical_generalstim(t=self.slider_duration_t,
                                                origin=self.slider_duration_origins,
                                                slope=self.slider_duration_slopes,
                                                **duration_parameters)
            if adapt_to_merit_type:
                if self.__is_nll_method: # log likelihood fit
                    slider_duration_mean = slider_duration_mean[self.inv_dur_stims_inds]
                    slider_duration_var = slider_duration_var[self.inv_dur_stims_inds]
        else:
            slider_duration_mean, slider_duration_var = None, None
        if compute_intsl:
            slider_intensity_mean, slider_intensity_var = \
                Leaky._theoretical_generalstim(t=self.slider_intensity_t,
                                                origin=self.slider_intensity_origins,
                                                slope=self.slider_intensity_slopes,
                                                **intensity_parameters)
            if adapt_to_merit_type:
                if self.__is_nll_method: # log likelihood fit
                    slider_intensity_mean = slider_intensity_mean[self.inv_int_stims_inds]
                    slider_intensity_var = slider_intensity_var[self.inv_int_stims_inds]
        else:
            slider_intensity_mean, slider_intensity_var = None, None
        
        return slider_duration_mean, slider_duration_var, \
               slider_intensity_mean, slider_intensity_var
    
    def theoretical_disc_predictions(self, duration_parameters, intensity_parameters,
                                     duration_error_rates, intensity_error_rates,
                                     adapt_to_merit_type=True, adapt_to_available_experiments=True):
        compute_dursl = adapt_to_available_experiments and self.__has_duration_disc
        compute_intsl = adapt_to_available_experiments and self.__has_intensity_disc
        if compute_dursl:
            if adapt_to_merit_type and self.__is_nll_method:
                disc_duration_mean, disc_duration_var = \
                    Leaky._theoretical_generalstim(t=self.disc_duration_t,
                                                    origin=self.disc_duration_origins,
                                                    slope=self.disc_duration_slopes,
                                                    **duration_parameters)
                mean1_dur = disc_duration_mean[:len(disc_duration_mean)//2]
                mean2_dur = disc_duration_mean[len(disc_duration_mean)//2:]
                var1_dur = disc_duration_var[:len(disc_duration_var)//2]
                var2_dur = disc_duration_var[len(disc_duration_var)//2:]
                prob_duration = prob2AFC(mean1_dur, var1_dur, mean2_dur, var2_dur, **duration_error_rates)
            else:
                disc_duration_mean, disc_duration_var = \
                    Leaky._theoretical_generalstim(t=self.psych_duration_t,
                                                    origin=self.psych_duration_origins,
                                                    slope=self.psych_duration_slopes,
                                                    **duration_parameters)
                mean1_dur = disc_duration_mean[:, 0]
                mean2_dur = disc_duration_mean[:, 1]
                var1_dur = disc_duration_var[:, 0]
                var2_dur = disc_duration_var[:, 1]
                prob_duration = prob2AFC(mean1_dur, var1_dur, mean2_dur, var2_dur, **duration_error_rates)
        else:
            prob_duration = None
        
        if compute_intsl:
            if adapt_to_merit_type and self.__is_nll_method:
                disc_intensity_mean, disc_intensity_var = \
                    Leaky._theoretical_generalstim(t=self.disc_intensity_t,
                                                    origin=self.disc_intensity_origins,
                                                    slope=self.disc_intensity_slopes,
                                                    **intensity_parameters)
                mean1_int = disc_intensity_mean[:len(disc_intensity_mean)//2]
                mean2_int = disc_intensity_mean[len(disc_intensity_mean)//2:]
                var1_int = disc_intensity_var[:len(disc_intensity_var)//2]
                var2_int = disc_intensity_var[len(disc_intensity_var)//2:]
                prob_intensity = prob2AFC(mean1_int, var1_int, mean2_int, var2_int, **intensity_error_rates)
            else:
                disc_intensity_mean, disc_intensity_var = \
                    Leaky._theoretical_generalstim(t=self.psych_intensity_t,
                                                    origin=self.psych_intensity_origins,
                                                    slope=self.psych_intensity_slopes,
                                                    **intensity_parameters)
                mean1_int = disc_intensity_mean[:,0]
                mean2_int = disc_intensity_mean[:,1]
                var1_int = disc_intensity_var[:,0]
                var2_int = disc_intensity_var[:,1]
                prob_intensity = prob2AFC(mean1_int, var1_int, mean2_int, var2_int, **intensity_error_rates)
        else:
            prob_intensity= None
        return prob_duration, prob_intensity
    
    # Method dependent merits
    def _merit_slider_nll(self, duration_parameters, intensity_parameters,
                          duration_error_rates, intensity_error_rates):
        nlog_likelihood = 0.
        
        slider_duration_mean, slider_duration_var, \
          slider_intensity_mean, slider_intensity_var = \
            self.theoretical_slider_predictions(duration_parameters, intensity_parameters)
        
        # Add slider log likelihood
        if self.__has_duration_slider:
            dur_norm = 0.5*(self.slider_duration_response-slider_duration_mean)**2/slider_duration_var + 0.5*np.log(2*np.pi*slider_duration_var)
            nlog_likelihood+= np.sum(dur_norm)
        if self.__has_intensity_slider:
            int_norm = 0.5*(self.slider_intensity_response-slider_intensity_mean)**2/slider_intensity_var + 0.5*np.log(2*np.pi*slider_intensity_var)
            nlog_likelihood+= np.sum(int_norm)
        return nlog_likelihood
    
    def _merit_disc_nll(self, duration_parameters, intensity_parameters,
                        duration_error_rates, intensity_error_rates):
        nlog_likelihood = 0.
        
        prob_duration, prob_intensity = \
            self.theoretical_disc_predictions(duration_parameters, intensity_parameters,
                                              duration_error_rates, intensity_error_rates)
        
        if self.__has_duration_disc:
            logp_dur_dec1 = np.log(1-prob_duration)
            logp_dur_dec2 = np.log(prob_duration)
            nlog_likelihood-= self.loglike_scale_dur
            nlog_likelihood-= np.sum(np.where(self.n_dur_dec1==0, 0., self.n_dur_dec1*logp_dur_dec1))
            nlog_likelihood-= np.sum(np.where(self.n_dur_dec2==0, 0., self.n_dur_dec2*logp_dur_dec2))
        if self.__has_intensity_disc:
            logp_int_dec1 = np.log(1-prob_intensity)
            logp_int_dec2 = np.log(prob_intensity)
            nlog_likelihood-= self.loglike_scale_int
            nlog_likelihood-= np.sum(np.where(self.n_int_dec1==0, 0., self.n_int_dec1*logp_int_dec1))
            nlog_likelihood-= np.sum(np.where(self.n_int_dec2==0, 0., self.n_int_dec2*logp_int_dec2))
        
        return nlog_likelihood
    
    def merit_parameter_regularization(self, duration_parameters, intensity_parameters,
                                       duration_error_rates, intensity_error_rates):
        reg = 0.
        dcinv = duration_parameters['C_inv']
        if dcinv is not None:
            linv = duration_parameters['leak']
            if linv!=0:
                duration_tau = 1. / dcinv * linv
            else:
                duration_tau = 1. / dcinv
            reg+= duration_tau/300.
        return reg
    
    def merit_slider_nll(self, x):
        """
        self.merit_slider_nll(x)
        
        Returns the dataset's negative log likelihood (nLL) of jointly
        observing a given slider experiment perceptual response and
        the discrimination psychometric.
        
        Input:
            x: A numpy array that is converted to the parameter dict with
                a call to self.get_parameters_dict_from_array(x)
        
        Output:
            the nLL as a floating point
        
        """
        args = self.get_parameters_dict_from_array(x, transform_taus=True, split_duration_intensity=True)
        
        return self._merit_slider_nll(*args) + \
               self.merit_parameter_regularization(*args)
    
    def merit_disc_nll(self,x):
        """
        self.merit_disc_nll(x)
        
        Returns the dataset's negative log likelihood (nLL) of
        observing a given discrimination psychometric response.
        
        Input:
            x: A numpy array that is converted to the parameter dict with
                a call to self.get_parameters_dict_from_array(x)
        
        Output:
            the nLL as a floating point
        
        """
        
        args = self.get_parameters_dict_from_array(x, transform_taus=True, split_duration_intensity=True)
        
        return self._merit_disc_nll(*args) + \
               self.merit_parameter_regularization(*args)
    
    def merit_slider_disc_nll(self, x):
        """
        self.merit_slider_disc_nll(x)
        
        Returns the dataset's negative log likelihood (nLL) of jointly
        observing a given slider experiment perceptual response and
        the discrimination psychometric.
        
        Input:
            x: A numpy array that is converted to the parameter dict with
                a call to self.get_parameters_dict_from_array(x)
        
        Output:
            the nLL as a floating point
        
        """
        args = self.get_parameters_dict_from_array(x, transform_taus=True, split_duration_intensity=True)
        
        slider_nll = self._merit_slider_nll(*args)
        disc_nll = self._merit_disc_nll(*args)
        
        return slider_nll + 50. * disc_nll + self.merit_parameter_regularization(*args)
    
    def _merit_slider_ls(self, duration_parameters, intensity_parameters,
                         duration_error_rates, intensity_error_rates):
        least_squares_sum = 0.
        
        slider_duration_mean, slider_duration_var, \
          slider_intensity_mean, slider_intensity_var = \
            self.theoretical_slider_predictions(duration_parameters, intensity_parameters)
        
        # Add slider least squares difference
        if self.__has_duration_slider:
            least_squares_sum+= np.sum(((self.slider_duration_curves-slider_duration_mean)/self.slider_duration_curves_std)**2)
        if self.__has_intensity_slider:
            least_squares_sum+= np.sum(((self.slider_intensity_curves-slider_intensity_mean)/self.slider_intensity_curves_std)**2)
        
        return least_squares_sum
    
    def _merit_disc_ls(self, duration_parameters, intensity_parameters,
                       duration_error_rates, intensity_error_rates):
        least_squares_sum = 0.
        
        prob_duration, prob_intensity = \
            self.theoretical_disc_predictions(duration_parameters, intensity_parameters,
                                              duration_error_rates, intensity_error_rates)
        
        # Add psychometric least squares difference
        if self.__has_duration_disc:
            least_squares_sum+= np.sum(((self.psych_duration-prob_duration)/self.psych_duration_std)**2)
        if self.__has_intensity_disc:
            least_squares_sum+= np.sum(((self.psych_intensity-prob_intensity)/self.psych_intensity_std)**2)
        
        return least_squares_sum
    
    def merit_slider_ls(self, x):
        """
        self.merit_slider_ls(x)
        
        Returns the dataset's negative log likelihood (nLL) of jointly
        observing a given slider experiment perceptual response and
        the discrimination psychometric.
        
        Input:
            x: A numpy array that is converted to the parameter dict with
                a call to self.get_parameters_dict_from_array(x)
        
        Output:
            the nLL as a floating point
        
        """
        args = self.get_parameters_dict_from_array(x, transform_taus=True, split_duration_intensity=True)
        
        return self._merit_slider_ls(*args) + 0.8 * self.merit_parameter_regularization(*args)
    
    def merit_disc_ls(self,x):
        """
        self.merit_disc_ls(x)
        
        Returns the dataset's negative log likelihood (nLL) of
        observing a given discrimination psychometric response.
        
        Input:
            x: A numpy array that is converted to the parameter dict with
                a call to self.get_parameters_dict_from_array(x)
        
        Output:
            the nLL as a floating point
        
        """
        args = self.get_parameters_dict_from_array(x, transform_taus=True, split_duration_intensity=True)
        
        return self._merit_disc_ls(*args) + 0.8 * self.merit_parameter_regularization(*args)
    
    def merit_slider_disc_ls(self, x):
        """
        self.merit_slider_disc_ls(x)
        
        Returns the dataset's negative log likelihood (nLL) of jointly
        observing a given slider experiment perceptual response and
        the discrimination psychometric.
        
        Input:
            x: A numpy array that is converted to the parameter dict with
                a call to self.get_parameters_dict_from_array(x)
        
        Output:
            the nLL as a floating point
        
        """
        args = self.get_parameters_dict_from_array(x, transform_taus=True, split_duration_intensity=True)
        
        slider_ls = self._merit_slider_ls(*args)
        disc_ls = self._merit_disc_ls(*args)
        return slider_ls + 50. * disc_ls + 0.6 * self.merit_parameter_regularization(*args)
    
    # Force to compute merit functions on an arbitrary parameter dict
    def forced_compute_merit(self, parameters, method=None):
        """
        self.forced_compute_merit_slider_and_disc(parameters)
        
        The same as self.merit_slider_and_disc but on a parameter dict instead of a
        parameter array.
        
        """
        
        duration_parameters = {k: parameters[v] for k, v in iteritems(self.duration_parameters) if not 'perror' in k}
        intensity_parameters = {k: parameters[v] for k, v in iteritems(self.intensity_parameters) if not 'perror' in k}
        duration_error_rates = {k: parameters[v] for k, v in iteritems(self.duration_parameters) if 'perror' in k}
        intensity_error_rates = {k: parameters[v] for k, v in iteritems(self.intensity_parameters) if 'perror' in k}
        
        if method is None:
            method = self.method.replace('_2alpha', '')
        
        if method.endswith('_nll'):
            if method.startswith('slider_disc'):
                merit = lambda *args: self._merit_slider_nll(*args) + \
                                      30. * self._merit_disc_nll(*args)
            elif method.startswith('slider'):
                merit = self._merit_slider_nll
            elif method.startswith('disc'):
                merit = self._merit_disc_nll
            else:
                raise ValueError('Invalid method, "{}", in forced_compute_merit'.format(method))
        elif method.endswith('_ls'):
            if method.startswith('slider_disc'):
                merit = lambda *args: self._merit_slider_ls(*args) + \
                                      50. * self._merit_disc_ls(*args)
            elif method.startswith('slider'):
                merit = self._merit_slider_ls
            elif method.startswith('disc'):
                merit = self._merit_disc_ls
            else:
                raise ValueError('Invalid method, "{}", in forced_compute_merit'.format(method))
        else:
            raise ValueError('Invalid method, "{}", in forced_compute_merit'.format(method))
        return merit(duration_parameters, intensity_parameters,
                     duration_error_rates, intensity_error_rates)
    
    # Theoretical predictions
    def theoretical_distributions(self,fit_output=None, transform_taus=True,
                                  adapt_to_merit_type=False):
        """
        self.theoretical_distributions(fit_output=None, transform_taus=True,
                                       adapt_to_merit_type=False)
        
        Returns the theoretically predicted joint probability density of
        slider and psychometrics.
        
        Input:
            fit_output: If None, it uses the instances fit_output. It is
                used to extract the model parameters for the computation
        
        Output:
            (pdf,t)
            pdf: A numpy array that can be like the output from a call
                to self.rt_confidence_pdf(...) or
                self.binary_confidence_rt_pdf(...) depending on the
                binary_confidence input.
            t: The time array over which the pdf is computed
        
        """
        duration_parameters, intensity_parameters, duration_error_rates, intensity_error_rates = \
            self.get_parameters_dict_from_fit_output(fit_output, transform_taus=transform_taus, split_duration_intensity=True)
        
        slider_duration_mean, slider_duration_var, \
          slider_intensity_mean, slider_intensity_var = \
            self.theoretical_slider_predictions(duration_parameters, intensity_parameters,
                                                adapt_to_merit_type=adapt_to_merit_type)
        
        prob_duration, prob_intensity = \
            self.theoretical_disc_predictions(duration_parameters, intensity_parameters,
                                              duration_error_rates, intensity_error_rates,
                                              adapt_to_merit_type=adapt_to_merit_type)
        
        return slider_duration_mean, slider_duration_var, slider_intensity_mean, slider_intensity_var,\
                prob_duration, prob_intensity

class Fitter_plot_handler(object):
    def __init__(self, obj, merge=None):
        self.__initlogger__()
        self.plotters = {}
        for key, val in iteritems(obj):
            self.plotters[key] = Fitter_ploter(val)
    
    def __initlogger__(self):
        self.logger = logging.getLogger("fits_module.Fitter_plot_handler")
    
    def keys(self):
        return self.plotters.keys()
    
    def merge(self, merge=None):
        self.logger.debug('Merging with option {0}'.format(merge))
        if merge:
            output = Fitter_plot_handler({})
            if merge=='sessions':
                key_merger = lambda key: re.sub('_session_[\[\]\- 0-9]+','',key)
            elif merge=='subjects':
                key_merger = lambda key: re.sub('_subject_[\[\]\- 0-9]+','',key)
            elif merge=='all':
                key_merger = lambda key: re.sub('_session_[\[\]\- 0-9]+','',re.sub('_subject_[\[\]\- 0-9]+','',key))
            for key, plotter in iteritems(self.plotters):
                out_key = key_merger(key)
                if not out_key in output.keys():
                    output[out_key] = plotter.copy()
                else:
                    output[out_key]+= plotter
        else:
            output = self
        self.logger.debug('Output number of keys {0} from input number {1}'.format(len(output.keys()), len(self.keys())))
        return output
    
    def copy(self):
        output = Fitter_plot_handler({})
        for key, val in iteritems(self.plotters):
            output[key] = val.copy()
        return output
    
    def __iadd__(self, other):
        self.logger.debug('Adding fitter_plot_handler')
        self.logger.debug('Input number of keys {0} / {1}'.format(len(self.keys()),len(other.keys())))
        for key, val in iteritems(other.plotters):
            if key in self.keys():
                self[key]+= val
            else:
                self[key] = val.copy()
        self.logger.debug('Result number of keys {0}'.format(len(self.keys())))
        return self
    
    def __add__(self,other):
        output = self.copy()
        output.__iadd__(other)
        return output
    
    def __getitem__(self,key):
        return self.plotters[key]
    
    def __setitem__(self,key,value):
        self.plotters[key] = value
    
    def plot(self, show=False, saver=None):
        for key, plotter in iteritems(self.plotters):
            fig = plotter.plot()
            fig.suptitle(key, fontsize=24)
            if saver:
                self.logger.debug('Saving figure')
                if isinstance(saver,str):
                    fig.savefig(saver,bbox_inches='tight')
                else:
                    saver.savefig(fig,bbox_inches='tight')
        if show:
            self.logger.debug('Showing figure')
            plt.show(True)
    
    def __getstate__(self):
        return {key: plotter.__getstate__() for key, plotter in iteritems(self.plotters)}
    
    def __setstate__(self, state):
        self.__initlogger__()
        self.logger.debug('Setting state from: {0}'.format(prettyformat(state)))
        self.plotters = {}
        for key, plotter_state in iteritems(state):
            self.logger.debug('Setting Fitter_ploter state with key = {0} and state =\n{1}'.format(key, prettyformat(plotter_state)))
            plotter = Fitter_ploter({})
            plotter.__setstate__(plotter_state)
            self.plotters[key] = plotter
    
    def save(self, fname):
        self.logger.debug('Fitter_plot_handler state that will be saved = {0}'.format(prettyformat(self.__getstate__())))
        self.logger.info('Saving Fitter_plot_handler state to file "%s"',fname)
        f = open(fname,'wb')
        pickle.dump(self, f, PICKLE_PROTOCOL)
        f.close()
    
    def savemat(self, fname, varname=None):
        if varname is None:
            varname = str('data')
        state = self.as_matlab_format()
        self.logger.debug('Fitter_plot_handler state that will be saved = {0}'.format(prettyformat(state)))
        self.logger.info('Saving Fitter_plot_handler state to file "%s"',fname)
        self.logger.info('Data will be saved with the variable name = {0}'.format(varname))
        scipy.io.savemat(fname, {varname: state})
    
    def as_matlab_format(self):
        data = {}
        keys = []
        colnames = {}
        for key, handler in iteritems(self.plotters):
            keys.append(str(key))
            temp = handler.as_matlab_format()
            for temp_key in temp.keys():
                if temp_key.endswith('_colnames'):
                    if temp_key in colnames.keys():
                        if tuple(temp[temp_key])!=tuple(colnames[temp_key]):
                            raise RuntimeError('Individual Fitter_ploters have inconsistent colnames. Save each separately using Fitter_ploter.savemat')
                    else:
                        colnames[temp_key] = temp[temp_key]
                else:
                    if temp_key in data.keys():
                        data[temp_key].append(temp[temp_key])
                    else:
                        data[temp_key] = [temp[temp_key]]
        for k in data.keys():
            data[k] = np.array(data[k])
            data[k] = np.transpose(data[k], (1, 2, 0))
        output = {str('third_dim_names'): keys}
        output.update(data)
        output.update(colnames)
        return output

class Fitter_ploter(object):
    def __init__(self, obj):
        """
        Fitter_ploter(dictionary)
        
        """
        self.__initlogger__()
        self.logger.debug('initing from {0}'.format(prettyformat(obj)))
        dataframe = obj.pop('dataframe', None)
        self.logger.debug("Setting dataframe from {0}".format(prettyformat(dataframe)))
        if not dataframe is None:
            self.has_data = True
            self.dataframe = dataframe.copy()
            self.experiments = self.dataframe['experiment'].unique()
            self.sessions = self.dataframe['session'].unique()
            self.subjects = self.dataframe['subject'].unique()
        else:
            self.has_data = False
        
        slider = obj.pop('slider', None)
        self.logger.debug("Setting slider from {0}".format(prettyformat(slider)))
        self.slider_duration_mean  = None
        self.slider_duration_var   = None
        self.slider_duration_n     = None
        self.slider_duration_t     = None
        self.slider_duration_sp    = None
        self.slider_intensity_mean = None
        self.slider_intensity_var  = None
        self.slider_intensity_n    = None
        self.slider_intensity_t    = None
        self.slider_intensity_sp   = None
        if slider:
            if not slider['duration_mean'] is None:
                self.__has_duration_slider = True
                self.slider_duration_mean  = slider['duration_mean'].copy()
                self.slider_duration_var   = slider['duration_var'].copy()
                self.slider_duration_n     = slider['duration_n'].copy()
                self.slider_duration_t     = slider['duration_t'].copy()
                self.slider_duration_sp    = slider['duration_sp'].copy()
            else:
                self.__has_duration_slider = False
            
            if not slider['intensity_mean'] is None:
                self.__has_intensity_slider = True
                self.slider_intensity_mean = slider['intensity_mean'].copy()
                self.slider_intensity_var  = slider['intensity_var'].copy()
                self.slider_intensity_n    = slider['intensity_n'].copy()
                self.slider_intensity_t    = slider['intensity_t'].copy()
                self.slider_intensity_sp   = slider['intensity_sp'].copy()
            else:
                self.__has_intensity_slider = False
        else:
            self.__has_duration_slider = False
            self.__has_intensity_slider = False
        
        self.has_slider = self.__has_duration_slider or self.__has_intensity_slider
        if self.has_slider and not 'slider' in self.experiments:
            raise ValueError('Dataframe has no slider experiment data, but slider model data was supplied')
        
        disc = obj.pop('disc', None)
        self.logger.debug("Setting disc from {0}".format(prettyformat(disc)))
        self.prob_duration      = None
        self.psych_duration_t   = None
        self.psych_duration_sp  = None
        self.psych_duration_n   = None
        self.prob_intensity     = None
        self.psych_intensity_t  = None
        self.psych_intensity_sp = None
        self.psych_intensity_n  = None
        if disc:
            if not disc['prob_duration'] is None:
                self.__has_duration_disc = True
                self.prob_duration      = disc['prob_duration'].copy()
                self.psych_duration_t   = disc['psych_duration_t'].copy()
                self.psych_duration_sp  = disc['psych_duration_sp'].copy()
                self.psych_duration_n   = disc['psych_duration_n'].copy()
            else:
                self.__has_duration_disc = False
            if not disc['prob_intensity'] is None:
                self.__has_intensity_disc = True
                self.prob_intensity     = disc['prob_intensity'].copy()
                self.psych_intensity_t  = disc['psych_intensity_t'].copy()
                self.psych_intensity_sp = disc['psych_intensity_sp'].copy()
                self.psych_intensity_n  = disc['psych_intensity_n'].copy()
            else:
                self.__has_intensity_disc = False
        else:
            self.__has_duration_disc = False
            self.__has_intensity_disc = False
        
        self.has_disc = self.__has_duration_disc or self.__has_intensity_disc
        if self.has_disc and not (('human_int_disc' in self.experiments and 'human_dur_disc' in self.experiments) or \
                     'rat_disc' in self.experiments):
                raise ValueError('Dataframe has no discrimination experiment data, but disc model data was supplied')
    
    def __initlogger__(self):
        self.logger = logging.getLogger("fits_module.Fitter_ploter")
    
    def copy(self):
        return Fitter_ploter(self.to_dict())
    
    def __iadd__(self, other):
        # Concatenate dataframes
        dfs = [x.dataframe for x in [self, other] if x.has_data]
        if dfs:
            df = pd.concat(dfs).copy()
        else:
            df = None
        obj = {'dataframe': df}
        
        # Handle slider data
        # By default use self slider data, which can be None, depending on self.__has_duration_slider and self.__has_intensity_slider
        slider = {'duration_mean': self.slider_duration_mean,
                  'duration_var': self.slider_duration_var,
                  'duration_n': self.slider_duration_n,
                  'duration_t': self.slider_duration_t,
                  'duration_sp': self.slider_duration_sp,
                  'intensity_mean': self.slider_intensity_mean,
                  'intensity_var': self.slider_intensity_var,
                  'intensity_n': self.slider_intensity_n,
                  'intensity_t': self.slider_intensity_t,
                  'intensity_sp': self.slider_intensity_sp
                 }
        
        if self.__has_duration_slider and other.__has_duration_slider:
            # If both have duration slider data, then merge
            n, xs, ys = \
                Fitter_ploter._merge_arrays(xs=(np.array([self.slider_duration_t, self.slider_duration_sp]),
                                                np.array([other.slider_duration_t, other.slider_duration_sp])),
                                            n=(self.slider_duration_n, other.slider_duration_n),
                                            ys=(np.array([self.slider_duration_mean, self.slider_duration_var]),
                                                np.array([other.slider_duration_mean, other.slider_duration_var]))
                                            )
            
            slider.update({'duration_mean': ys[0],
                           'duration_var': ys[1],
                           'duration_n': n,
                           'duration_t': xs[0],
                           'duration_sp': xs[1]})
        elif other.__has_duration_slider:
            # If only other has duration slider data, use it; else keep the default self data (that is None if self.__has_duration_slider is False)
            slider.update({'duration_mean': other.slider_duration_mean,
                           'duration_var': other.slider_duration_var,
                           'duration_n': other.slider_duration_n,
                           'duration_t': other.slider_duration_t,
                           'duration_sp': other.slider_duration_sp})
        
        if self.__has_intensity_slider and other.__has_intensity_slider:
            # If both have intensity slider data, then merge
            n, xs, ys = \
                Fitter_ploter._merge_arrays(xs=(np.array([self.slider_intensity_t, self.slider_intensity_sp]),
                                                np.array([other.slider_intensity_t, other.slider_intensity_sp])),
                                            n=(self.slider_intensity_n, other.slider_intensity_n),
                                            ys=(np.array([self.slider_intensity_mean, self.slider_intensity_var]),
                                                np.array([other.slider_intensity_mean, other.slider_intensity_var]))
                                            )
            
            slider.update({'intensity_mean': ys[0],
                           'intensity_var': ys[1],
                           'intensity_n': n,
                           'intensity_t': xs[0],
                           'intensity_sp': xs[1]})
        elif other.__has_intensity_slider:
            # If only other has intensity slider data, use it; else keep the default self data
            slider.update({'intensity_mean': other.slider_intensity_mean,
                           'intensity_var': other.slider_intensity_var,
                           'intensity_n': other.slider_intensity_n,
                           'intensity_t': other.slider_intensity_t,
                           'intensity_sp': other.slider_intensity_sp})
        obj['slider'] = slider
        
        # Handle discrimination data
        # By default use self disc data, which can be None, depending on self.__has_duration_disc and self.__has_intensity_disc
        disc = {'prob_duration': self.prob_duration,
                'psych_duration_t': self.psych_duration_t,
                'psych_duration_sp': self.psych_duration_sp,
                'psych_duration_n': self.psych_duration_n,
                'prob_intensity': self.prob_intensity,
                'psych_intensity_t': self.psych_intensity_t,
                'psych_intensity_sp': self.psych_intensity_sp,
                'psych_intensity_n': self.psych_intensity_n,
               }
        # If both have discrimination data, they must be merged taking care of None data
        if self.__has_duration_disc and other.__has_duration_disc:
            # If both have duration discrimination data, then merge
            n, xs, ys = \
                Fitter_ploter._merge_arrays(xs=(np.array([self.psych_duration_t[:, 0], self.psych_duration_t[:, 1], self.psych_duration_sp[:, 0], self.psych_duration_sp[:, 1]]),
                                                np.array([other.psych_duration_t[:, 0], other.psych_duration_t[:, 1], other.psych_duration_sp[:, 0], other.psych_duration_sp[:, 1]])),
                                            n=(self.psych_duration_n, other.psych_duration_n),
                                            ys=(self.prob_duration, other.prob_duration)
                                            )
            
            disc.update({'prob_duration': ys,
                         'psych_duration_t': np.concatenate((xs[0].reshape((-1,1)), xs[1].reshape((-1,1))), axis=-1),
                         'psych_duration_sp': np.concatenate((xs[2].reshape((-1,1)), xs[3].reshape((-1,1))), axis=-1),
                         'psych_duration_n': n})
        elif other.__has_duration_disc:
            # If only other has duration discrimination data, use it; else keep the default self data
            disc.update({'prob_duration': other.prob_duration,
                         'psych_duration_t': other.psych_duration_t,
                         'psych_duration_sp': other.psych_duration_sp,
                         'psych_duration_n': other.psych_duration_n})
        
        if self.__has_intensity_disc and other.__has_intensity_disc:
            # If both have intensity discrimination data, then merge
            n, xs, ys = \
                Fitter_ploter._merge_arrays(xs=(np.array([self.psych_intensity_t[:, 0], self.psych_intensity_t[:, 1], self.psych_intensity_sp[:, 0], self.psych_intensity_sp[:, 1]]),
                                                np.array([other.psych_intensity_t[:, 0], other.psych_intensity_t[:, 1], other.psych_intensity_sp[:, 0], other.psych_intensity_sp[:, 1]])),
                                            n=(self.psych_intensity_n, other.psych_intensity_n),
                                            ys=(self.prob_intensity, other.prob_intensity)
                                            )
            
            disc.update({'prob_intensity': ys,
                         'psych_intensity_t': np.concatenate((xs[0].reshape((-1,1)), xs[1].reshape((-1,1))), axis=-1),
                         'psych_intensity_sp': np.concatenate((xs[2].reshape((-1,1)), xs[3].reshape((-1,1))), axis=-1),
                         'psych_intensity_n': n})
        elif other.__has_intensity_disc:
            # If only other has duration discrimination data, use it; else keep the default self data
            disc.update({'prob_intensity': other.prob_intensity,
                         'psych_intensity_t': other.psych_intensity_t,
                         'psych_intensity_sp': other.psych_intensity_sp,
                         'psych_intensity_n': other.psych_intensity_n})
        obj['disc'] = disc
        
        # The merged data is used to reinit self
        self.__init__(obj)
        return self
    
    def __add__(self, other):
        output = self.copy()
        output.__iadd__(other)
        return output
    
    def get_slider_data(self):
        if self.__has_duration_slider:
            # Subject duration perception
            d = self.dataframe.query(Fitter.human_duration_slider_query)
            sps = np.unique(np.array(d.intensity1))
            vals = d.groupby(['intensity1', 'duration1'])['answer']\
                .apply(lambda x: Bootstraper.Median(x, 1000, 0.5, return_std=True))
            means = vals.xs('mean', level=2).values.flatten().reshape((len(sps),-1))
            stds = vals.xs('std', level=2).values.flatten().reshape((len(sps),-1))
            Ts = np.unique(np.array([d.intensity1, d.duration1]).T, axis=0)[:,1].reshape((len(sps),-1))
            subject_duration = []
            for sp, T, val, err in zip(sps, Ts, means, stds):
                subject_duration.append({'x': T,
                                         'y': val,
                                         'ey': err,
                                         'label': '{0:.0f}'.format(sp)})
            
            # Model duration predictions
            duration = []
            for i, sp in enumerate(np.unique(self.slider_duration_sp)):
                inds = self.slider_duration_sp==sp
                duration.append({'x': self.slider_duration_t[inds],
                                 'y': self.slider_duration_mean[inds],
                                 'ey': np.sqrt(self.slider_duration_var[inds]),
                                 'label': '{0:.0f}'.format(sp)})
        else:
            subject_duration = None
            duration = None
        
        if self.__has_intensity_slider:
            # Subject intensity perception
            d = self.dataframe.query(Fitter.human_intensity_slider_query)
            Ts = np.unique(np.array(d.duration1))
            vals = d.groupby(['duration1', 'intensity1'])['answer']\
                .apply(lambda x: Bootstraper.Median(x, 1000, 0.5, return_std=True))
            means = vals.xs('mean', level=2).values.flatten().reshape((len(Ts),-1))
            stds = vals.xs('std', level=2).values.flatten().reshape((len(Ts),-1))
            sps = np.unique(np.array([d.duration1, d.intensity1]).T, axis=0)[:,1].reshape((len(Ts),-1))
            subject_intensity = []
            for T, sp, val, err in zip(Ts, sps, means, stds):
                subject_intensity.append({'x': sp,
                                          'y': val,
                                          'ey': err,
                                          'label': '{0:.0f}'.format(T)})
            
            # Model intensity predictions
            intensity = []
            for i, T in enumerate(np.unique(self.slider_intensity_t)):
                inds = self.slider_intensity_t==T
                intensity.append({'x': self.slider_intensity_sp[inds],
                                  'y': self.slider_intensity_mean[inds],
                                  'ey': np.sqrt(self.slider_intensity_var[inds]),
                                  'label': '{0:.0f}'.format(T)})
        else:
            subject_intensity = None
            intensity = None
        
        data = {'duration': duration,
                'subject_duration': subject_duration,
                'intensity': intensity,
                'subject_intensity': subject_intensity}
        return data
    
    def get_disc_data(self):
        if self.__has_duration_disc:
            if 'human_dur_disc' in self.experiments:
                # Human vertical psychometric
                query = Fitter.human_duration_psychometric_query
            else:
                # Rat vertical psychometric
                query = Fitter.rat_duration_psychometric_query
            d = self.dataframe.query(query)
            
            # Subject duration data
            bla = d.assign(prob2=(d['answer']).values)\
                    .groupby(['NTD', 'NSD'])['prob2']\
                    .apply(lambda x: Bootstraper.Mean(x, 1000, 0.5, return_std=True))
            means = bla.xs('mean', level=2)
            stds = bla.xs('std', level=2)
            
            stim2 = means.index.get_values()
            ntds = np.unique(np.array([bla[0] for bla in stim2]))
            nsds = np.unique(np.array([bla[1] for bla in stim2]))
            ntd_grid, nsd_grid = np.meshgrid(ntds, nsds, indexing='ij')
            ps = np.nan * np.ones_like(ntd_grid)
            eps = np.nan * np.ones_like(ntd_grid)
            for (ntd, nsd), psych in iteritems(means):
                ps[ntds==ntd, nsds==nsd] = psych
                eps[ntds==ntd, nsds==nsd] = stds[(ntd, nsd)]
            
            subject_duration1 = [{'x': ntd, 'y': p, 'ey': ep, 'label': '{0:.1f}'.format(nsd)}
                                  for nsd, p, ep, ntd in zip(nsd_grid[0], ps.T, eps.T, ntd_grid.T)]
            subject_duration2 = [{'x': nsd, 'y': p, 'ey': ep, 'label': '{0:.1f}'.format(ntd)}
                                  for nsd, p, ep, ntd in zip(nsd_grid, ps, eps, ntd_grid[:,0])]
        
            # Model duration predictions
            ntd_arr = np.round(np.squeeze(np.diff(self.psych_duration_t, axis=1))/np.sum(self.psych_duration_t, axis=1), 3)
            ntds = np.unique(ntd_arr)
            nsd_arr = np.round(np.squeeze(np.diff(self.psych_duration_sp, axis=1))/np.sum(self.psych_duration_sp, axis=1), 3)
            nsds = np.unique(nsd_arr)
            ntd_grid, nsd_grid = np.meshgrid(ntds, nsds, indexing='ij')
            ps = np.nan * np.ones_like(ntd_grid)
            for ntd, nsd, psych in zip(ntd_arr, nsd_arr, self.prob_duration):
                ps[ntd==ntds, nsd==nsds] = psych
            
            duration1 = [{'x': ntd, 'y': p, 'label': '{0:.1f}'.format(nsd)} for nsd, p, ntd in zip(nsd_grid[0], ps.T, ntd_grid.T)]
            duration2 = [{'x': nsd, 'y': p, 'label': '{0:.1f}'.format(ntd)} for nsd, p, ntd in zip(nsd_grid, ps, ntd_grid[:,0])]
        else:
            subject_duration1 = None
            subject_duration2 = None
            duration1 = None
            duration2 = None
        
        if self.__has_intensity_disc:
            if 'human_int_disc' in self.experiments:
                # Human vertical psychometric
                query = Fitter.human_intensity_psychometric_query
            else:
                # Rat vertical psychometric
                query = Fitter.rat_intensity_psychometric_query
            d = self.dataframe.query(query)
            sp1 = np.reshape(np.array(sorted(d.intensity1.unique())), (1, -1))
            T1 = np.reshape(np.array(sorted(d.duration1.unique())), (-1, 1))
            
            # Subject intensity data
            bla = d.assign(prob2=(d['answer']).values)\
                    .groupby(['NTD', 'NSD'])['prob2']\
                    .apply(lambda x: Bootstraper.Mean(x, 1000, 0.5, return_std=True))
            means = bla.xs('mean', level=2)
            stds = bla.xs('std', level=2)
            
            stim2 = means.index.get_values()
            ntds = np.unique(np.array([bla[0] for bla in stim2]))
            nsds = np.unique(np.array([bla[1] for bla in stim2]))
            ntd_grid, nsd_grid = np.meshgrid(ntds, nsds, indexing='ij')
            ps = np.nan * np.ones_like(ntd_grid)
            eps = np.nan * np.ones_like(ntd_grid)
            for (ntd, nsd), psych in iteritems(means):
                ps[ntds==ntd, nsds==nsd] = psych
                eps[ntds==ntd, nsds==nsd] = stds[(ntd, nsd)]
            
            subject_intensity1 = [{'x': nsd, 'y': p, 'ey': ep, 'label': '{0:.1f}'.format(ntd)}
                                  for nsd, p, ep, ntd in zip(nsd_grid, ps, eps, ntd_grid[:,0])]
            subject_intensity2 = [{'x': ntd, 'y': p, 'ey': ep, 'label': '{0:.1f}'.format(nsd)}
                                  for nsd, p, ep, ntd in zip(nsd_grid[0], ps.T, eps.T, ntd_grid.T)]
            
            # Model intensity predictions
            ntd_arr = np.round(np.squeeze(np.diff(self.psych_intensity_t, axis=1))/np.sum(self.psych_intensity_t, axis=1), 3)
            ntds = np.unique(ntd_arr)
            nsd_arr = np.round(np.squeeze(np.diff(self.psych_intensity_sp, axis=1))/np.sum(self.psych_intensity_sp, axis=1), 3)
            nsds = np.unique(nsd_arr)
            ntd_grid, nsd_grid = np.meshgrid(ntds, nsds, indexing='ij')
            ps = np.nan * np.ones_like(ntd_grid)
            for ntd, nsd, psych in zip(ntd_arr, nsd_arr, self.prob_intensity):
                ps[ntd==ntds, nsd==nsds] = psych
            
            intensity1 = [{'x': nsd, 'y': p, 'label': '{0:.1f}'.format(ntd)} for nsd, p, ntd in zip(nsd_grid, ps, ntd_grid[:,0])]
            intensity2 = [{'x': ntd, 'y': p, 'label': '{0:.1f}'.format(nsd)} for nsd, p, ntd in zip(nsd_grid[0], ps.T, ntd_grid.T)]
        else:
            subject_intensity1 = None
            subject_intensity2 = None
            intensity1 = None
            intensity2 = None
        
        disc_data = {'duration': duration1,
                     'subject_duration': subject_duration1,
                     'intensity': intensity1,
                     'subject_intensity': subject_intensity1}
        irrelevant_disc_data = {'duration': duration2,
                                'subject_duration': subject_duration2,
                                'intensity': intensity2,
                                'subject_intensity': subject_intensity2}
        return disc_data, irrelevant_disc_data
    
    def plot(self, fig=None):
        """
        plot(self, fig=None)
        
        Main plotting routine has two completely different output forms
        that depend on the parameter is_binary_confidence. If
        is_binary_confidence is True (or if it is None but the
        Fitter_plot_handler caller was constructed from a binary
        confidence Fitter instance) then this function produces a figure
        with 4 axes distributed as a subplot(22i).
        subplot(221) will hold the rt distribution of hits and misses
        subplot(222) will hold the rt distribution of high and low
        confidence reports
        subplot(223) will hold the rt distribution of high and low hit
        confidence reports
        subplot(224) will hold the rt distribution of high and low miss
        confidence reports
        
        If is_binary_confidence is False (or if it is None but the
        Fitter_plot_handler caller was constructed from a continuous
        confidence Fitter instance) then this function produces a figure
        with 6 axes distributed as a subplot(32i).
        subplot(321) will hold the rt distribution of hits and misses
        subplot(322) will hold the confidence distribution of hits and
        misses. This plot can be in logscale if the input logscale is
        True.
        subplot(323) and subplot(324) will hold the experimental joint
        rt-confidence distributions of hits and misses respectively.
        subplot(325) and subplot(326) will hold the theoretical joint
        rt-confidence distributions of hits and misses respectively.
        All four of the above mentioned graph are affected by the
        logscale input parameter. If True, the colorscale is logarithmic.
        
        Other input arguments:
        xlim_rt_cutoff: A bool that indicates whether to set the
            theoretical graphs xlim equal to experimental xlim for all
            plots that involve response times.
        show: A bool indicating whether to show the figure after it has
            been created and freezing the execution until it is closed.
        saver: If it is not None, saver must be a string or an object
            that implements the savefig method similar to
            matplotlib.pyplot.savefig. If it is a string it will be used
            to save the figure as:
            matplotlib.pyplot.savefig(saver,,bbox_inches='tight')
        binary_split_method: Override the way in which the continuous
            confidence reports are binarized. Available methods are
            None, 'median', 'half' and 'mean'. If None, the
            Fitter_plot_handler's binary_split_method attribute will be
            used. If supplied value is not None, the binarization method
            will be overriden. Be aware that this parameter will affect
            the subject's data and it will also affect the Fitter's data
            only if the Fitter's data is encoded in a continuous way.
            If said data is already binary, the binary_split_method will
            have no additional effect. These methods are only used
            when is_binary_confidence is True. For a detailed
            description of the three methods mentioned above, refer to
            Fitter.get_binary_confidence.
        
        """
        if not can_plot:
            raise ImportError('Could not import matplotlib package and it is imposible to produce any plot')
        if not self.has_data:
            raise ValueError('No data to plot')
        
        n_cols = 0
        if self.has_slider:
            slider_data = self.get_slider_data()
            slider_col = 0
            n_cols+= 1
        else:
            slider_data = None
        if self.has_disc:
            disc_data, irrelevant_disc_data = self.get_disc_data()
            disc_col = n_cols
            n_cols+= 2
        else:
            disc_data = irrelevant_disc_data = None
        if fig is None:
            fig = plt.figure(figsize=(15,9))
        fig_gs = gridspec.GridSpec(1, n_cols, left=0.08, right=0.92, top=0.90, bottom=0.1, wspace=0.35)
        
        slider_gs = None
        slider_colors_duration = None
        slider_colors_intensity = None
        if self.has_slider:
            slider_gs = gridspec.GridSpecFromSubplotSpec(2, 2, fig_gs[slider_col], width_ratios=[14, 1], wspace=0.03, hspace=0.20)
            if self.__has_duration_slider:
                slider_colors_duration  = np.array([plt.get_cmap('hot')(xx) for xx in np.linspace(0.25, 0.75, len(slider_data['duration']))])
            if self.__has_intensity_slider:
                slider_colors_intensity = np.array([plt.get_cmap('winter')(xx) for xx in np.linspace(0.25, 0.75, len(slider_data['intensity']))])
        
        psycho_gs = None
        irrelevant_psycho_gs = None
        pysho_colors_ntd = None
        pysho_colors_nsd = None
        irrelevant_pysho_colors_ntd = None
        irrelevant_pysho_colors_nsd = None
        if self.has_disc:
            psycho_gs = gridspec.GridSpecFromSubplotSpec(2, 2, fig_gs[disc_col], width_ratios=[14, 1], wspace=0.03, hspace=0.20)
            irrelevant_psycho_gs = gridspec.GridSpecFromSubplotSpec(2, 2, fig_gs[disc_col+1], width_ratios=[14, 1], wspace=0.03, hspace=0.20)
            if self.__has_duration_disc:
                pysho_colors_ntd = np.array([plt.get_cmap('hot')(xx) for xx in np.linspace(0.25, 0.75, len(disc_data['duration']))])
                irrelevant_pysho_colors_nsd = np.array([plt.get_cmap('winter')(xx) for xx in np.linspace(0.25, 0.75, len(irrelevant_disc_data['duration']))])
            if self.__has_intensity_disc:
                pysho_colors_nsd = np.array([plt.get_cmap('winter')(xx) for xx in np.linspace(0.25, 0.75, len(disc_data['intensity']))])
                irrelevant_pysho_colors_ntd = np.array([plt.get_cmap('hot')(xx) for xx in np.linspace(0.25, 0.75, len(irrelevant_disc_data['intensity']))])
        
        slider_xlabels = {'duration': 'Duration [ms]', 'intensity': 'Sp [mm/s]'}
        disc_xlabels = {'duration': 'NTD', 'intensity': 'NSD'}
        irrelevant_disc_xlabels = {'duration': 'NSD', 'intensity': 'NTD'}
        
        slider_ylabels = {'duration': 'Duration\nEstimation', 'duration_cbar': 'Sp [mm/s]',
                          'intensity': 'Intensity\nEstimation', 'intensity_cbar': 'T [ms]'}
        disc_ylabels = {'duration': 'Prob T2>T1', 'duration_cbar': 'NSD',
                        'intensity': 'Prob S2>S1', 'intensity_cbar': 'NTD'}
        irrelevant_disc_ylabels = {'duration': 'Prob S2>S1', 'duration_cbar': 'NTD',
                                   'intensity': 'Prob T2>T1', 'intensity_cbar': 'NSD'}
        
        it = zip([slider_gs, psycho_gs, irrelevant_psycho_gs],
                 [slider_data, disc_data, irrelevant_disc_data],
                 [slider_colors_intensity, pysho_colors_nsd, irrelevant_pysho_colors_ntd],
                 [slider_colors_duration, pysho_colors_ntd, irrelevant_pysho_colors_nsd],
                 ['Slider', 'Informative var', 'Non informative var'],
                 [slider_xlabels, disc_xlabels, irrelevant_disc_xlabels],
                 [slider_ylabels, disc_ylabels, irrelevant_disc_ylabels]
                )
        for gs, data, colors_dur, colors_int, title, xlabel, ylabel in it:
            try:
                dur_ax   = fig.add_subplot(gs[0,0])
                dur_cbar = fig.add_subplot(gs[0,1])
                int_ax   = fig.add_subplot(gs[1,0])
                int_cbar = fig.add_subplot(gs[1,1])
            except:
                continue
            
            if not data['duration'] is None:
                for d, color in zip(data['duration'],colors_int):
                    x = d['x']
                    y = d['y']
                    dur_ax.plot(x, y, color=color)
                    try:
                        ey = d['ey']
                        dur_ax.fill_between(x, y+ey, y-ey, alpha=0.1, color=color)
                    except:
                        pass
                for d, color in zip(data['subject_duration'],colors_int):
                    dur_ax.errorbar(d['x'], d['y'], d['ey'], linestyle='--', color=color, linewidth=2)
                dur_ax.autoscale(enable=True, axis='x', tight=True)
                dur_ax.set_xlabel(xlabel['duration'], fontsize=14)
                dur_ax.set_ylabel(ylabel['duration'], color='b', fontsize=14)
                dur_ax.set_title(title, fontsize=20)
                dur_cbar.imshow(colors_int[:,None,:], extent=[0, 1, -0.5, len(colors_int)-0.5],
                                interpolation='nearest', aspect='auto', origin='lower')
                dur_cbar.set_yticks(range(len(colors_int)))
                dur_cbar.set_yticklabels([i['label'] for i in data['duration']])
                dur_cbar.tick_params(axis='both', direction='inout', labelleft='off',
                        right='on', labelright='on', bottom='off', labelbottom='off')
                dur_cbar.yaxis.set_label_position('right')
                dur_cbar.set_ylabel(ylabel['duration_cbar'])
            
            if not data['intensity'] is None:
                for d, color in zip(data['intensity'], colors_dur):
                    x = d['x']
                    y = d['y']
                    int_ax.plot(x, y, color=color)
                    try:
                        ey = d['ey']
                        int_ax.fill_between(x, y+ey, y-ey, alpha=0.1, color=color)
                    except:
                        pass
                for d, color in zip(data['subject_intensity'], colors_dur):
                    int_ax.errorbar(d['x'], d['y'], d['ey'], linestyle='--', color=color, linewidth=2)
                int_ax.autoscale(enable=True, axis='x', tight=True)
                int_ax.set_xlabel(xlabel['intensity'], fontsize=14)
                int_ax.set_ylabel(ylabel['intensity'], color='r', fontsize=14)
                int_cbar.imshow(colors_dur[:,None,:], extent=[0, 1, -0.5, len(colors_dur)-0.5],
                                interpolation='nearest', aspect='auto', origin='lower')
                int_cbar.set_yticks(range(len(colors_dur)))
                int_cbar.set_yticklabels([i['label'] for i in data['intensity']])
                int_cbar.tick_params(axis='both', direction='inout', labelleft='off',
                        right='on', labelright='on', bottom='off', labelbottom='off')
                int_cbar.yaxis.set_label_position('right')
                int_cbar.set_ylabel(ylabel['intensity_cbar'])
        return fig
    
    def save(self, fname):
        self.logger.debug('Fitter_plot_handler state that will be saved = {0}'.format(prettyformat(self.__getstate__())))
        self.logger.info('Saving Fitter_plot_handler state to file "%s"', fname)
        f = open(fname, 'wb')
        pickle.dump(self, f, PICKLE_PROTOCOL)
        f.close()
    
    def savemat(self, fname, varname=None):
        if varname is None:
            varname = str('data')
        state = self.as_matlab_format()
        self.logger.debug('Fitter_plot_handler state that will be saved = {0}'.format(prettyformat(state)))
        self.logger.info('Saving Fitter_plot_handler state to file "%s" in matlab format', fname)
        self.logger.info('Data will be saved with the variable name = {0}'.format(varname))
        scipy.io.savemat(fname, {varname: state})
    
    def __setstate__(self, state):
        self.__initlogger__()
        self.logger.debug('Setting state from {0}'.format(prettyformat(state)))
        df = pd.DataFrame()
        df.__setstate__(state['dataframe'])
        state['dataframe'] = df
        self.__init__(state)
    
    def to_dict(self):
        d = {'dataframe': self.dataframe,
             'slider': {'duration_mean': self.slider_duration_mean,
                        'duration_var': self.slider_duration_var,
                        'duration_n': self.slider_duration_n,
                        'duration_t': self.slider_duration_t,
                        'duration_sp': self.slider_duration_sp,
                        'intensity_mean': self.slider_intensity_mean,
                        'intensity_var': self.slider_intensity_var,
                        'intensity_n': self.slider_intensity_n,
                        'intensity_t': self.slider_intensity_t,
                        'intensity_sp': self.slider_intensity_sp,
                     },
             'disc': {'prob_duration': self.prob_duration,
                      'psych_duration_t': self.psych_duration_t,
                      'psych_duration_sp': self.psych_duration_sp,
                      'psych_duration_n': self.psych_duration_n,
                      'prob_intensity': self.prob_intensity,
                      'psych_intensity_t': self.psych_intensity_t,
                      'psych_intensity_sp': self.psych_intensity_sp,
                      'psych_intensity_n': self.psych_intensity_n,
                   }
            }
        return d
    
    def as_matlab_format(self):
        d = {str('slider'): np.array([self.slider_duration_mean.astype(np.float64),
                                   self.slider_duration_var.astype(np.float64),
                                   self.slider_duration_n.astype(np.float64),
                                   self.slider_duration_t.astype(np.float64),
                                   self.slider_duration_sp.astype(np.float64),
                                   self.slider_intensity_mean.astype(np.float64),
                                   self.slider_intensity_var.astype(np.float64),
                                   self.slider_intensity_n.astype(np.float64),
                                   self.slider_intensity_t.astype(np.float64),
                                   self.slider_intensity_sp.astype(np.float64)
                                  ]).T,
             str('slider_colnames'): [str('duration_mean'),
                                     str('duration_var'),
                                     str('duration_n'),
                                     str('duration_t'),
                                     str('duration_sp'),
                                     str('intensity_mean'),
                                     str('intensity_var'),
                                     str('intensity_n'),
                                     str('intensity_t'),
                                     str('intensity_sp')],
             str('disc'): np.array([self.prob_duration.astype(np.float64),
                                   self.psych_duration_t.astype(np.float64),
                                   self.psych_duration_sp.astype(np.float64),
                                   self.psych_duration_n.astype(np.float64),
                                   self.prob_intensity.astype(np.float64),
                                   self.psych_intensity_t.astype(np.float64),
                                   self.psych_intensity_sp.astype(np.float64),
                                   self.psych_intensity_n.astype(np.float64)
                                  ]).T,
             str('disc_colnames'): [str('prob_duration'),
                                   str('psych_duration_t'),
                                   str('psych_duration_sp'),
                                   str('psych_duration_n'),
                                   str('prob_intensity'),
                                   str('psych_intensity_t'),
                                   str('psych_intensity_sp'),
                                   str('psych_intensity_n')]
            }
        return d
    
    def __getstate__(self):
        d = self.to_dict()
        d['dataframe'] = d['dataframe'].__getstate__()
        return d
    
    @staticmethod
    def _merge_arrays(n, xs, ys):
        if any([not isinstance(arg, tuple) for arg in (n, xs, ys)]):
            raise TypeError("_merge_arrays expects all inputs to be tuples of the same length. Got types n={0}, xs={1}, ys={2}".format(type(n), type(xs), type(ys)))
        if any([len(arg)!=len(n) for arg in (n, xs, ys)]):
            raise TypeError("_merge_arrays expects all inputs to be tuples of the same length. Got lengths n={0}, xs={1}, ys={2}".format(len(n), len(xs), len(ys)))
        
        
        unique_xs, inds = np.unique(np.concatenate(xs, axis=-1), axis=-1, return_inverse=True)
        concat_n = np.concatenate(n, axis=-1)
        concat_ys = np.concatenate(ys, axis=-1)
        out_n = []
        out_ys = []
        for ind_x in range(unique_xs.shape[-1]):
            ind = inds==ind_x
            temp_n = concat_n[ind]
            temp_ys = concat_ys[..., ind]
            out_n.append(np.sum(temp_n, axis=-1))
            out_ys.append(np.sum(temp_ys*temp_n, axis=-1)/out_n[-1])
        out_n = np.array(out_n)
        out_ys = np.array(out_ys)
        out_ys = np.rollaxis(np.array(out_ys), 0, out_ys.ndim)
        return out_n, unique_xs, out_ys

def parse_input(script_name = "fits_module.py", argv=None):
    if argv is None:
        argv = sys.argv
    script_help = """
{0} help
Sintax:
{0} [option flag] [option value]
 
{0} -h [or --help] displays help
 
Optional arguments are:
'-t' or '--task': Integer that identifies the task number when running multiple tasks
    in parallel. By default it is one based but this behavior can be
    changed with the option --task_base. [Default 1]
'-nt' or '--ntasks': Integer that identifies the number tasks working in parallel [Default 1]
'-tb' or '--task_base': Integer that identifies the task base. Can be 0 or 1, indicating
    the task number of the root task. [Default 1]
'-m' or '--method': String that identifies the fit method. Available values are 'None', 'slider_disc',
'disc', 'slider_disc_2alpha' and 'disc_2alpha'. [Default None]
'-o' or '--optimizer': String that identifies the optimizer used for fitting.
    Available values are 'cma', scipy's 'basinhopping', all the
    scipy.optimize.minimize and scipy.optimizer.minimize_scalar methods.
    WARNING, cma is suited for problems with more than one dimensional
    parameter spaces. If the optimization is performed on a single
    dimension, the optimizer is changed to 'Nelder-Mead' before
    processing the supplied optimizer_kwargs. If one of the
    minimize_scalar methods is supplied but more than one parameter is
    being fitted, a ValueError is raised. [Default cma]
'-s' or '--save': This flag takes no values. If present it saves the figure.
'--fit': This flag takes no values. If present it performs the fit for the selected
    method. By default, this flag is always set.
'--no-fit': This flag takes no values. If present no fit is performed for the selected
    method. This flag should be used when it is only necesary to plot the results.
'-sf' or '--suffix': A string suffix to paste to the filenames. [Default '']
'--merge': Can be 'None', 'all', 'all_sessions' or 'all_subjects'. This parameter
    controls if and how the subject-session data should be merged before
    performing the fits. If merge is set to 'all', all the data is merged
    into a single "subjectSession". If merge is 'all_sessions', the
    data across all sessions for the same subject is merged together.
    If merge is 'all_subjects', the data across all subjects for a
    single session is merged. For all the above, the experiments are
    always treated separately. If merge is None, the data of every
    subject and session is treated separately. [Default 'None']
'-e' or '--experiment': Can be 'humans', 'human_slider', 'human_disc' or 'rats'.
    Indicates the experiment that you wish to fit. If set to
    'humans', all experiment data from human subjects that
    performed both the slider and discrimination experiments.
    If 'human_slider', it will only use the data from the subjects
    with slider experiment. A similar thing is done for the
    'human_disc'. 'rats' just uses the rats' data, which has
    discrimination experiments only, and separate tasks for
    each rat. [Default 'humans']
    WARNING: is case insensitive.
'-g' or '--debug': Activates the debug messages
'-v' or '--verbose': Activates info messages (by default only warnings and errors
    are printed).
'-w': Override an existing saved fitter. This flag is only used if the 
'--fit' flag is not disabled. If the flag '-w' is supplied, the script
    will override the saved fitter instance associated to the fitted
    subjectSession. If this flag is not supplied, the script will skip
    the subjectSession's that were already fitted.
'--fits_path': The path to the directory where the fit results should
    be saved or loaded from. This path can be absolute or
    relative. Be aware that the default is relative to the
    current directory. [Default 'fits']
 
The following argument values must be supplied as JSON encoded strings.
JSON dictionaries are written as '{{"key":val,"key2":val2}}'
JSON arrays (converted to python lists) are written as '[val1,val2,val3]'
Note that the single quotation marks surrounding the brackets, and the
double quotation marks surrounding the keys are mandatory. Furthermore,
if a key value should be a string, it must also be enclosed in double
quotes.

'--fixed_parameters': A dictionary of fixed parameters. The dictionary must be written as
    '{{"fixed_parameter_name":fixed_parameter_value,...}}'. For example,
    '{{"duration_x0":0.2,"duration_var0":0.5}}'. Note that the value null
    can be passed as a fixed parameter. In that case, the
parameter will be fixed to its default value or, if the flag
-f is also supplied, to the parameter value loaded from the
    previous fitting round.
    Depending on the used method, some parameters may be fixed
    a posteriori.

'--start_point': A dictionary of starting points for the fitting procedure.
    The dictionary must be written as '{{"parameter_name":start_point_value,etc}}'.
    If a parameter is omitted, its default starting value is used. You only need to specify
    the starting points for the parameters that you wish not to start at the default
    start point. Default start points are estimated from the subjectSession data.

'-bo' or '--bounds': A dictionary of lower and upper bounds in parameter space.
    The dictionary must be written as '{{"parameter_name":[low_bound_value,up_bound_value],etc}}'
    As for the --start_point option, if a parameter is omitted, its default bound is used.

'--optimizer_kwargs': A dictionary of options passed to the optimizer with a few additions.
    If the optimizer is cma, refer to fmin in cma.py for the list of
    posible cma options. The additional option in this case is only
    'restarts':INTEGER that sets the number of restarts used in the cmaes fmin
    function.
    If 'basinhopping' is selected, refer to scipy.optimize.basinhopping for
    a detailed list of all the available options.
    If another scipy optimizer is selected, refer to scipy minimize for
    posible fmin options. The additional option in this case is
    the 'repetitions':INTEGER that sets the number of independent
    repetitions used by repeat_minize to find the minimum.
    [Default depends on the optimizer. If 'cma', '{{"restarts":1}}'.
    If 'basinhopping', '{{"stepsize":0.25,"minimizer_kwargs":{{"method":"Nelder-Mead"}},"T":10.,"niter":100,"interval":10}}.
    If not 'cma' or 'basinhopping', '{{"disp": False, "maxiter": 1000, "maxfev": 10000, "repetitions": 10}}']
    Note that the basinhopping can also accept options 'take_step', 'accept_test'
    and 'callback' which must be callable. To achieve this functionality
    you must pass the callable's full string definition with proper
    indentation. This string will be evaled and the returned value
    will be used to set the callable. Keep in mind that at the moment
    the string is evaled, 4 variables will be available:
    self: the Fitter instance
    start_point: a numpy array with the starting points for the fitted parameters,
    bounds: a 2D numpy array of shape (2,len(start_point)). bounds[0]
        is the lower bound and bounds[1] is the upper bound.
    optimizer_kwargs: a dict with the optimizer keyword arguments
        A default take_step and accept_test method is used. The latter
        only checks if the basinhopping's solution is within the bounds.
        The former makes normally distributed steps with standard
        deviation equal to stepsize*(bounds[1]-bounds[0]).
 
'-f' or '--start_point_from_fit_output': A flag that tells the script to set the unspecified start_points
    equal to the results of a previously saved fitting round. After the flag
    the user must pass a dictionary of the form:
    '{{"method":"value","optimizer":"value","suffix":"value"}}'
    where the values must be the corresponding method, optimizer,
    suffix and confidence_mapping_method used by the
    previous fitting round. The script will then try to load the fitted parameters
    from the file:
    {{fits_path}}/{{experiment}}_fit_{{method}}_subject_{{name}}_session_{{session}}_{{optimizer}}{{suffix}}.pkl
    depending on the cmapmeth value. The fits_path value will
    be the one supplied with the option --fits_path.
    The experiment, name and session are taken from the subjectSession that
    is currently being fitted, and the method, optimizer and suffix are the
    values passed in the previously mentioned dictionary.
 
'--performance_filters': This is used to drop the subjects that have performance
    below an admisible level before fitting or plotting.
    Admisible performance is defined by a list criteria.
    Each criteria must contain a query string, with which
    to query the subject's trials upon which to compute
    performance, and a float from 0 to 1 which is the
    lowest admisible performance value.
    Example: We want to drop the human subjects that
    have less than 70% performance on their vertical
    psychometric data for the duration and intensity
    tasks separately. This is achieved with a list of
    two criteria. Each criteria is itself a list too.
    The query string for the human intensity discrimination
    task vertical psychometric is the following
    experiment=="human_disc_dur" and intensity1==10
    and duration1==300. Thus, the json string that should
    be fed in is:
    [["experiment==\"human_dur_disc\" and intensity1==10 and duration1==300", 0.7],
    ["experiment==\"human_int_disc\" and intensity1==10 and duration1==300", 0.7]]
    Note that the string must contain the special
    characters \ and " to be decoded properly.

Example:
python {0} -t 1 -n 1 --save
 """. format(script_name)
 
    options =  {'task':1,'ntasks':1,'task_base':1,'method':'None','optimizer':'cma','save':False,
                'fit':True,'suffix':'','save_plot_handler':False,'load_plot_handler':False,
                'merge':'all_sessions','fixed_parameters':None,'start_point':{},'bounds':{},
                'optimizer_kwargs':{},'experiment':'humans','debug':True,
                'plot_merge':None, 'verbose':False, 'show': False,
                'start_point_from_fit_output':None,'override':False,
                'fits_path':'fits', 'performance_filters': None}
    if '-g' in argv or '--debug' in argv:
        options['debug'] = True
        logging.basicConfig(level=logging.DEBUG)
        package_logger.setLevel(logging.DEBUG)
    elif '-v' in argv or '--verbose' in argv:
        options['verbose'] = True
        logging.disable(logging.DEBUG)
        logging.basicConfig(level=logging.INFO)
        package_logger.setLevel(logging.INFO)
    else:
        logging.disable(logging.INFO)
        logging.basicConfig(level=logging.WARNING)
        package_logger.setLevel(logging.WARNING)
    expecting_key = True
    json_encoded_key = False
    key = None
    for i,arg in enumerate(argv[1:]):
        package_logger.debug('Argument {0} found in position {1}'.format(arg,i))
        if expecting_key:
            if arg=='-t' or arg=='--task':
                key = 'task'
                expecting_key = False
            elif arg=='-nt' or arg=='--ntasks':
                key = 'ntasks'
                expecting_key = False
            elif arg=='-tb' or arg=='--task_base':
                key = 'task_base'
                expecting_key = False
            elif arg=='-m' or arg=='--method':
                key = 'method'
                expecting_key = False
            elif arg=='--merge':
                key = 'merge'
                expecting_key = False
            elif arg=='-o' or arg=='--optimizer':
                key = 'optimizer'
                expecting_key = False
            elif arg=='-s' or arg=='--save':
                options['save'] = True
            elif arg=='-g' or arg=='--debug':
                continue
            elif arg=='-v' or arg=='--verbose':
                continue
            elif arg=='--fit':
                options['fit'] = True
            elif arg=='--no-fit':
                options['fit'] = False
            elif arg=='-w':
                options['override'] = True
            elif arg=='-sf' or arg=='--suffix':
                key = 'suffix'
                expecting_key = False
            elif arg=='--fits_path':
                key = 'fits_path'
                expecting_key = False
            elif arg=='-e' or arg=='--experiment':
                key = 'experiment'
                expecting_key = False
            elif arg=='--fixed_parameters':
                key = 'fixed_parameters'
                expecting_key = False
                json_encoded_key = True
            elif arg=='--start_point':
                key = 'start_point'
                expecting_key = False
                json_encoded_key = True
            elif arg=='-bo' or arg=='--bounds':
                key = 'bounds'
                expecting_key = False
                json_encoded_key = True
            elif arg=='--optimizer_kwargs':
                key = 'optimizer_kwargs'
                expecting_key = False
                json_encoded_key = True
            elif arg=='-f' or arg=='--start_point_from_fit_output':
                key = 'start_point_from_fit_output'
                expecting_key = False
                json_encoded_key = True
            elif arg=='--performance_filters':
                key = 'performance_filters'
                expecting_key = False
                json_encoded_key = True
            elif arg=='--save_plot_handler':
                options['save_plot_handler'] = True
            elif arg=='--load_plot_handler':
                options['load_plot_handler'] = True
            elif arg=='--plot_merge':
                key = 'plot_merge'
                expecting_key = False
            elif arg=='--show':
                options['show'] = True
            elif arg=='-h' or arg=='--help':
                print(script_help)
                sys.exit(2)
            else:
                raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
        else:
            expecting_key = True
            if key in ['task','ntasks','task_base']:
                options[key] = int(arg)
            elif json_encoded_key:
                try:
                    options[key] = json.loads(arg)
                except Exception as e:
                    exc_type, exc_inst, tb = sys.exc_info()
                    if __is_py3__:
                        raise type(e)(str(e) + '\nHappens for arg={0}'.format(arg)).with_traceback(tb)
                    else:
                        from .py2_reraise import reraise
                        reraise(e, '\nHappens for arg={0}'.format(arg), tb)
                    raise
                json_encoded_key = False
            elif key in ['merge', 'method', 'plot_merge']:
                options[key] = None if arg=='None' else arg.lower()
            else:
                options[key] = arg
    if options['debug']:
        options['debug'] = True
        options['optimizer_kwargs']['disp'] = True
    elif options['verbose']:
        options['optimizer_kwargs']['disp'] = True
    else:
        options['optimizer_kwargs']['disp'] = False
    if not expecting_key:
        raise RuntimeError("Expected a value after encountering key '{0}' but no value was supplied".format(arg))
    if options['task_base'] not in [0,1]:
        raise ValueError('task_base must be either 0 or 1')
    # Shift task from 1 base to 0 based if necessary
    options['task']-=options['task_base']
    options['experiment'] = options['experiment'].lower()
    if options['experiment'] not in ['humans','human_slider','human_disc','rats']:
        raise ValueError("Unknown experiment supplied: '{0}'. Available values are 'humans', 'human_slider', 'human_disc' or 'rats'".format(options['experiment']))
    options['method'] = None if options['method']=='None' else options['method']
    
    if not options['start_point_from_fit_output'] is None:
        keys = options['start_point_from_fit_output'].keys()
        if (not 'method' in keys) or (not 'optimizer' in keys) or (not 'suffix' in keys):
            raise ValueError("The supplied dictionary for 'start_point_from_fit_output' does not contain the all the required keys: 'method', 'optimizer' and 'suffix'")
    
    if not os.path.isdir(options['fits_path']):
        raise ValueError('Supplied an invalid fits_path value: {0}. The fits_path must be an existing directory.'.format(options['fits_path']))
    
    package_logger.debug('Parsed options: {0}'.format(options))
    
    return options

def prepare_fit_args(fixed_parameters, start_point, fname):
    temp = load_Fitter_from_file(fname)
    raw_loaded_parameters = temp.get_parameters_dict_from_fit_output(temp._fit_output)
    previous_fitted_parameters = {par: val for par, val in iteritems(raw_loaded_parameters) if par in temp.get_fitted_parameters()}
    package_logger.debug('Loaded parameters: {0}'.format(prettyformat(raw_loaded_parameters)))
    package_logger.debug('Fitted parameters of previous fit: {0}'.format(prettyformat(previous_fitted_parameters)))
    temp_start_point = previous_fitted_parameters.copy()
    temp_fixed_parameters = {}
    
    if fixed_parameters is not None:
        for k in fixed_parameters.keys():
            if fixed_parameters[k] is None:
                try:
                    temp_fixed_parameters[k] = temp_start_point[k]
                except:
                    temp_fixed_parameters[k] = raw_loaded_parameters[k]
            else:
                temp_fixed_parameters[k] = fixed_parameters[k]
    temp_start_point.update(start_point)
    package_logger.debug('Prepared fixed_parameters = {0}'.format(prettyformat(temp_fixed_parameters)))
    package_logger.debug('Prepared start_point = {0}'.format(prettyformat(temp_start_point)))
    return temp_fixed_parameters, temp_start_point

def main(task=1, ntasks=1, task_base=1, method=None, optimizer='cma',
         save=False, fit = True, suffix = '', save_plot_handler = False,
         load_plot_handler = False, merge = 'all_sessions',
         fixed_parameters = None, start_point = {}, bounds = {},
         optimizer_kwargs = {}, experiment = 'humans', debug = True,
         plot_merge = None, verbose = False, show =  False,
         start_point_from_fit_output = None, override = False,
         fits_path = 'fits', performance_filters =  None):
    
    # Prepare subjectSessions list
    alldata = io.AllData('/home/lpaz/Dropbox/Luciano/Duration/leaky/raw_data')
    subject_type = 'humans'
    if experiment=='humans':
        data = io.get_humans_with_all_experiments(alldata)
    elif experiment=='human_slider':
        data = alldata.data.query("experiment=='slider'")
    elif experiment=='human_disc':
        data = alldata.data.query("experiment=='human_int_disc' or experiment=='human_dur_disc'")
    elif experiment=='rats':
        data = alldata.data.query("experiment=='rat_disc'")
        subject_type = 'rats'
    package_logger.debug('Total number of trials = {0}'.format(data.shape[0]))
    
    if method is None:
        if subject_type=='rats':
            method = 'disc_ls'
        else:
            if experiment=='human_slider':
                method = 'slider_ls'
            elif experiment=='human_disc':
                method = 'disc_ls'
            else:
                method = 'slider_disc_ls'
    
    if performance_filters:
        package_logger.debug('Will filter subjects based group performance'
                             'on the following queries:\n {0}'.format(performance_filters))
        data = io.drop_subjects_by_performance(data, performance_filters)
    
    if merge!='all':
        if merge is None:
            temp = data.groupby(['subject', 'session'])
        elif merge== 'all_sessions':
            temp = data.groupby(['subject'])
        elif merge== 'all_subjects':
            temp = data.groupby(['session'])
        data = [temp.get_group(k) for k in sorted(temp.groups.keys())]
    else:
        data = [data]
    package_logger.debug('Total number of subject session pairs = {0}'.format(len(data)))
    package_logger.debug('Total number of subject session pairs that will be fitted = {0}'.format(len(range(task,len(data),ntasks))))
    
    fitter_plot_handler = None
    
    # Main loop over subjectSessions
    for i,df in enumerate(data):
        package_logger.debug('Dataframe {0} of {1}'.format(i+1, len(data)))
        if (i-task)%ntasks==0:
            package_logger.info('Task will execute for dataframe {0} of {1}'.format(i+1, len(data)))
            # Fit parameters if the user did not disable the fit flag
            if fit:
                package_logger.debug('Flag "fit" was True')
                fitter = Fitter(df,method=method,
                       optimizer=optimizer,
                       suffix=suffix,
                       fits_path=fits_path)
                fname = fitter.get_save_file_name()
                if override or not (os.path.exists(fname) and os.path.isfile(fname)):
                    # Set start point and fixed parameters to the user supplied values
                    # Or to the parameters loaded from a previous fit round
                    if start_point_from_fit_output:
                        package_logger.debug('Flag start_point_from_fit_output was present. Will load parameters from previous fit round')
                        if isinstance(start_point_from_fit_output, six.string_types):
                            fname = start_point_from_fit_output
                        else:
                            loaded_method = start_point_from_fit_output['method']
                            loaded_optimizer = start_point_from_fit_output['optimizer']
                            loaded_suffix = start_point_from_fit_output['suffix']
                            fname = Fitter_filename(method=loaded_method, name=fitter.subjects,
                                                    session=fitter.sessions, optimizer=loaded_optimizer,
                                                    suffix=loaded_suffix, fits_path=fits_path)
                        package_logger.debug('Will load parameters from file: {0}'.format(fname))
                        used_fixed_parameters, used_start_point = prepare_fit_args(fixed_parameters, start_point, fname)
                    else:
                        used_fixed_parameters = fixed_parameters
                        used_start_point = start_point
                    used_bounds = bounds
                    
                    # Perform fit and save fit output
                    fit_output = fitter.fit(fixed_parameters=used_fixed_parameters,
                                            start_point=used_start_point,
                                            bounds=used_bounds,
                                            optimizer_kwargs=optimizer_kwargs)
                    fitter.save()
                else:
                    package_logger.warning('File {0} already exists, will skip enumerated dataframe {1}. If you wish to override saved Fitter instances, supply the flag -w.'.format(fname,i))
            
            # Prepare plotable data
            if show or save or save_plot_handler:
                package_logger.debug('show, save or save_plot_fitter flags were True.')
                if load_plot_handler:
                    fname = Fitter_filename(method=method,
                                name=sorted(df['subject'].unique()),
                                session=sorted(df['session'].unique()),
                                optimizer=optimizer,
                                suffix=suffix,
                                fits_path=fits_path).replace('.pkl','_plot_handler.pkl')
                    package_logger.debug('Loading Fitter_plot_handler from file={0}'.format(fname))
                    
                    try:
                        with open(fname,'rb') as f:
                            temp = pickle.load(f)
                    except:
                        package_logger.warning('Failed to load Fitter_plot_handler from file={0}. Will continue to next subject.'.format(fname))
                        continue
                else:
                    fname = Fitter_filename(method=method,name=sorted(df['subject'].unique()),
                            session=sorted(df['session'].unique()),optimizer=optimizer,suffix=suffix,
                            fits_path=fits_path)
                    # Try to load the fitted data from file 'fname' or continue to next subject
                    fitter = load_Fitter_from_file(fname)
                    try:
                        package_logger.debug('Attempting to load fitter from file "{0}".'.format(fname))
                        fitter = load_Fitter_from_file(fname)
                    except:
                        package_logger.warning('Failed to load fitter from file {0}. Will continue to next subject.'.format(fname))
                        continue
                    # Create Fitter_plot_handler for the loaded Fitter instance
                    package_logger.debug('Getting Fitter_plot_handler with merge_plot={0}.'.format(plot_merge))
                    temp = fitter.get_Fitter_plot_handler()
                    if save_plot_handler:
                        fname = fname.replace('.pkl','_plot_handler.pkl')
                        if override or not (os.path.exists(fname) and os.path.isfile(fname)):
                            package_logger.debug('Saving Fitter_plot_handler to file={0}.'.format(fname))
                            temp.save(fname)
                        else:
                            package_logger.warning('Could not save Fitter_plot_handler. File {0} already exists. To override supply the flag -w.'.format(fname))
                # Add the new Fitter_plot_handler to the bucket of plot handlers
                package_logger.info('Adding Fitter_plot_handlers')
                if fitter_plot_handler is None:
                    fitter_plot_handler = temp
                else:
                    fitter_plot_handler+= temp
    
    # Prepare figure saver
    if save:
        if task==0 and ntasks==1:
            fname = "fits_{subject_type}_{method}{suffix}".format(
                    subject_type = subject_type,
                    method=method,suffix=suffix)
        else:
            fname = "fits_{subject_type}_{method}_{task}_{ntasks}{suffix}".format(
                    subject_type = subject_type,
                    method=method,task=task,ntasks=ntasks,suffix=suffix)
        if os.path.isdir("./figs"):
            fname = "./figs/"+fname
        fname+='.pdf'
        saver = PdfPages(fname)
    else:
        saver = None
    # Plot and show, or plot and save depending on the flags supplied by the user
    if show or save:
        package_logger.debug('Plotting results from fitter_plot_handler')
        assert not fitter_plot_handler is None, 'Could not create the Fitter_plot_handler to plot the fitter results'
        if plot_merge and load_plot_handler:
            fitter_plot_handler = fitter_plot_handler.merge(plot_merge)
        fitter_plot_handler.plot(saver=saver, show=show)
        if save:
            package_logger.debug('Closing saver')
            saver.close()
