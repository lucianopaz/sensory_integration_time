#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_sf.h>
#undef I  // We undefine the shorhand macro for _Complex_I because it could clash with our code

/*
 * leaky_integral_calculator.c
 * This is the C code for creating a NumPy ufunc that computes the
 * theoretical leaky integrator stochastic integrals used to compute
 * the theoretical mean and variance of the perception model.
 *
 * It uses the GNU scientific library's implementation of the incomplete
 * gamma function and the gamma function, and also copies the implementation
 * of the incomplete gamma function for negative parameters from
 * Gil, A., Ruiz-Antolín, D., Segura, J. and Temme, N. M. " Algorithm 969,
 * Computation of the Incomplete Gamma Function for Negative Values of
 * the Argument". ACM Transactions on Mathematical Software (TOMS). 43(3) 2017.
 *
 * It also defines some utility ufuncs that can turn out to be useful for
 * testing: lowergamma and uppergamma. 
 * 
 * Author: Luciano Paz
 * Year: 2017
 * 
 */

/*
 * Algorithm ACM969 is intended to be accessed through the incgamstar
 * subroutine. This subroutine handles negative 'a' values, which I don't
 * need. I link with the 'gexpan' and 'gseries' subroutines instead, so
 * I don't meddle with __float128 types in c, which are needed for 'a'
 * in incgamstar. The source of ACM969 that is included here was cropped
 * to remove all the negative a handling, to link nicely with libquadmath.
 */
extern void gexpan(double* a, double* z, double* igam, int* ierr);
extern void gseries(double* a, double* z, double* igam, int* ierr);

#define _ZEROTOLERANCE 1e-13
#define _ISZERO(X) fabs(X)<=_ZEROTOLERANCE
#define _NOTISZERO(X) fabs(X)>_ZEROTOLERANCE

double lowergamma(double a, double z, int* ierr)
{
    double out = 0.;
    *ierr = 0;
    
    if (z>0)
    {
        // Positive z is handled by the GNU scientific library upper incomplete gamma
        gsl_sf_result temp_out;
        *ierr = gsl_sf_gamma_inc_e(a, z, &temp_out);
        if (!(*ierr))
        {
            out = gsl_sf_gamma(a)-temp_out.val;
        }
    }
    else
    {
        // Negative z is handled by ACM969
        double zz = -z;
        if (zz>50.)
        {
            // Large |z| is handled with poincare type expansion
            gexpan(&a, &zz, &out, ierr);
        }
        else
        {
            // Small |z| is handled with series expansion
            gseries(&a, &z, &out, ierr);
        }
        if (!(*ierr))
        {
            out*= pow(zz, a) * gsl_sf_gamma(a);
            if (out<0)
            {
                out*= -1;
            }
        }
    }
    return out;
}

double uppergamma(double a, double z, int* ierr)
{
    double out = 0.;
    *ierr = 0;
    
    if (z>0)
    {
        // Positive z is handled by the GNU scientific library upper incomplete gamma
        gsl_sf_result temp_out;
        *ierr = gsl_sf_gamma_inc_e(a, z, &temp_out);
        if (!(*ierr))
        {
            out = temp_out.val;
        }
    }
    else
    {
        // Negative z is handled by ACM969
        double zz = -z;
        if (zz>50.)
        {
            // Large |z| is handled with poincare type expansion
            gexpan(&a, &zz, &out, ierr);
        }
        else
        {
            // Small |z| is handled with series expansion
            gseries(&a, &z, &out, ierr);
        }
        if (!(*ierr))
        {
            out*= pow(zz, a) * gsl_sf_gamma(a);
            if (out<0)
            {
                out*= -1;
            }
            out = gsl_sf_gamma(a) - out;
        }
    }
    return out;
}

static PyMethodDef LeakyIntegralCalculatorMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void lowergamma_ufunc(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *a = args[0];
    char *z = args[1];
    char *out = args[2];
    npy_intp a_step = steps[0];
    npy_intp z_step = steps[1];
    npy_intp out_step = steps[2];

    // Unused for now
    int error_flag = 0;

    for (i = 0; i < n; i++) {
        /**BEGIN main ufunc computation**/
        
        *((double *)out) = lowergamma(*(double *)a, *(double *)z, &error_flag);
        
        /**END main ufunc computation**/
        
        // Step the arrays
        a += a_step;
        z += z_step;
        out += out_step;
    }
}

static void uppergamma_ufunc(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *a = args[0];
    char *z = args[1];
    char *out = args[2];
    npy_intp a_step = steps[0];
    npy_intp z_step = steps[1];
    npy_intp out_step = steps[2];

    // Unused for now
    int error_flag = 0;

    for (i = 0; i < n; i++) {
        /**BEGIN main ufunc computation**/
        
        *((double *)out) = uppergamma(*(double *)a, *(double *)z, &error_flag);
        
        /**END main ufunc computation**/
        
        // Step the arrays
        a += a_step;
        z += z_step;
        out += out_step;
    }
}

static void leaky_integral_calculator(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *T                    = args[0];
    char *slope                = args[1];
    char *origin               = args[2];
    char *alpha                = args[3];
    char *Tau_inv              = args[4];
    char *adaptation_amplitude = args[5];
    char *adaptation_baseline  = args[6];
    char *adaptation_tau_inv   = args[7];
    char *mean_integral        = args[8];
    char *var_integral         = args[9];
    npy_intp T_step                    = steps[0];
    npy_intp slope_step                = steps[1];
    npy_intp origin_step               = steps[2];
    npy_intp alpha_step                = steps[3];
    npy_intp Tau_inv_step              = steps[4];
    npy_intp adaptation_amplitude_step = steps[5];
    npy_intp adaptation_baseline_step  = steps[6];
    npy_intp adaptation_tau_inv_step   = steps[7];
    npy_intp mean_integral_step        = steps[8];
    npy_intp var_integral_step         = steps[9];

    double temp1;
    double temp2;
    int error_flag1, error_flag2;

    for (i = 0; i < n; i++) {
        /**BEGIN main ufunc computation**/
        temp1 = 0.;
        temp2 = 0.;
        
        const double t = *(double *)T;
        const double m = *(double *)slope;
        const double b = *(double *)origin;
        const double a = *(double *)alpha;
        const double tau_inv = *(double *)Tau_inv;
        const double A = *(double *)adaptation_amplitude;
        const double B = *(double *)adaptation_baseline;
        const double A1 = A * B;
        const double A2 = A * (1-B);
        const double A3 = 2 * A1 * A2;
        const double k1 = tau_inv;
        const double k2 = tau_inv - *(double *)adaptation_tau_inv;
        const double k3 = 2*tau_inv - *(double *)adaptation_tau_inv;
        
        if (_NOTISZERO(A1)){
            if (_ISZERO(m)){
                if (_ISZERO(k1)){
                    // slope==0 and k1==0
                    temp1+= A1 * pow(fabs(b), a) * t * exp(-tau_inv * t);
                    temp2+= A1*A1 * pow(b*b, a) * t * exp(-2 * tau_inv * t);
                } else {
                    // slope==0 and k1!=0
                    temp1+= A1 * pow(fabs(b), a)/k1 * ( exp((k1 - tau_inv) * t) - exp(-tau_inv * t) );
                    temp2+= A1*A1 * 0.5 * pow(b*b, a)/k1 * ( exp(2 * (k1 - tau_inv) * t) - exp(-2 * tau_inv * t) );
                }
            } else {
                if (_ISZERO(k1)){
                    // slope!=0 and k1==0
                    const double sigma = fabs(m * t + b);
                    temp1+= A1/(a+1)/m * (pow(sigma, a+1) - pow(fabs(b), a+1)) * exp(-tau_inv * t);
                    temp2+= A1*A1/(2*a+1)/m * (pow(sigma, 2*a+1) - pow(fabs(b), 2*a+1)) * exp(-2 * tau_inv * t);
                } else {
                    // slope!=0 and k1!=0
                    const double kbm = k1*b/m;
                    const double _2kbm = 2*k1*b/m;
                    const double absmk = fabs(m/k1);
                    const double a1 = a+1;
                    const double _2a1 = 2*a+1;
                    temp1+= A1 * pow(absmk, a1) / m * exp(-kbm) * exp(-tau_inv*t) *
                           (lowergamma(a1, -k1*t-kbm, &error_flag1) - lowergamma(a1, -kbm, &error_flag2));
                    temp2+= A1*A1 * pow(0.5*absmk, _2a1) / m * exp(-_2kbm) * exp(-2*tau_inv*t) *
                           (lowergamma(_2a1, -2*k1*t-_2kbm, &error_flag1) - lowergamma(_2a1, -_2kbm, &error_flag2));
                }
            }
        }
        if (_NOTISZERO(A2)){
            if (_ISZERO(m)){
                if (_ISZERO(k2)){
                    // slope==0 and k2==0
                    temp1+= A2 * pow(fabs(b), a) * t * exp(-tau_inv * t);
                    temp2+= A2*A2 * pow(b*b, a) * t * exp(-2 * tau_inv * t);
                } else {
                    // slope==0 and k2!=0
                    temp1+= A2 * pow(fabs(b), a)/k2 * ( exp((k2 - tau_inv) * t) - exp(-tau_inv * t) );
                    temp2+= A2*A2 * 0.5 * pow(b*b, a)/k2 * ( exp(2 * (k2 - tau_inv) * t) - exp(-2 * tau_inv * t) );
                }
            } else {
                if (_ISZERO(k2)){
                    // slope!=0 and k2==0
                    const double sigma = fabs(m * t + b);
                    temp1+= A2/(a+1)/m * (pow(sigma, a+1) - pow(fabs(b), a+1)) * exp(-tau_inv * t);
                    temp2+= A2*A2/(2*a+1)/m * (pow(sigma, 2*a+1) - pow(fabs(b), 2*a+1)) * exp(-2 * tau_inv * t);
                } else {
                    // slope!=0 and k2!=0
                    const double kbm = k2*b/m;
                    const double _2kbm = 2*k2*b/m;
                    const double absmk = fabs(m/k2);
                    const double a1 = a+1;
                    const double _2a1 = 2*a+1;
                    temp1+= A2 * pow(absmk, a1) / m * exp(-kbm) * exp(-tau_inv*t) *
                           (lowergamma(a1, -k2*t-kbm, &error_flag1) - lowergamma(a1, -kbm, &error_flag2));
                    temp2+= A2*A2 * pow(0.5*absmk, _2a1) / m * exp(-_2kbm) * exp(-2*tau_inv*t) *
                           (lowergamma(_2a1, -2*k2*t-_2kbm, &error_flag1) - lowergamma(_2a1, -_2kbm, &error_flag2));
                }
            }
        }
        if (_NOTISZERO(A3)){
            if (_ISZERO(m)){
                if (_ISZERO(k3)){
                    // slope==0 and k3==0
                    temp2+= A3 * pow(b*b, a) * t * exp(-2 * tau_inv * t);
                } else {
                    // slope==0 and k3!=0
                    temp2+= A3 * pow(b*b, a)/k3 * ( exp((k3 - 2*tau_inv) * t) - exp(-2 * tau_inv * t) );
                }
            } else {
                if (_ISZERO(k3)){
                    // slope!=0 and k3==0
                    const double sigma = fabs(m * t + b);
                    temp2+= A3/(2*a+1)/m * (pow(sigma, 2*a+1) - pow(b, 2*a+1)) * exp(-2 * tau_inv * t);
                } else {
                    // slope!=0 and k3!=0
                    const double kbm = k3*b/m;
                    const double absmk = fabs(m/k3);
                    const double _2a1 = 2*a+1;
                    temp2+= A3 * pow(absmk, _2a1) / m * exp(-kbm) * exp(-2*tau_inv*t) *
                           (lowergamma(_2a1, -k3*t-kbm, &error_flag1) - lowergamma(_2a1, -kbm, &error_flag2));
                }
            }
        }
        
        *((double *)mean_integral) = temp1;
        *((double *)var_integral) = temp2;
        
        /**END main ufunc computation**/
        
        // Step the arrays
        T += T_step;
        slope += slope_step;
        origin += origin_step;
        alpha += alpha_step;
        Tau_inv += Tau_inv_step;
        adaptation_amplitude += adaptation_amplitude_step;
        adaptation_baseline += adaptation_baseline_step;
        adaptation_tau_inv += adaptation_tau_inv_step;
        mean_integral += mean_integral_step;
        var_integral += var_integral_step;
    }
}

#undef _ZEROTOLERANCE
#undef _ISZERO
#undef _NOTISZERO

/*This gives pointers to the above functions*/
PyUFuncGenericFunction main_func[1] = {&leaky_integral_calculator};
PyUFuncGenericFunction util_func1[1] = {&lowergamma_ufunc};
PyUFuncGenericFunction util_func2[1] = {&uppergamma_ufunc};


static char main_types[10] = {NPY_DOUBLE, NPY_DOUBLE,
                NPY_DOUBLE, NPY_DOUBLE,
                NPY_DOUBLE, NPY_DOUBLE,
                NPY_DOUBLE, NPY_DOUBLE,
                NPY_DOUBLE, NPY_DOUBLE};
static char util_types[3] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
                
static void *data[1] = {NULL};

char *leaky_integral_calculator_docstring = "Syntax:\n\n"
"leaky_integral_calculator(t, slope, origin, alpha, tau_inv, "
"adaptation_amplitude, adaptation_baseline, adaptation_tau_inv, "
"[, integral_mean, integral_var], / [, out=(None, None)], *, "
"where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])\n"
"\n"
"Calculate the stochastic integrals necessary to compute the "
"theoretical mean and variance from the stochastic process\n"
"\n"
"Input:\n"
"    t: time at which the to compute the integrals\n"
"    slope: slope of the sigma time dependence. sigma(t) = slope * t + origin.\n"
"    origin: origin of the sigma time dependence. sigma(t) = slope * t + origin.\n"
"    alpha: power law transformation of raw sigma to sigma(t)**alpha.\n"
"    tau_inv: inverse of the time constant, tau, of the leaky integration process.\n"
"    adaptation_amplitude: amplitude, A, of the adaptation function that\n"
"        multiplies the total raw input. This function is\n"
"        A * (B + (1 - B)*exp(-t*tau_inv_ad)).\n"
"    adaptation_baseline: baseline, B, of the adaptation function that\n"
"        multiplies the total raw input. This function is\n"
"        A * (B + (1 - B)*exp(-t*tau_inv_ad)).\n"
"    adaptation_tau_inv: inverse time constant, Tau_inv_ad, of the\n"
"        adaptation function that multiplies the total raw input. This\n"
"        function is A * (B + (1 - B)*exp(-t*tau_inv_ad)).\n"
"    Other inputs are a signature of numpy's ufuncs and is only available\n"
"    for succesfully imported extension modules as the pure python alternative\n"
"    does not implement the rest of this signature.\n"
"\n"
"Output:\n"
"    integral_mean: The mean of the stochastic integral.\n"
"    integral_var: The variance of the stochastic integral.\n\n";

char *lowergamma_docstring = "Syntax:\n\n"
"lowergamma(a, z, / [, out=(None)], *, "
"where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])\n"
"\n"
"Calculate the lower incomplete gamma function which is defined as:\n"
"\\gamma(a,z) = \\int_{0}^{z} x^{a-1} \\exp(-x) dx\n"
"\n"
"Input:\n"
"    a: the 'a' parameter of the lower incomplete gamma. Only handles positive double precision floating points\n"
"    z: the 'z' integration limit. Only handles double precision floating points\n"
"\n"
"Output: the value of the incomplete gamma function.\n\n";

char *uppergamma_docstring = "Syntax:\n\n"
"uppergamma(a, z, / [, out=(None)], *, "
"where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])\n"
"\n"
"Calculate the upper incomplete gamma function which is defined as:\n"
"\\Gamma(a,z) = \\int_{z}^{+\\infty} x^{a-1} \\exp(-x) dx\n"
"\n"
"Input:\n"
"    a: the 'a' parameter of the upper incomplete gamma. Only handles positive double precision floating points\n"
"    z: the 'z' integration limit. Only handles double precision floating points\n"
"\n"
"Output: the value of the incomplete gamma function.\n\n";

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "leaky_integral_calculator",
    NULL,
    -1,
    LeakyIntegralCalculatorMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_leaky_integral_calculator(void)
{
    PyObject *m, *leaky_integral_calculator, *d;
    PyObject *lowergamma_py, *uppergamma_py;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    leaky_integral_calculator = PyUFunc_FromFuncAndData(main_func, data, main_types, 1, 8, 2,
                                    PyUFunc_None, "leaky_integral_calculator",
                                    leaky_integral_calculator_docstring, 0);

    lowergamma_py = PyUFunc_FromFuncAndData(util_func1, data, util_types, 1, 2, 1,
                                    PyUFunc_None, "lowergamma",
                                    lowergamma_docstring, 0);

    uppergamma_py = PyUFunc_FromFuncAndData(util_func2, data, util_types, 1, 2, 1,
                                    PyUFunc_None, "uppergamma",
                                    uppergamma_docstring, 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "leaky_integral_calculator", leaky_integral_calculator);
    PyDict_SetItemString(d, "lowergamma", lowergamma_py);
    PyDict_SetItemString(d, "uppergamma", uppergamma_py);
    Py_DECREF(leaky_integral_calculator);
    Py_DECREF(lowergamma_py);
    Py_DECREF(uppergamma_py);

    return m;
}
#else
PyMODINIT_FUNC initleaky_integral_calculator(void)
{
    PyObject *m, *leaky_integral_calculator, *d;
    PyObject *lowergamma_py, *uppergamma_py;


    m = Py_InitModule("leaky_integral_calculator", LeakyIntegralCalculatorMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    leaky_integral_calculator = PyUFunc_FromFuncAndData(main_func, data, main_types, 1, 8, 2,
                                    PyUFunc_None, "leaky_integral_calculator",
                                    leaky_integral_calculator_docstring, 0);

    lowergamma_py = PyUFunc_FromFuncAndData(util_func1, data, util_types, 1, 2, 1,
                                    PyUFunc_None, "lowergamma",
                                    lowergamma_docstring, 0);

    uppergamma_py = PyUFunc_FromFuncAndData(util_func2, data, util_types, 1, 2, 1,
                                    PyUFunc_None, "uppergamma",
                                    uppergamma_docstring, 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "leaky_integral_calculator", leaky_integral_calculator);
    PyDict_SetItemString(d, "lowergamma", lowergamma_py);
    PyDict_SetItemString(d, "uppergamma", uppergamma_py);
    Py_DECREF(leaky_integral_calculator);
    Py_DECREF(lowergamma_py);
    Py_DECREF(uppergamma_py);
}
#endif
