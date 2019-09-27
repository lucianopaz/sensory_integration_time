#!/usr/bin/python
#-*- coding: UTF-8 -*-

"""
Utility functions module

Author: Luciano Paz
Year:2016

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import copy, collections, pandas

class Bootstraper(object):
    def __init__(self, arr, n=None, fraction=None):
        self._df_columns = None
        self._return_df = False
        self._return_series = False
        if isinstance(arr, pandas.DataFrame):
            self._df_columns = arr.columns
            self._return_df = True
            arr = np.array(arr.values)
        elif isinstance(arr, pandas.Series):
            arr = np.array(arr.values)
            self._return_series = True
        else:
            arr = np.array(arr)
        self.arr = copy.deepcopy(arr)
        self.n = n
        self.fraction = fraction
    
    def _convert_to_output_type(self, out, return_boot, return_std):
        if len(out)==0:
            return None
        if return_boot:
            m = out[0]
            if return_std:
                s = out[1]
            else:
                s = None
        else:
            m = None
            s = out[0]
        if self._return_df:
            out = tuple()
            if not m is None:
                out+= (pandas.DataFrame(data=m[None, ...], columns=self._df_columns), )
            if not s is None:
                out+= (pandas.DataFrame(data=m[None, ...], columns=self._df_columns), )
        elif self._return_series:
            if not m is None and not s is None:
                out = (pandas.Series([m, s], index=['mean', 'std']), )
            elif not m is None:
                out = (pandas.Series([m], index=['mean']), )
            elif not s is None:
                out = (pandas.Series([s], index=['std']), )
        else:
            pass
        if len(out)==1:
            out = out[0]
        return out
    
    def get_sample(self, n=None, fraction=None):
        if n is None:
            n = self.n
        if fraction is None:
            fraction = self.fraction
        sample_len = int(len(self.arr)*fraction)
        if sample_len<1:
            sample_len = 1
        sample = []
        for n_sample in range(n):
            sample.append(self.arr[np.random.permutation(len(self.arr))[:sample_len]])
        return np.array(sample)
    
    def wrapper(self, func, n=None, fraction=None, return_boot=True, return_std=False):
        sample = self.get_sample(n=n, fraction=fraction)
        temp = func(sample, axis=1)
        out = tuple()
        if return_boot:
            tempout = np.mean(temp, axis=0)
            out+= (tempout,)
        if return_std:
            tempout = np.std(temp, axis=0)
            out+= (tempout,)
        out = self._convert_to_output_type(out, return_boot, return_std)
        return out
    
    def mean(self, n=None, fraction=None, **kwargs):
        return self.wrapper(np.mean, n=n, fraction=fraction, **kwargs)
    
    def median(self, n=None, fraction=None, **kwargs):
        return self.wrapper(np.median, n=n, fraction=fraction, **kwargs)
    
    def std(self, n=None, fraction=None, **kwargs):
        return self.wrapper(np.std, n=n, fraction=fraction, **kwargs)
    
    def var(self, n=None, fraction=None, **kwargs):
        return self.wrapper(np.var, n=n, fraction=fraction, **kwargs)
    
    @staticmethod
    def Mean(arr, n, fraction, **kwargs):
        obj = Bootstraper(arr, n=n, fraction=fraction)
        return obj.mean(**kwargs)
    
    @staticmethod
    def Median(arr, n, fraction, **kwargs):
        obj = Bootstraper(arr, n=n, fraction=fraction)
        return obj.median(**kwargs)
    
    @staticmethod
    def Std(arr, n, fraction, **kwargs):
        obj = Bootstraper(arr, n=n, fraction=fraction)
        return obj.std(**kwargs)
    
    @staticmethod
    def Var(arr, n, fraction, **kwargs):
        obj = Bootstraper(arr, n=n, fraction=fraction)
        return obj.var(**kwargs)

def average_downsample(a,output_len,axis=None,ignore_nans=True,dtype=np.float):
    """
    b = average_downsample(a,output_len,axis=None,ignore_nans=True,dtype=np.float)
    
    This function downsamples a numpy array of arbitrary shape. It does
    so by computing the average value of the input array 'a' inside a
    window of steps in order to produce an array with the supplied
    output_len. The output_len does not need to be submultiple of the
    original array's shape, and the function handles the averaging of
    the edges of each window properly.
    
    Inputs:
    - a:           np.array that will be downsampled
    - output_len:  Scalar that specifies the length of the output in the
                   downsampled axis.
    - axis:        The axis of the input array along which the array
                   will be downsampled. By default axis is None, and in
                   that case, the downsampling is performed on the
                   flattened array.
    - ignore_nans: Bool that specifies whether to ignore nans in while
                   averaging or not. Default is to ignore.
    - dtype:       Specifies the output array's dtype. Default is
                   np.float
    
    Output:
    - b:           The downsampled array. If axis is None, it will be a
                   flat array of shape equal to (int(output_len)).
                   If axis is not None, 'b' will have the same shape as
                   'a' except for the specified axis, that will have
                   int(output_len) elements
    
    Example
    >>> import numpy as np
    >>> a = a = np.reshape(np.arange(100,dtype=np.float),(10,-1))
    >>> a[-1] = np.nan
    
    >>> utils.average_downsample(a,10)
    array([  4.5,  14.5,  24.5,  34.5,  44.5,  54.5,  64.5,  74.5,  84.5,   nan])
    
    >>> utils.average_downsample(a,10,axis=0)
    array([[ 40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.]])
    
    >>> utils.average_downsample(a,10,axis=1)
    array([[  4.5],
           [ 14.5],
           [ 24.5],
           [ 34.5],
           [ 44.5],
           [ 54.5],
           [ 64.5],
           [ 74.5],
           [ 84.5],
           [  nan]])
    
    """
    if axis is None:
        a = a.flatten()
        axis = 0
        sum_weight = 0
        window = float(a.shape[0])/float(output_len)
        b = np.zeros((int(output_len)),dtype=dtype)
    else:
        a = np.swapaxes(a,0,axis)
        sum_weight = np.zeros_like(a[0],dtype=np.float)
        b_shape = list(a.shape)
        window = float(b_shape[0]/output_len)
        b_shape[0] = int(output_len)
        b = np.zeros(tuple(b_shape),dtype=dtype)
    flat_array = a.ndim==1
    
    step_size = 1./window
    position = 0.
    i = 0
    prev_index = 0
    L = len(a)
    Lb = len(b)
    all_indeces = np.ones_like(a[0],dtype=np.bool)
    step = True
    while step:
        if ignore_nans:
            valid_indeces = np.logical_not(np.isnan(a[i]))
        else:
            valid_indeces = all_indeces
        position = (i+1)*step_size
        index = int(position)
        if prev_index==index:
            weight = valid_indeces.astype(dtype)*step_size
            sum_weight+= weight
            if flat_array:
                b[index]+= a[i]*weight if valid_indeces else 0.
            else:
                b[index][valid_indeces]+= a[i][valid_indeces]*weight[valid_indeces]
        elif prev_index!=index:
            weight = valid_indeces*(position-index)
            prev_weight = valid_indeces*(index+step_size-position)
            if flat_array:
                b[prev_index]+= a[i]*prev_weight if valid_indeces else 0.
                sum_weight+= prev_weight
                b[prev_index]/=sum_weight
                if index<Lb:
                    b[index]+= a[i]*weight if valid_indeces else 0.
            else:
                b[prev_index][valid_indeces]+= a[i][valid_indeces]*prev_weight[valid_indeces]
                sum_weight+= prev_weight
                b[prev_index]/=sum_weight
                if index<Lb:
                    b[index][valid_indeces]+= a[i][valid_indeces]*weight[valid_indeces]
            sum_weight = weight
        
        prev_index = index
        
        i+=1
        if i==L:
            step = False
            if index<Lb:
                b[index]/=sum_weight
    b = np.swapaxes(b,0,axis)
    return b
