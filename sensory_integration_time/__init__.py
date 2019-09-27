#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Full package related to the publication ...

Author: Luciano Paz
Year: 2018

"""

from __future__ import division, print_function, absolute_import, unicode_literals

__version__ = '1.0.0'

from . import data_io
from . import model
from . import fits_module
from .data_io import AllData, get_humans_with_all_experiments, \
    drop_subjects_by_performance, flag_invalid_performance
from .model import Stimulator, Leaky, prob2AFC
from .fits_module import load_Fitter_from_file, Fitter_filename, Fitter, \
    Fitter_plot_handler, Fitter_ploter, parse_input, prepare_fit_args
