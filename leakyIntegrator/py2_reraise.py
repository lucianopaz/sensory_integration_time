#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Module that defines the function reraise, which takes enables exception
chaining with preserved tracebacks with python 2.

This module should only be imported when using python2.x or else a
syntax error will be thrown.

Author: Luciano Paz
Year: 2017

"""

from __future__ import division, print_function, absolute_import, unicode_literals

def reraise(exception, additional_message, traceback):
    raise type(exception), type(exception)(exception.message + additional_message), traceback
    
