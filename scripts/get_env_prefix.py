#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:16:21 2019

@author: lucianopaz
"""


import sys
import re


if __name__ == "__main__":
    if len(sys.argv) == 2:
        target_name = sys.argv[1]
        inp = sys.stdin.readlines()
    if len(sys.argv) == 3:
        target_name = sys.argv[1]
        inp = sys.argv[2].splitlines()
    for line in inp:
        if re.match(r'{}\s'.format(target_name), line):
            pattern = r"(?<=(\*|\s))[^\s\*].*"
            obj = re.search(pattern, line)
            print(obj.group())
            sys.exit(0)
    raise RuntimeError("Could not find conda environment {}".format(target_name))
