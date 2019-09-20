#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Setup script for the leakyIntegrator package related to the publication ...
It involves the use of a Makefile during the build phase, and it does so
by calling subprocess.Popen with shell=True. This means that you must
trust this source not to have been tampered with, or install without sudo.
For a more detailed explanation of why the build_ext class was overloaded
refer to https://stackoverflow.com/a/48641638/7253166

Author: Luciano Paz
Year: 2018

"""

import os, subprocess, codecs, re, platform
import setuptools
from setuptools.command.build_ext import build_ext
from distutils.errors import DistutilsSetupError
from distutils import log as distutils_logger



extension1 = setuptools.extension.Extension('leakyIntegrator.leaky_integral_calculator',
                    sources = ['leakyIntegrator/src/leaky_integral_calculator.c'],
                    libraries = ['m',
                                 'dl',
                                 'gsl',
                                 'gslcblas',
                                 'gfortran',
                                 'quadmath'],
                    )

class specialized_build_ext(build_ext, object):
    """
    Specialized builder for incgamNEG library
    
    """
    special_extension = extension1.name
    
    def finalize_options(self):
        # The reason for this is the need to get numpy include and
        # library dirs but numpy could a priori not be available! This
        # means that you need to bootstrap numpy's installation.
        # This hackish solution was taken from
        # https://stackoverflow.com/a/21621689/7253166
        
        
        super(specialized_build_ext, self).finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())
    
    def build_extension(self, ext):
        
        if ext.name!=self.special_extension:
            # Handle unspecial extensions with the parent class' method
            super(specialized_build_ext, self).build_extension(ext)
        else:
            # Get the package build dir
            build_py = self.get_finalized_command('build_py')
            fullname = self.get_ext_fullname(ext.name)
            modpath = fullname.split('.')
            package = '.'.join(modpath[:-1])
            package_dir = build_py.get_package_dir(package)
            
            # Handle special extension
            sources = ext.sources
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'ext_modules' option (extension '%s'), "
                       "'sources' must be present and must be "
                       "a list of source filenames" % ext.name)
            sources = list(sources)
        
            if len(sources)>1:
                sources_path = os.path.commonprefix(sources)
            else:
                sources_path = os.path.dirname(sources[0])
            sources_path = os.path.realpath(sources_path)
            if not sources_path.endswith(os.path.sep):
                sources_path+= os.path.sep
            
            if not os.path.exists(sources_path) or not os.path.isdir(sources_path):
                raise DistutilsSetupError(
                       "in 'extensions' option (extension '%s'), "
                       "the supplied 'sources' base dir "
                       "must exist" % ext.name)
            
            output_dir = os.path.realpath(os.path.join(self.build_lib, package_dir))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Append the library dir
            if output_dir not in ext.library_dirs:
                ext.library_dirs.append(output_dir)
            if output_dir not in ext.runtime_library_dirs:
                ext.runtime_library_dirs.append(output_dir)
            
            plat = platform.system().lower()
            if plat.startswith('windows'):
                shared_lib = 'incgamNEG.*'
                static_lib = 'incgamNEG.*'
                mv = 'move'
            elif plat.startswith('darwin'):
                shared_lib = 'libincgamNEG.dylib'
                static_lib = 'libincgamNEG.a'
                mv = 'mv'
            else:
                shared_lib = 'libincgamNEG.so'
                static_lib = 'libincgamNEG.a'
                mv = 'mv'
            library_type = 'shared'
            output_lib = shared_lib if library_type=='shared' else static_lib
            ext.libraries.append(':{0}'.format(output_lib))
            
            shell_command = 'make {lib_type} && {mv} {source} {dest}'.format(lib_type = library_type,
                                                                             mv = mv,
                                                                             source = output_lib,
                                                                             dest = os.path.join(output_dir, output_lib))

            distutils_logger.info('Will execute the following command in with subprocess.Popen: \n{0}'.format(
                  shell_command))


            make_process = subprocess.Popen(shell_command,
                                            cwd=sources_path,
                                            #~ stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            shell=True)
            stdout, stderr = make_process.communicate()
            distutils_logger.debug(stdout)
            if stderr:
                raise DistutilsSetupError('An ERROR occured while running the '
                                          'Makefile for the {0} library. '
                                          'Error status: {1}'.format(output_lib, stderr))
            # After making the library build the c library's python interface with the parent build_extension method
            super(specialized_build_ext, self).build_extension(ext)

description = \
"""
This package is the source code linked to publication ...

It provides an interface to the leaky integrator perception model, the
data handling tools and also the modules used to perform the parameter
search.

"""

here = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setuptools.setup(name = 'leakyIntegrator',
        version = find_version('leakyIntegrator', '__init__.py'),
        ext_modules = [extension1],
        packages = setuptools.find_packages(),
        cmdclass = {'build_ext': specialized_build_ext},
        setup_requires=['numpy>=1.13'],
        install_requires = ['matplotlib>=2',
                            'numpy>=1.13',
                            'sympy>=1.1.1',
                            'scipy>=1',
                            'six',
                            'pandas>=0.21',
                            'cma>=2.5',
                            'mpmath>=1',
                           ],
        # metadata for upload to PyPI
        author="Luciano Paz",
        author_email="lpaz@sissa.it",
        description=description,
        license="MIT",
        keywords="Leaky integrator perceptual decision making model, "
                 "Stochastic dynamics, "
                 "Ornstein Ulhenbeck process, "
                 "Parametric model fit",
        url="https://github.com/lucianopaz/leakyIntegrator",
       )
