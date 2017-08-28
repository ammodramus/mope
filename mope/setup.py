from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

link_args = ["-lgsl", "-lopenblas"]
#link_args = ["-lgsl", "-lmkl_sequential", "-lmkl_core", "-lmkl_intel_lp64"]
compile_args = ['-march=native', '-O3']
try:
    library_dirs = os.environ['LIBRARY_PATH'].split(':')
except KeyError:
    library_dirs = []
try:
    include_dirs = [np.get_include()] + os.environ['CPATH'].split(':')
except KeyError:
    include_dirs = [np.get_include()]

extensions = [
        Extension(
            '_wf',
            #['_wf.pyx'],
            ['_wf.c'],
            include_dirs = include_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args,
            library_dirs = library_dirs),
        Extension(
            '_util',
            #['_util.pyx'],
            ['_util.c'],
            include_dirs = include_dirs,
            library_dirs = library_dirs),
        Extension(
            '_binom',
            #['_binom.pyx'],
            ['_binom.c'],
            include_dirs = include_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args,
            library_dirs = library_dirs),
        Extension(
            '_interp',
            #['_interp.pyx'],
            ['_interp.c'],
            include_dirs = include_dirs,
            extra_compile_args = compile_args,
            library_dirs = library_dirs),
        Extension(
            '_likes',
            #['_likes.pyx'],
            ['_likes.c'],
            include_dirs = include_dirs,
            extra_link_args = link_args,
            library_dirs = library_dirs,
            extra_compile_args = compile_args),
        Extension(
            '_poisson_binom',
            #['_poisson_binom.pyx'],
            ['_poisson_binom.c'],
            include_dirs = include_dirs,
            library_dirs = library_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args),
        Extension(
            '_transition',
            #['_transition.pyx'],
            ['_transition.c'],
            include_dirs = include_dirs,
            library_dirs = library_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args)
        ]

setup(
        ext_modules = cythonize(extensions)
        )
