from setuptools import setup, Extension, find_packages
import os
import numpy as np

requirements = [
        'emcee',
        'scipy',
        'numpy',
        'h5py',
        'pandas',
        'future',
        'lru-dict'
        ]

have_mkl = False
try:
    mklinfo = np.__config__.blas_mkl_info
    if mklinfo != {}:
        have_mkl = True
except:
    False
have_openblas = False
try:
    openblasinfo = np.__config__.openblas_info
    if openblasinfo != {}:
        have_openblas = True
except:
    pass
openblas_link_args = ["-lgsl", "-lopenblas"]
mkl_link_args = ["-lgsl", "-lmkl_sequential", "-lmkl_core", "-lmkl_intel_lp64"]
cblas_link_args = ["-lgsl", "-lblas"]
if have_mkl:
    link_args = mkl_link_args
elif have_openblas:
    link_args = openblas_link_args
else:
    link_args = cblas_link_args
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
            'mope._wf',
            ['mope/_wf.c'],
            include_dirs = include_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args,
            library_dirs = library_dirs),
        Extension(
            'mope._util',
            ['mope/_util.c'],
            include_dirs = include_dirs,
            library_dirs = library_dirs),
        Extension(
            'mope._binom',
            ['mope/_binom.c'],
            include_dirs = include_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args,
            library_dirs = library_dirs),
        Extension(
            'mope._interp',
            ['mope/_interp.c'],
            include_dirs = include_dirs,
            extra_compile_args = compile_args,
            library_dirs = library_dirs),
        Extension(
            'mope._likes',
            ['mope/_likes.c'],
            include_dirs = include_dirs,
            extra_link_args = link_args,
            library_dirs = library_dirs,
            extra_compile_args = compile_args),
        Extension(
            'mope._poisson_binom',
            ['mope/_poisson_binom.c'],
            include_dirs = include_dirs,
            library_dirs = library_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args),
        Extension(
            'mope._transition',
            ['mope/_transition.c'],
            include_dirs = include_dirs,
            library_dirs = library_dirs,
            extra_link_args = link_args,
            extra_compile_args = compile_args)
        ]

entry_points = {'console_scripts': ['mope = mope.cli:main']}


setup(
        name='mope',
        version = '0.5',
        description='Molecular ontogenetic phylogeny estimation',
        author='Peter Wilton',
        author_email='pwilton@berkeley.edu',
        license='GPLv3',
        install_requires = requirements,
        python_requires = '>=2.7',
        packages = find_packages(),
        entry_points = entry_points,
        ext_modules=extensions)
