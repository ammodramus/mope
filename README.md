mope
======

Mitochondrial Ontogenetic Phylogeny Estimation
-----------------------------------------------

`mope` is a program for making inferences about genetic drift and mutation
during different reproductive and developmental stages.


Requirements
--------------

`mope` is written in Python and requires Python 2.7+ or Python 3.1+. Additional
third-party Python packages are required, including [Numpy](http://numpy.org),
[Scipy](http://scipy.org), [Pandas](http://pandas.pydata.org),
[H5py](http://h5py.org) (for HDF5 data processing), and
[emcee](http://dan.iel.fm/emcee/current/) for Ensemble MCMC machinery. All of
these, except for emcee, are included with the
[Anaconda Python distribution](https://www.anaconda.com/).

To install these dependencies, you can use `pip`:

```
# pip install -U numpy scipy pandas h5py emcee
```

`mope` also requires some compilation of code written in C into python modules.
This requires Python development headers and libraries. The [GNU Scientific
Library](https://www.gnu.org/software/gsl/) is also required. On Ubuntu, these
can be installed with the following command:

```
# apt-get install python-dev libgsl0-dev
```


