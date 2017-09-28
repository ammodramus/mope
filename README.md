mope
======

Mitochondrial Ontogenetic Phylogeny Estimation
-----------------------------------------------

Mope is a program for making inferences about genetic drift and mutation
during different reproductive and developmental stages.


Requirements
--------------

Mope is written in Python and requires Python 2.7+ or Python 3.1+. Additional
third-party Python packages are required, including [Numpy](http://numpy.org),
[Scipy](http://scipy.org), [Pandas](http://pandas.pydata.org),
[H5py](http://h5py.org) (for HDF5 data processing), and
[emcee](http://dan.iel.fm/emcee/current/) for Ensemble MCMC machinery. All of
these, except for emcee, are included with the
[Anaconda Python distribution](https://www.anaconda.com/).

To install these dependencies, you can use pip:

```
# pip install -U numpy scipy pandas h5py emcee
```

Mope also requires some compilation of code written in C into python modules.
This requires Python development headers and libraries. The [GNU Scientific
Library](https://www.gnu.org/software/gsl/) is also required. On Ubuntu, these
can be installed with the following command:

```
# apt-get install python-dev libgsl0-dev
```

Installation
---------------

The easiest way to install mope is to use pip:

```
# pip install mope
```

This may require superuser priveleges on some systems; in this case, use the
command

```
# pip install --user mope
```

This will install the mope library for use in Python (`import mope`), and it
will install an executable script called mope.

To uninstall mope:

```
# pip uninstall mope
```

Usage
--------------


Ontogenetic tree file
-----------------------

Ontogenetic trees are specified in a modified
[NEWICK](https://en.wikipedia.org/wiki/Newick_format) format. Each node
requires a unique name, and a length. Only alphanumeric characters and
underscores are allowed in node names.

Node lengths specify the name of the parameter pair (vz., genetic drift and
mutation parameters) associated with the branch. Optionally, this parameter
name may be multiplied by an age variable, indicating that this parameter is to
be interpreted as a branch length that depends on some age. (Note that this age
name must be a variable in the data file -- see [Data file](#data-file).)

It is also possible to specify that the genetic drift for a certain parameter
is modeled as a bottleneck. This done by appending `^` to the parameter name.

These three ways of specifying a node are demonstrated here:

```
mother_blood:blo               # simple genetic drift, no dependence on age
mother_blood:blo*mother_age    # rate of accumulation of drift, with mother_age
mother_blood:blo^              # blo is a bottleneck
```


```
(((mother_blood:blo*mother_age)mother_fixed_blood:fblo,(mother_cheek:buc*mother_age)mother_fixed_cheek:fbuc)somM:som,((((child_blood:blo*child_age)child_fixed_blood:fblo,(child_cheek:buc*child_age)child_fixed_cheek:fbuc)som1:som)loo:loo*mother_birth_age)eoo:eoo)emb;

(((mother_blood:blo*mother_age)mother_fixed_blood:fblo,(mother_cheek:buc*mother_age)mother_fixed_cheek:fbuc)somM:som,((((child_blood:blo*child_age)child_fixed_blood:fblo,(child_cheek:buc*child_age)child_fixed_cheek:fbuc)som1:som)loo:loo*mother_birth_age)eoo:eoo)emb;
```

Parameters file (simulations only)
-----------------------------------

The parameters file specifies simulation parameters. It is a
whitespace-delimited table of parameter names (first column, must match tree
file) and their values (second column).
