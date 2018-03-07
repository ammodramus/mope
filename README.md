mope
======

Mitochondrial Ontogenetic Phylogeny Estimation
-----------------------------------------------

Mope is a program for making inferences about the dynamics of mitochondrial
heteroplasmy during different life stages, using approaches from population
genetics and phylogenetics. See the preprint
[here](https://www.biorxiv.org/content/early/2017/10/17/204479).


Requirements
--------------

Mope is written in Python and requires Python 2.7+ or Python 3.1+. The
following python module are required:
 - [Numpy](http://numpy.org)
 - [Scipy](http://scipy.org)
 - [Pandas](http://pandas.pydata.org)
 - [H5py](http://h5py.org) (for HDF5 data processing)
 - [emcee](http://dan.iel.fm/emcee/current/) for Ensemble MCMC machinery
 - [future](http://python-future.org/) for Python 2/3 compatibility
 - [lru-dict](https://pypi.python.org/pypi/lru-dict/) for fast caching
 
All of these, except for emcee, future, and lru-dict, are included with the
[Anaconda Python distribution](https://www.anaconda.com/).

To install these dependencies, you can use pip:

```
# pip install -U numpy scipy pandas h5py emcee future lru-dict
```

Mope also requires some compilation of code written in C into python modules.
This requires Python development headers and libraries. The [GNU Scientific
Library](https://www.gnu.org/software/gsl/) development libraries are also
required. On Ubuntu, these can be installed with the following command:

```
# apt-get install python-dev libgsl0-dev
```

Installation
---------------

The easiest way to install mope is to use pip:

```
# pip install mope
```

This command may require superuser priveleges on some systems; in this case,
use the command

```
# pip install --user mope
```

This will install the mope library for use in Python and at the same time
install an executable script called `mope`, which can be used to run data
analysis and inference, perform simulations, and execute a number of other
utility functionalities. This executable script should be in the user's PATH
after installation with pip.

To uninstall mope:

```
# pip uninstall mope
```

Mope can also be installed manually:

```
# git clone https://github.com/ammodramus/mope.git
# cd mope
# python setup.py install --record files.txt
```

In this case, uninstall by removing files manually using `cat files.txt | xargs
rm`.

To obtain example files and scripts, clone this repository rather than install
by pip. Mope is not supported on Windows.

Obtaining allele frequency transition files
-----------------

Likelihood calculations with mope require precomputed allele frequency
transition distributions. Mope can download these automatically:

```
mope download-transitions
```

Transition distributions can also downloaded
[here](https://berkeley.box.com/shared/static/27ghsfp00xa7g8470ndrp5ft49y7a47y.gz)
(1.1 GB,
[md5](https://berkeley.box.com/shared/static/sk4tzx5ii5a37g62rch53zpgr40gmvta.md5)).
Note that if you are downloading this file programatically (e.g., using wget or
curl), you will need to rename the downloaded file to `transitions.tar.gz`, due
to limitations of our filehosting service.

To generate allele frequency transition distributions locally, run

`mope generate-commands`

This command will generate many commands to be run in parallel so that the
transition distributions can be generated more efficiently.

Usage
--------------

For usage, try

```
# mope run --help
```

or see `examples/`.

Format specifications
==============================

Data input
------------

For inference with mope, allele frequency data can be provided in two formats,
either as allele frequency data or as allele count data. In each case, the data
takes the form of a tab-delimited table.

Required columns are the data columns, having the names of the different
tissues in the ontogenetic phylogeny (and corrosponding to the leaf nodes of
the phylogeny) and any age columns for ages corresponding to ontogenetic
phylogeny components that accumulate drift and mutation with time.

For allele frequency data, the above data columns contain the allele
frequencies. For allele count data, the data columns contain the counts of the
focal heteroplasmic allele and additional (required) coverage columns contain
the total coverage. Count columns must be named `x_n`, where `x` is the name
of a data column.

See `examples/data/` for an example dataset in each format.

Ontogenetic tree file
-----------------------

Ontogenetic trees are specified in a modified
[NEWICK](https://en.wikipedia.org/wiki/Newick_format) format. Each node
requires a unique name, and a length. Only alphanumeric characters and
underscores are allowed in node names.

Node lengths specify the name of the parameter pair (i.e., the genetic drift
and mutation parameters) associated with the branch. Optionally, this parameter
name may be multiplied by an age variable, indicating that this parameter is to
be interpreted as a branch length that depends on some age. (Note that this age
name must be a variable in the data file -- see [Data file](#data-file).)

It is also possible to specify that the genetic drift for a certain parameter
is to be modeled as a bottleneck. This done by appending `^` to the parameter
name.

These three ways of specifying a node are demonstrated here for a node named
`mother_blood`:

```
mother_blood:blo               # simple genetic drift, no dependence on age
mother_blood:blo*mother_age    # rate of accumulation of drift, with mother_age
mother_blood:blo^              # mother_blood is a bottleneck
```

For a complete example, here is the ontogenetic phylogeny used in the original
study.

```
(((mother_blood:blo*mother_age)mother_fixed_blood:fblo,(mother_cheek:buc*mother_age)mother_fixed_cheek:fbuc)somM:som,((((child_blood:blo*child_age)child_fixed_blood:fblo,(child_cheek:buc*child_age)child_fixed_cheek:fbuc)som1:som)loo:loo*mother_birth_age)eoo:eoo)emb;
```

Parameters file (simulations only)
-----------------------------------

The parameters file specifies simulation parameters. It is a
whitespace-delimited table of parameter names (first column, must match tree
file) and their values (second column). See `examples/params/` for examples.
