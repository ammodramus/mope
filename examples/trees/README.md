This directory contains three example ontogenetic phylogenies specified in the
modified NEWICK format used by mope. These three trees correspond to the three
trees shown and tested in the [original mope
manuscript](https://www.biorxiv.org/content/early/2017/10/17/204479). 

Each node requires a unique name, and a length. Only alphanumeric characters
and underscores are allowed in node names.

Node lengths specify the name of the parameter pair (i.e., the genetic drift
and mutation parameters) associated with the branch. Optionally, this parameter
name may be multiplied by an age variable, indicating that this parameter is to
be interpreted as a branch length that depends on some age.

It is also possible to specify that the genetic drift for a certain parameter
is to be modeled as a bottleneck. This done by appending `^` to the parameter
name. The `eoo` branch components in these three trees are marked as such,
representing the hypothesized bottleneck during early oogenesis..

These three ways of specifying a node are demonstrated here for a node named
`mother_blood`:

```
mother_blood:blo               # simple genetic drift, no dependence on age
mother_blood:blo*mother_age    # rate of accumulation of drift, with mother_age
mother_blood:blo^              # mother_blood is a bottleneck
```
