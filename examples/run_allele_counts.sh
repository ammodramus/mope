#!/usr/bin/env bash

# this command runs 10000 MCMC iterations with 500 MCMC chains ("walkers") on
# allele frequency data

# this assumes you have downloaded the precomputed allele frequency transitions
# and that they are stored in transitions/. This can be accomplished with 
# `mope download-transitions`, for example.

# remove --debug to stop printing all proposed evaluations (lines for which
# start with @@)
mope run data/data_example_allele_counts.txt trees/mope_study_both.newick transitions/drift_transitions.h5 transitions/bottleneck_transitions.h5 data/data_example_allele_counts_ages.txt 10000 --debug --num-walkers 500
