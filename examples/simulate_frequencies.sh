#!/usr/bin/env bash

# this command simulates heteroplasmy frequencies using MAP parameters
mope simulate trees/mope_study_both.newick params/mope_map.params --num-families 39 -N 1000 --frequencies --heteroplasmic-only -s 16571
