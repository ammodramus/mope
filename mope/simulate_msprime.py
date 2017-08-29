from __future__ import division

import re
import sys
import argparse
import numpy as np
import numpy.random as npr
import scipy.stats as st
import pandas as pd

import msprime

from scipy.misc import comb
from collections import OrderedDict

import likelihoods_somatic as lis

import _wf
import _util
import newick
import ages
from util import *

def get_parameters(
        params_file,
        tree,
        num_families = None,
        ages = None):
    bls = {}
    mrs = {}
    ages_found = False
    ret_ages = {}
    with open(params_file) as fin:
        for line in fin:
            line = line.strip()
            if '---' in line:
                if not num_families:
                    break
                ages_found = True
                mathpat = '([^\s]+)\s+([^\s]+)\s*([\+-])\s*([^\s]+)'
                mathre =  re.compile(mathpat)
                normpat = '([^\s]+)\s+([^\s]+)\s+([^\s]+)'
                normre = re.compile(normpat)
                continue
            if not ages_found:
                spline = [el for el in line.split() if el != '']
                name = spline[0]
                bl = float(spline[1])
                mr = float(spline[2])
                if name == tree.name:
                    mrs[name] = mr
                    continue
                else:
                    bls[name] = bl
                    mrs[name] = mr
            else:  # (ages found)
                normmat = normre.match(line)
                mathmat = mathre.match(line)
                if ages is not None:
                    age_indices = np.sort(npr.choice(ages.shape[0],
                            size = num_families,
                            replace = True))
                if normmat:
                    name, mean, sd = normmat.group(1,2,3)
                    if ages is None:
                        mean = float(mean)
                        sd = float(sd)
                        family_ages = np.maximum(
                                npr.normal(mean, sd, size = num_families), 0.4)
                    else:  # ages is not None
                        try:
                            family_ages = ages[name].iloc[age_indices].values
                        except:
                            raise ValueError(
                                    '{} not found in ages dataframe'.format(
                                        name))
                    ret_ages[name] = family_ages

                elif mathmat:
                    name, arg1, signstr, arg2 = mathmat.group(1,2,3,4)
                    if signstr == '-':
                        ret_ages[name] = ret_ages[arg1]-ret_ages[arg2]
                    elif signstr == '+':
                        ret_ages[name] = ret_ages[arg1]+ret_ages[arg2]
                    else:
                        raise ValueError('invalid expression in {}'.format(
                                params_file))
                else:
                    raise ValueError('bad age line: {}'.format(line))


    for node in tree.walk('postorder'):
        if node == tree:
            continue
        varname = node.varname
        if varname not in bls:
            raise ValueError('parameter(s) for {} not found in {}'.format(
                varname, params_file))
    if num_families:
        family_idxs = np.arange(num_families)
        ret_ages['family'] = family_idxs
        ret_ages = pd.DataFrame(ret_ages)
        return bls, mrs, ret_ages
    else:
        return bls, mrs

def get_stationary_distn(N, a, b, double_beta = False, ppoly = None):
    assert 3/2 == 1.5
    if double_beta:
        assert ppoly is not None, "ppoly must not be None for double beta"
        p = lis.discretize_double_beta_distn(a, b, N, ppoly)
    else:
        p = lis.discretize_beta_distn(a, b, N)
        p /= p.sum()
    return p

def get_header(leaves, frequencies):
    if not frequencies:
        header = leaves + ['coverage', 'age']
    else:
        header = leaves + ['age']
    return header

def run_simulations(reps, tree, bls, mrs, N, mean_coverage,
        family_indices, ages, genome_size, free_recomb):
    '''
    num families
    tree
    bls
    mrs
    N
    stationary_distn
    binomial_sample_size
    family_indices
    '''

    leaves = []
    multipliers = {}
    for node in tree.walk('postorder'):
        if node.is_leaf:
            leaves.append(node.name)
        if node.multipliername is not None:
            multipliers[node.multipliername] = node.multipliervalues

    multipliers = pd.DataFrame(multipliers)
    mu = 1e-8
    N0 = mrs[tree.name] / (4*mu)
    pop_sizes = {}
    depths = {}
    lengths = {}
    pop_sizes = {}
    num_families = family_indices.shape[0]
    for node in tree.walk():
        if node == tree:
            depths[node.name] = 0
            continue
        theta = mrs[node.varname]
        pop_size = theta / (4*mu)
        pop_sizes[node.varname] = pop_size
        if node.multipliername is not None:
            mults = multipliers[node.multipliername].values
        else:
            mults = np.ones(num_families)
        bl = bls[node.varname] * mults
        #numgens = np.round(bl * pop_size).astype(np.int)
        numgens = bl * (2.0*pop_size)
        depths[node.name] = numgens + depths[node.ancestor.name]
        lengths[node.name] = numgens

    # make the depth of the *leaves* 0
    max_depths = np.zeros(num_families)
    for node in tree.walk():
        if node == tree:
            continue
        max_depths = np.maximum(max_depths, depths[node.name])
    for node in tree.walk():
        depths[node.name] = max_depths - depths[node.name]

    # simulation configuration
    n = 1000   # number of chromosomes sampled from each pop
    pop_idxs = {
            'child_blood':  0,
            'child_cheek':  1,
            'mother_blood': 2,
            'mother_cheek': 3
            }

    results = {
            'mother_blood' : [],
            'mother_cheek' : [],
            'child_blood'  : [],
            'child_cheek'  : [],
            'mother_birth_age' : [],
            'child_age': [],
            'mother_age': [],
            'family': []
            }

    mbs = []
    mcs = []
    cbs = []
    ccs = []
    mas = []
    cas = []
    mbas = []
    fams = []

    if free_recomb:
        num_reps = genome_size
        genome_size = 1
    else:
        num_reps = 1


    for f in xrange(num_families):
        fam_depths = {}
        for node in tree.walk('postorder'):
            fam_depths[node.name] = depths[node.name][f]

        # sample sizes and timing
        samps = []
        for pop in pop_idxs.keys():
            samps.extend([[pop_idxs[pop], fam_depths[pop]]] * n)

        child_blood_config = msprime.PopulationConfiguration(
                sample_size = None,
                initial_size = pop_sizes['blo'],
                growth_rate = 0)
        child_cheek_config = msprime.PopulationConfiguration(
                sample_size = None,
                initial_size = pop_sizes['buc'],
                growth_rate = 0)
        mother_blood_config = msprime.PopulationConfiguration(
                sample_size = None,
                initial_size = pop_sizes['blo'],
                growth_rate = 0)
        mother_cheek_config = msprime.PopulationConfiguration(
                sample_size = None,
                initial_size = pop_sizes['buc'],
                growth_rate = 0)
        pop_configs = [
                child_blood_config, child_cheek_config, mother_blood_config,
                mother_cheek_config]

        mig_mat = [[0]*4]*4

        # demographic events

        # first, mass migrations (population mergers)
        child_gast = msprime.MassMigration(
                time = fam_depths['som1'],
                source = pop_idxs['child_cheek'],  # 1
                destination = pop_idxs['child_blood'],  # 0
                proportion = 1.0)
        mother_gast = msprime.MassMigration(
                time = fam_depths['somM'],
                source = pop_idxs['mother_cheek'],  # 1
                destination = pop_idxs['mother_blood'],  # 0
                proportion = 1.0)
        emb = msprime.MassMigration(
                time = fam_depths['emb'],
                source = 2,
                destination = 0,
                proportion = 1.0)

        # then, population size changes
        fbloM = msprime.PopulationParametersChange(
                time = fam_depths['mother_fixed_blood'],
                initial_size = pop_sizes['fblo'],
                growth_rate = 0.0,
                population_id = 2)
        fbucM = msprime.PopulationParametersChange(
                time = fam_depths['mother_fixed_cheek'],
                initial_size = pop_sizes['fbuc'],
                growth_rate = 0.0,
                population_id = 3)
        fblo1 = msprime.PopulationParametersChange(
                time = fam_depths['child_fixed_blood'],
                initial_size = pop_sizes['fblo'],
                growth_rate = 0.0,
                population_id = 0)
        fbuc1 = msprime.PopulationParametersChange(
                time = fam_depths['child_fixed_cheek'],
                initial_size = pop_sizes['fbuc'],
                growth_rate = 0.0,
                population_id = 1)
        som = msprime.PopulationParametersChange(
                time = fam_depths['som1'],
                initial_size = pop_sizes['som'],
                growth_rate = 0.0,
                population_id = 0)
        loo = msprime.PopulationParametersChange(
                time = fam_depths['loo'],
                initial_size = pop_sizes['loo'],
                growth_rate = 0.0,
                population_id = 0)
        eoo = msprime.PopulationParametersChange(
                time = fam_depths['eoo'],
                initial_size = pop_sizes['eoo'],
                growth_rate = 0.0,
                population_id = 0)
        root_size = msprime.PopulationParametersChange(
                time = fam_depths['emb'],
                initial_size = N0,
                growth_rate = 0.0,
                population_id = 0)

        dem_evts = [child_gast, mother_gast, emb, som, eoo, loo, root_size, fbloM, fbucM, fblo1, fbuc1]
        dem_evts = sorted(dem_evts, key = lambda x: x.time)

        '''
        child_blood_config_dbg = msprime.PopulationConfiguration(
                sample_size = 2,
                initial_size = pop_sizes['blo'],
                growth_rate = 0)
        child_cheek_config_dbg = msprime.PopulationConfiguration(
                sample_size = 2,
                initial_size = pop_sizes['buc'],
                growth_rate = 0)
        mother_blood_config_dbg = msprime.PopulationConfiguration(
                sample_size = 2,
                initial_size = pop_sizes['blo'],
                growth_rate = 0)
        mother_cheek_config_dbg = msprime.PopulationConfiguration(
                sample_size = 2,
                initial_size = pop_sizes['buc'],
                growth_rate = 0)
        pop_configs_dbg = [
                child_blood_config_dbg, child_cheek_config_dbg, mother_blood_config_dbg,
                mother_cheek_config_dbg]
        dp = msprime.DemographyDebugger(
                Ne = 1.0,
                population_configurations = pop_configs_dbg,
                migration_matrix = mig_mat,
                demographic_events = dem_evts)
        dp.print_history()
        asdf
        '''

        mba = ages['mother_birth_age'].iloc[f]
        ca = ages['child_age'].iloc[f]
        ma = ages['mother_age'].iloc[f]
        fam = f

        samps_obtained = False
        for rep in xrange(num_reps):
            sims = msprime.simulate(
                    Ne = 1.0,
                    length = genome_size,
                    recombination_rate = 0.0,
                    mutation_rate = mu,
                    population_configurations = pop_configs,
                    migration_matrix = mig_mat,
                    demographic_events = dem_evts,
                    samples = samps)

            muts = sims.variants()

            if not samps_obtained:
                mbsamps = sims.get_samples(pop_idxs['mother_blood'])
                mcsamps = sims.get_samples(pop_idxs['mother_cheek'])
                cbsamps = sims.get_samples(pop_idxs['child_blood'])
                ccsamps = sims.get_samples(pop_idxs['child_cheek'])
                samps_obtained = True

            for mut in muts:
                freq_mom_blood = mut.genotypes[mbsamps].sum() / n
                freq_mom_cheek = mut.genotypes[mcsamps].sum() / n
                freq_child_blood = mut.genotypes[cbsamps].sum() / n
                freq_child_cheek = mut.genotypes[ccsamps].sum() / n
                
                mbs.append(freq_mom_blood)
                mcs.append(freq_mom_cheek)
                cbs.append(freq_child_blood)
                ccs.append(freq_child_cheek)
                mas.append(ma)
                cas.append(ca)
                mbas.append(mba)
                fams.append(fam)

    results = {}

    if mean_coverage is None:
        results['mother_blood'] = mbs
        results['mother_cheek'] = mcs
        results['child_blood'] = cbs
        results['child_cheek'] = ccs
        results['mother_age'] = mas
        results['child_age'] = cas
        results['mother_birth_age'] = mbas
        results['family'] = fams

    else:  # mean_coverage is not None
        num_muts = len(mbs)
        mbs_n = npr.poisson(mean_coverage, num_muts)
        mcs_n = npr.poisson(mean_coverage, num_muts)
        cbs_n = npr.poisson(mean_coverage, num_muts)
        ccs_n = npr.poisson(mean_coverage, num_muts)

        mbs_c = npr.binomial(mbs_n, mbs, num_muts)
        mcs_c = npr.binomial(mcs_n, mcs, num_muts)
        cbs_c = npr.binomial(cbs_n, cbs, num_muts)
        ccs_c = npr.binomial(ccs_n, ccs, num_muts)

        results['mother_blood'] = mbs_c
        results['mother_cheek'] = mcs_c
        results['child_blood'] = cbs_c
        results['child_cheek'] = ccs_c
        results['mother_blood_n'] = mbs_n
        results['mother_cheek_n'] = mcs_n
        results['child_blood_n'] = cbs_n
        results['child_cheek_n'] = ccs_n
        results['mother_age'] = mas
        results['child_age'] = cas
        results['mother_birth_age'] = mbas
        results['family'] = fams

    return pd.DataFrame(results)


def run_sim_ms(args):
    with open(args.tree) as fin:
        tree_str = fin.read().strip()
    tree = newick.loads(tree_str, length_parser = length_parser_str,
            look_for_multiplier = True)[0]
    for node in tree.walk('postorder'):
        if node.is_bottleneck:
            raise ValueError('no bottlenecks allowed')
    ages_dat = None
    if args.ages:
        ages_dat = pd.read_csv(args.ages, sep = '\t')
    branch_lengths, mut_rates, ages = (
            get_parameters(args.parameters, tree, args.num_families, ages_dat))
    tree.set_multiplier_values(ages)

    leaves = []
    for node in tree.walk('postorder'):
        if node.is_leaf:
            leaves.append(node.name)

    N = args.N
    comment_strings = []
    for varname in branch_lengths.keys():
        comment_strings.append("# {} {} {}".format(varname,
            branch_lengths[varname], mut_rates[varname]))
    comment_strings.append('# data columns: ' + ','.join(leaves))
    comment_strings.append('# ' + ' '.join(sys.argv))
    print '\n'.join(comment_strings)

    results = run_simulations(args.num_families, tree,  branch_lengths,
            mut_rates, N, args.mean_coverage, family_indices = ages['family'],
            ages = ages, genome_size = args.genome_size,
            free_recomb = args.free_recombination)

    
    results.to_csv(sys.stdout, sep = '\t', index = False,
            float_format = '%.4f')
