import numpy as np

def get_branch_lengths(scenario):
    branch_lengths = np.array((
        0.01,  # 0, brain -> forebrain
        0.5,  # 1, embryo -> brain
        0.2,   # 2, mesoderm -> urogenital split
        0.15,  # 3, myocardial divergence -> smooth muscle divergence
        0.1,   # 4, blood divergence -> myocardial muscle divergence
        0.05,  # 5, mesoderm -> blood diverges from lateral plate mesoderm
        0.01,  # 6, embryo -> mesoderm
        0.01   # 7, embryo -> endoderm
        ))
    return branch_lengths

def get_mutation_rates(scenario):
    '''
    Assume per-replication error rate is around 1e-8
    '''
    mutation_rates = np.array((
        5e-3,  # 0, brain -> forebrain
        1e-4,  # 1, embryo -> brain
        5e-3,  # 2, mesoderm -> urogenital split
        5e-3,  # 3, myocardial divergence -> smooth muscle divergence
        1e-5,  # 4, blood divergence -> myocardial muscle divergence
        1e-6,  # 5, mesoderm -> blood diverges from lateral plate mesoderm
        1e-7,  # 6, embryo -> mesoderm
        1e-7   # 7, embryo -> endoderm
        ))
    return mutation_rates

def get_branch_rates(scenario):
    branch_rates = np.array((
        0.005,   # 0, cortex, 0.35 in 70 years
        0.005,   # 1, cerebrum
        0.005,   # 2, cerebellum
        0.005,  # 3, ovary
        0.005,   # 4, kidney
        0.005,   # 5, skeletal muscle
        0.005,   # 6, blood
        0.005,   # 7, myocardial muscle
        0.005,   # 8, small intestine
        0.005,   # 9, large intestine
        0.005    # 10, liver
        ))
    return branch_rates

def get_rate_mutation_rates(scenario):
    rate_mutation_rates = np.array((
        2e-4,   # 0, cortex
        2e-4,   # 1, cerebrum
        2e-4,   # 2, cerebellum
        2e-4,   # 3, ovary
        2e-4,   # 4, kidney
        2e-4,   # 5, skeletal muscle
        2e-4,   # 6, blood
        2e-4,   # 7, myocardial muscle
        2e-4,   # 8, small intestine
        2e-4,   # 9, large intestine
        2e-4    # 10,  liver
        ))
    return rate_mutation_rates

def get_params(scenario):
    branch_lengths = get_branch_lengths(scenario)
    mutation_rates = get_mutation_rates(scenario)
    branch_rates = get_branch_rates(scenario)
    rate_mutation_rates = get_rate_mutation_rates(scenario)
    return branch_lengths, mutation_rates, branch_rates, rate_mutation_rates

def get_root_parameter(scenario):
    return 1e-2

def get_num_reps(scenario):
    return 20000
