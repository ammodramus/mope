import transition_data_mut as tdm
import numpy as np

trans = tdm.TransitionData('transition_matrices_mutation_gens3_symmetric.h5',
        memory = True)

class TestTransitionData:

    def test_symmetry(self):
        gen = trans._sorted_gens[0]
        mut = trans._sorted_us[0]
        dist = trans.get_distribution(gen, mut)
        if not dist[0,0] == dist[-1,-1]:
            raise ValueError('distribution is not symmetric')


    def test_distributions(self):
        gen0 = trans._sorted_gens[0]
        gen1 = trans._sorted_gens[1]
        u0 = trans._sorted_us[0]
        u1 = trans._sorted_us[1]
        dist = trans.get_distribution(gen0, u0)
        dist1through5 = np.array([0.36769542, 0.36806349,
            0.18403174, 0.06128251, 0.01528996])
        assert np.all(np.isclose(dist[1,:5], dist1through5)), "nope"
        dist = trans.get_distribution(gen1, u1)
        dist1through5 = np.array([0.85333514, 0.01798609,
            0.01793035, 0.01606402, 0.01402782])
        assert np.all(np.isclose(dist[1,:5], dist1through5)), "nope"


    def test_stochastic_matrix(self):
        gen1 = trans._sorted_gens[1]
        u1 = trans._sorted_us[1]
        dist = trans.get_distribution(gen1, u1)
        assert np.all(np.isclose(np.sum(dist, 1), 1.0))

    def test_none_bounds(self):
        P = trans.get_transition_probabilities_time_mutation(-1e-4, 0.00025)
        assert P is None, "P is not None for negative time"

        # test the upper and lower boundaries, should be able to get
        # distributions at boundaries
        time = trans._min_coal_time
        mut = trans._min_mutation_rate
        P = trans.get_transition_probabilities_time_mutation(time, mut)
        assert np.all(np.isclose(np.sum(P, 1), 1.0))

        time = trans._min_coal_time
        mut = trans._max_mutation_rate
        P = trans.get_transition_probabilities_time_mutation(time, mut)
        assert np.all(np.isclose(np.sum(P, 1), 1.0))

        time = trans._max_coal_time
        mut = trans._min_mutation_rate
        P = trans.get_transition_probabilities_time_mutation(time, mut)
        assert np.all(np.isclose(np.sum(P, 1), 1.0))

        time = trans._max_coal_time
        mut = trans._max_mutation_rate
        P = trans.get_transition_probabilities_time_mutation(time, mut)
        assert np.all(np.isclose(np.sum(P, 1), 1.0))

    def test_interpolation_stochastic_matrix(self):
        time = (trans._min_coal_time + trans._max_coal_time) / 2.0
        mut = (trans._min_mutation_rate + trans._max_mutation_rate) / 2.0
        P1 = trans.get_transition_probabilities_time_mutation(time, mut)
        assert np.all(np.isclose(np.sum(P1, 1), 1.0))
        assert np.all(P1 >= 0) and np.all(P1 <= 1)

    def test_linear_interpolation(self):
        # tests for stochastic matrix anyway
        time = 0.8*trans._max_coal_time
        mut = 0.8*trans._max_mutation_rate
        desired_gen_time = trans._N * time
        gen_idx = np.searchsorted(trans._sorted_gens, desired_gen_time)
        desired_u = mut / (2.0 * trans._N)
        u_idx = np.searchsorted(trans._sorted_us, desired_u)
        P1 = trans.bilinear_interpolation(desired_gen_time, desired_u,
                gen_idx, u_idx)
        assert np.all(np.isclose(np.sum(P1, 1), 1.0))
        assert np.all(P1 >= 0) and np.all(P1 <= 1)

    '''
    def test_quadratic_interpolation(self):
        # tests for stochastic matrix anyway
        time = 1.001*trans._min_coal_time
        mut = (trans._min_mutation_rate + trans._max_mutation_rate) / 2.0
        desired_gen_time = trans._N * time
        gen_idx = np.searchsorted(trans._sorted_gens, desired_gen_time)
        desired_u = mut / (2.0 * trans._N)
        u_idx = np.searchsorted(trans._sorted_us, desired_u)
        P1 = trans.biquadratic_interpolation(desired_gen_time, desired_u,
                gen_idx, u_idx)
        assert np.all(np.isclose(np.sum(P1, 1), 1.0))
        assert np.all(P1 >= 0) and np.all(P1 <= 1)
    '''
