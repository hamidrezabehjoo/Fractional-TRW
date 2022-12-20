"""Test class for approximate maximum likelihood learner"""
import unittest
from mrftools import *


class TestApproxMaxLikelihood(unittest.TestCase):
    """
    Unit test class for approximate maximum likelihood learner
    """
    def test_sampled_data(self):
        """
        Create random model, sample data using Gibbs sampling, then try to learn back the model from the 
        sampled data
        """
        np.random.seed(0)

        model = create_model(0)
        learner = ApproxMaxLikelihood(model)
        learner.set_regularization(0, 0.01)
        set_up_learner(learner, model)

        print("Loaded sampled data. Starting learning...")

        plotter = ObjectivePlotter(learner.subgrad_obj)
        plotter.interval = 0.01

        weights = np.zeros(learner.weight_dim)

        new_weights = learner.learn(weights)#, plotter.callback)
        new_model = learner.belief_propagators[0].mn
        new_model.load_factors_from_matrices()

        bp = BeliefPropagator(model)
        bp.infer()
        learned_bp = BeliefPropagator(new_model)
        learned_bp.infer()

        for var in new_model.variables:
            learned_marg = np.exp(learned_bp.var_beliefs[var])
            true_marg = np.exp(bp.var_beliefs[var])
            print("Learned vs true marginals for %d:" % var)
            print(learned_marg)
            print(true_marg)
            assert np.argmax(learned_marg) == np.argmax(true_marg), "learned marginal decoding disagrees with truth"


def set_up_learner(learner, model):
    """
    Add sampled data to learner.
    
    :param learner: learner object that accepts training data
    :type learner: Learner
    :param model: Markov net model
    :type model: MarkovNet
    :return: None
    """
    data = sample_data(model)

    for example in data:
        learner.add_data(example)


def create_model(seed):
    """
    Create a chain structured MarkovNet with three variables
    :param seed: random seed
    :type seed: number
    :return: Generated Markov net with random potentials
    :rtype: MarkovNet
    """
    np.random.seed(seed)
    model = MarkovNet()

    num_states = [3, 2, 2]

    for i in range(len(num_states)):
        model.set_unary_factor(i, np.random.randn(num_states[i]))

    edges = [(0, 1), (0, 2)]

    for edge in edges:
        model.set_edge_factor(edge, np.random.randn(num_states[edge[0]], num_states[edge[1]]))

    return model


def sample_data(model):
    """
    Sample data from model
    
    :param model: Markov net to sample from
    :type model: MarkovNet 
    :return: list of samples
    :rtype: list
    """
    sampler = GibbsSampler(model)
    sampler.init_states(0)

    mix = 5000
    num_samples = 200

    sampler.gibbs_sampling(mix, num_samples)

    return sampler.samples

if __name__ == '__main__':
    unittest.main()
