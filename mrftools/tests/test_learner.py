"""Test class for Learner and its subclasses"""
import unittest
import numpy as np
from scipy.optimize import check_grad, approx_fprime
import matplotlib.pyplot as plt
from mrftools import *


class TestLearner(unittest.TestCase):
    """Test class for Learner and its subclasses"""
    def set_up_learner(self, learner, latent=True):
        """
        Provide synthetic training data for a learner.
        :param learner: Learner object
        :type learner: Learner
        :param latent: Boolean value indicating whether to have latent variables in training data
        :type latent: bool
        :return: None
        """
        d = 2
        num_states = 4

        np.random.seed(0)

        if latent:
            labels = [{0: 2,       2: 1},
                      {      1: 2, 2: 0},
                      {0: 2, 1: 3,     },
                      {0: 0, 1: 2, 2: 3}]
        else:
            labels = [{0: 2, 1: 3, 2: 1},
                      {0: 3, 1: 2, 2: 0},
                      {0: 2, 1: 3, 2: 1},
                      {0: 0, 1: 2, 2: 3}]

        models = []
        for i in range(len(labels)):
            m = self.create_random_model(num_states, d)
            models.append(m)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

    def test_gradient(self):
        """
        Test that the provided gradient is consistent with a numerically estimated gradient when some variables are 
        latent.
        """
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)
        learner.start_time = time.time()
        learner.set_regularization(0.0, 1.0)
        gradient_error = check_grad(learner.subgrad_obj, learner.subgrad_grad, weights)

        # numerical_grad = approx_fprime(weights, learner.subgrad_obj, 1e-4)
        # analytical_grad = learner.subgrad_grad(weights)
        # plt.plot(numerical_grad, 'r')
        # plt.plot(analytical_grad, 'b')
        # plt.show()

        print("Gradient error: %e" % gradient_error)
        assert gradient_error < 1e-1, "Gradient is wrong"

    def test_fully_observed_gradient(self):
        """Test that the gradient is consistent with a numerically estimated gradient when all variables are observed"""
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=False)
        learner.start_time = time.time()
        learner.set_regularization(0.0, 1.0)
        gradient_error = check_grad(learner.subgrad_obj, learner.subgrad_grad, weights)

        # numerical_grad = approx_fprime(weights, learner.subgrad_obj, 1e-4)
        # analytical_grad = learner.subgrad_grad(weights)
        # plt.plot(numerical_grad, 'r')
        # plt.plot(analytical_grad, 'b')
        # plt.show()

        print("Gradient error: %f" % gradient_error)
        assert gradient_error < 1e-1, "Gradient is wrong"

    def test_m_step_gradient(self):
        """Test that the gradient for the EM m-step is consistent with numerically estimated gradient."""
        weights = np.zeros(8 + 32)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)
        learner.set_regularization(0.0, 1.0)
        learner.start = time.time()
        learner.e_step(weights)
        gradient_error = check_grad(learner.objective, learner.gradient, weights)

        # numerical_grad = approx_fprime(weights, learner.objective, 1e-4)
        # analytical_grad = learner.gradient(weights)
        # plt.plot(numerical_grad, 'r')
        # plt.plot(analytical_grad, 'b')
        # plt.show()

        print("Gradient error: %f" % gradient_error)
        assert gradient_error < 1e-1, "Gradient is wrong"

    def test_learner(self):
        """Test that the learner decreases the objective value and that it stays non-negative."""
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = weight_record.shape[0]
        old_obj = np.Inf
        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i,:])
            assert (new_obj <= old_obj + 1e-8), "subgradient objective is not decreasing"
            old_obj = new_obj

            assert new_obj >= 0, "Learner objective was not non-negative"

    def test_EM(self):
        """Test that the EM learner decreases the objective value and that it stays non-negative."""
        weights = np.zeros(8 + 32)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = weight_record.shape[0]

        old_obj = learner.subgrad_obj(weight_record[0,:])
        new_obj = learner.subgrad_obj(weight_record[-1,:])
        assert (new_obj <= old_obj), "EM objective did not decrease"

        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert new_obj >= 0, "EM objective was not non-negative"

    def test_paired_dual(self):
        """Test that the paired-dual learner decreases the objective value and that it stays non-negative."""
        weights = np.zeros(8 + 32)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = weight_record.shape[0]

        old_obj = learner.subgrad_obj(weight_record[0, :])
        new_obj = learner.subgrad_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "paired dual objective did not decrease"

        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert new_obj >= 0, "Paired dual objective was not non-negative"

    def test_primal_dual(self):
        """Test that the primal-dual learner decreases the objective value and that it stays non-negative."""
        weights = np.zeros(8 + 32)
        learner = PrimalDual(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = weight_record.shape[0]

        old_obj = learner.subgrad_obj(weight_record[0, :])
        new_obj = learner.subgrad_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "Primal Dual objective did not decrease"

        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert new_obj >= 0, "Primal Dual objective was not non-negative"

    def test_overflow(self):
        """Initialize weights to a huge number and see if learner can escape it"""
        weights = 1000 * np.random.randn(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        assert not np.isnan(learner.subgrad_obj(weights)), \
            "Objective for learner was not a number"

    def create_random_model(self, num_states, d):
        """
        Create a random LogLinearModel with random features fo all unary and edge potentials
        :param num_states: cardinality of each variable
        :type num_states: int
        :param d: dimensionality of feature vectors
        :type d: int
        :return: random model
        :rtype: LogLinearModel
        """
        model = LogLinearModel()

        model.declare_variable(0, num_states)
        model.declare_variable(1, num_states)
        model.declare_variable(2, num_states)

        model.set_unary_weights(0, np.random.randn(num_states, d))
        model.set_unary_weights(1, np.random.randn(num_states, d))
        model.set_unary_weights(2, np.random.randn(num_states, d))

        model.set_unary_features(0, np.random.randn(d))
        model.set_unary_features(1, np.random.randn(d))
        model.set_unary_features(2, np.random.randn(d))

        model.set_all_unary_factors()

        model.set_edge_factor((0, 1), np.zeros((num_states, num_states)))
        model.set_edge_factor((1, 2), np.zeros((num_states, num_states)))

        model.set_edge_features((0, 1), np.random.randn(d))
        model.set_edge_features((1, 2), np.random.randn(d))

        edge_probabilities = dict()

        for edge in model.edge_potentials:
            edge_probabilities[edge] = 0.75

        model.tree_probabilities = edge_probabilities

        return model

    def test_early_stopping(self):
        """Test that early-stopping timer correctly stops learning"""
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=True)
        start = time.time()
        learner.learn(weights)
        default_time = time.time() - start

        print("Default learner took %f seconds." % default_time)

        start = time.time()
        learner.max_time = 0.0001
        learner.learn(weights)
        early_time = time.time() - start

        print("Without early stopping: %f seconds. With early stopping %f seconds." % (default_time, early_time))
        assert early_time < default_time, "Early stopping was no faster than default"
