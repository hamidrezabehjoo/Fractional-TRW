"""Tests for the log-linear model objects"""
import unittest
import numpy as np
from mrftools import *


class TestLogLinearModel(unittest.TestCase):
    """Test class for LogLinearModel"""
    def create_chain_model(self, k):
        """
        Create a chain-structured Markov net with random potentials and different variable cardinalities.
        
        :param k: list of length 5 of cardinalities
        :type k: arraylike
        :return: chain markov net
        :rtype: MarkovNet
        """
        mn = LogLinearModel()

        np.random.seed(1)

        for i in range(len(k)):
            mn.set_unary_factor(i, np.random.randn(k[i]))

        factor4 = np.random.randn(k[4])
        factor4[2] = -float('inf')

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
        mn.set_edge_factor((1, 4), np.random.randn(k[1], k[4]))

        d = 4

        for i in range(len(k)):
            mn.set_unary_features(i, np.random.randn(d))

        return mn

    def test_structure(self):
        """
        Test that the graph structure of the log-linear model is correct, 
        and that it handles different orders of nodes when referring to the same edge
        """
        mn = LogLinearModel()

        mn.set_unary_factor(0, np.random.randn(4))
        mn.set_unary_factor(1, np.random.randn(3))
        mn.set_unary_factor(2, np.random.randn(5))

        mn.set_edge_factor((0, 1), np.random.randn(4, 3))
        mn.set_edge_factor((1, 2), np.random.randn(3, 5))

        print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
        print("Neighbors of 1: " + repr(mn.get_neighbors(1)))
        print("Neighbors of 2: " + repr(mn.get_neighbors(2)))

        assert mn.get_neighbors(0) == set([1]), "Neighbors are wrong"
        assert mn.get_neighbors(1) == set([0, 2]), "Neighbors are wrong"
        assert mn.get_neighbors(2) == set([1]), "Neighbors are wrong"

    def test_matrix_shapes(self):
        """Test that the matrix is correctly shaped based on the maximum cardinality variable"""
        k = [2, 3, 4, 5, 6]
        mn = self.create_chain_model(k)

        max_states = max(k)

        assert mn.matrix_mode is False, "Matrix mode flag was set prematurely"

        mn.create_matrices()

        assert mn.matrix_mode, "Matrix mode flag wasn't set correctly"

        assert mn.unary_mat.shape == (max_states, 5)

    def test_feature_computation(self):
        """Test that creating unary features runs without crashing (i.e., that the matrix shapes are correct)"""
        k = [2, 3, 4, 5, 6]
        mn = self.create_chain_model(k)
        d = 4

        for i in range(len(k)):
            mn.set_unary_weights(i, np.random.randn(k[i], d))

    def test_edge_features(self):
        """
        Test that edge features can be set and set properly in matrix mode, including testing that
        conditioning a center node in a chain leads to independence.
        """
        k = [4, 4, 4, 4, 4]
        mn = self.create_chain_model(k)

        d = 3

        for i in range(5):
            mn.set_edge_features((i, i+1), np.random.randn(d))

        mn.create_matrices()
        mn.set_unary_weight_matrix(np.random.randn(4, 4))
        mn.set_edge_weight_matrix(np.random.randn(d, 16))

        bp = MatrixBeliefPropagator(mn)

        bp.infer()
        bp.load_beliefs()

        unconditional_marginals = bp.var_beliefs[4]

        bp.condition(0, 2)
        bp.infer()
        bp.load_beliefs()

        conditional_marginals = bp.var_beliefs[4]

        assert not np.allclose(unconditional_marginals, conditional_marginals), \
                "Conditioning on variable 0 did not change marginal of variable 4"

        mn.set_edge_features((2, 3), np.zeros(d))
        mn.create_matrices()
        mn.set_unary_weight_matrix(np.random.randn(4, 4))
        mn.set_edge_weight_matrix(np.random.randn(d, 16))

        bp.infer()
        bp.load_beliefs()

        unconditional_marginals = bp.var_beliefs[4]

        bp.condition(0, 2)
        bp.infer()
        bp.load_beliefs()

        conditional_marginals = bp.var_beliefs[4]

        assert np.allclose(unconditional_marginals, conditional_marginals), \
            "Conditioning on var 0 changed marginal of var 4, when the features should have made them independent"

    def test_indicator_form(self):
        """Test the overcomplete indicator-feature-function log-linear model and compare against the MarkovNet form"""
        mn = MarkovNet()

        num_states = [3, 2, 4, 5]

        for i in range(len(num_states)):
            mn.set_unary_factor(i, np.random.randn(num_states[i]))

        edges = [(0, 1), (0, 2), (2, 3), (1, 3)]

        for edge in edges:
            mn.set_edge_factor(edge, np.random.randn(num_states[edge[0]], num_states[edge[1]]))

        model = LogLinearModel()
        model.create_indicator_model(mn)
        model.update_edge_tensor()
        model.update_unary_matrix()

        bp = BeliefPropagator(mn)
        bp.infer(display='final')
        bp.compute_beliefs()
        bp.compute_pairwise_beliefs()

        bp_ind = MatrixBeliefPropagator(model)
        bp_ind.infer(display='final')
        bp_ind.load_beliefs()

        for i in range(len(num_states)):
            assert np.allclose(bp_ind.var_beliefs[i], bp.var_beliefs[i]), "unary beliefs disagree"

        for edge in edges:
            assert np.allclose(np.sum(np.exp(bp_ind.pair_beliefs[edge])), 1.0), "Pair beliefs don't normalize to 1"
            assert np.allclose(bp_ind.pair_beliefs[edge], bp.pair_beliefs[edge]), "edge beliefs disagree: \n" +\
                "indicator:\n" + repr(bp_ind.pair_beliefs[edge]) + "\noriginal:\n" + repr(bp.pair_beliefs[edge])

    def test_matrix_structure(self):
        """Test that the sparse matrix structure in matrix mode is correct."""
        k = [2, 3, 4, 5, 6]
        model = self.create_chain_model(k)

        model.create_matrices()

        for edge, i in model.message_index.items():
            from_index = model.var_index[edge[0]]
            to_index = model.var_index[edge[1]]
            assert model.message_from[i] == from_index, "Message sender index is wrong"
            assert model.message_to[i] == to_index, "Message receiver index is wrong"
            model.message_to_map.getrow(i).getcol(to_index), "Message receiver matrix map is wrong"

        assert np.all(np.sum(model.message_to_map.todense(), axis=1) == 1), \
            "Message sender map has a row that doesn't sum to 1.0"
