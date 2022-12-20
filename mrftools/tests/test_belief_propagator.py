"""Test class to test belief propagator implementation"""
import unittest

import numpy as np
from mrftools import *


class TestBeliefPropagator(unittest.TestCase):
    """
    Unit test class for belief propagation implementation
    """
    def create_chain_model(self):
        """Create a chain-structured Markov net with random potentials and different variable cardinalities."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))

        factor4 = np.random.randn(k[4])
        factor4[2] = -float('inf')

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((2, 3), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((3, 4), np.random.randn(k[3], k[4]))

        return mn

    def create_loop_model(self):
        """
        Create a loop-structured Markov net with random potentials and variable cardinalities.
        This method is implemented by calling create_chain_model and then connecting the ends of the chain.
        """
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        mn.set_edge_factor((3, 0), np.random.randn(k[3], k[0]))
        return mn

    def test_exactness(self):
        """
        Test that the belief propagator on a chain model infers the exact marginals within numerical precision of
        marginals computed by brute force counting.
        :return: None
        """
        mn = self.create_chain_model()
        bp = BeliefPropagator(mn)

        bp.infer(display='full')

        bp.compute_pairwise_beliefs()

        bf = BruteForce(mn)

        for i in mn.variables:
            print("Brute force unary marginal of %d: %s" % (i, repr(bf.unary_marginal(i))))
            print("Belief prop unary marginal of %d: %s" % (i, repr(np.exp(bp.var_beliefs[i]))))
            assert np.allclose(bf.unary_marginal(i), np.exp(bp.var_beliefs[i])), "beliefs aren't exact on chain model"

        print("Brute force pairwise marginal: " + repr(bf.pairwise_marginal(0, 1)))
        print("Belief prop pairwise marginal: " + repr(np.exp(bp.pair_beliefs[(0, 1)])))

        print("Bethe energy functional: %f" % bp.compute_energy_functional())

        print("Brute force log partition function: %f" % np.log(bf.compute_z()))

        assert np.allclose(np.log(bf.compute_z()), bp.compute_energy_functional()),\
            "log partition function is not exact on chain model"

    def test_consistency(self):
        """
        Test that the marginals inferred by loopy belief propagation are locally consistent with neighboring variables.
        :return: None
        """
        mn = self.create_loop_model()

        bp = BeliefPropagator(mn)
        bp.infer(display='full')

        bp.compute_beliefs()
        bp.compute_pairwise_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                print(pair_belief, unary_belief)
                assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"

    def test_normalization(self):
        """
        Test that the unary and pairwise beliefs after inference are normalized property.
        :return: None
        """
        mn = self.create_loop_model()

        bp = BeliefPropagator(mn)
        bp.infer(display='full')

        bp.compute_beliefs()
        bp.compute_pairwise_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            assert np.allclose(np.sum(unary_belief), 1.0), "unary belief is not normalized"
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.exp(bp.pair_beliefs[(var, neighbor)])
                assert np.allclose(np.sum(pair_belief), 1.0), "pairwise belief is not normalize"

