"""Test class for max-product linear programming"""
import numpy as np
from mrftools import *
import unittest
import time


class TestMaxProductLinearProgramming(unittest.TestCase):
    """Test class for max-product linear programming"""
    def create_chain_model(self):
        """Create chain-structured MRF with different cardinalities."""
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
        mn.create_matrices()

        return mn

    def create_loop_model(self):
        """Create loop-structured MRF with different cardinalities."""
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        mn.set_edge_factor((3, 0), np.random.randn(k[3], k[0]))
        mn.create_matrices()
        return mn

    def test_consistency(self):
        """Test that MPLP leads to locally consistent estimates"""
        mn = self.create_loop_model()

        bp = MaxProductLinearProgramming(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                print(pair_belief)
                print(unary_belief)
                assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"

    def test_exactness(self):
        """Test that MPLP infers the true MAP state in a chain."""
        mn = self.create_chain_model()
        bp = MaxProductLinearProgramming(mn)
        bp.infer(display='full')
        bp.load_beliefs()

        bf = BruteForce(mn)
        print(bp.belief_mat)
        print(bf.map_inference())
        assert (np.array_equal(bp.belief_mat,bf.map_inference())), "beliefs are not exact"

    def test_overflow(self):
        """Test that MPLP does not fail when given a large, poorly scaled potential."""
        mn = self.create_loop_model()

        # set a really large factor
        mn.set_unary_factor(0, [-1000, 2000, 3000, 4000])

        mn.create_matrices()

        bp = MaxProductLinearProgramming(mn)

        with np.errstate(all='raise'):
            bp.infer()
            bp.load_beliefs()