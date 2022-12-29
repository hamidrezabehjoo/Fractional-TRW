"""Test class for Gibbs sampler"""
from __future__ import division
from mrftools import *
import numpy as np

import unittest

class TestGibbsSampling(unittest.TestCase):
    """Test class for Gibbs sampler"""
    def test_gibbs_sampling(self):
        """Test that estimating marginals from Gibbs sampling leads to the same marginals as brute force counting."""
        mn = MarkovNet()

        np.random.seed(0)

        unary_potential = np.random.randn(2)
        edge_potential = np.random.randn(2, 2)

        mn.set_unary_factor(0, unary_potential)
        mn.set_unary_factor(1, unary_potential)
        mn.set_unary_factor(2, unary_potential)
        mn.set_unary_factor(3, unary_potential)

        mn.set_edge_factor((0, 1), edge_potential)
        mn.set_edge_factor((1, 2), edge_potential)
        mn.set_edge_factor((2, 3), edge_potential)
        # mn.set_edge_factor((3, 0), edge_potential) # uncomment this to make loopy

        gb = GibbsSampler(mn)
        gb.init_states()
        itr = 1000
        num = 10000
        gb.gibbs_sampling(itr, num)

        bf = BruteForce(mn)
        for var in mn.variables:
            gb_result = gb.count_occurrences(var) / num
            bf_result = bf.unary_marginal(var)
            print(gb_result)
            print(bf_result)
            np.testing.assert_allclose(gb_result, bf_result, rtol=1e-1, atol=0)




