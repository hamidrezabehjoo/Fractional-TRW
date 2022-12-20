"""Test class for matrix implementation of tree-reweighted belief propagator"""
import unittest
from mrftools import *
import numpy as np


class TestMatrixTreeBeliefPropagator(unittest.TestCase):
    """Test class for matrix implementation of tree-reweighted belief propagator"""
    def create_chain_model(self):
        """Create chain-structured MRF with different variable cardinalities."""
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
        """Create loop-structured MRF with different variable cardinalities."""
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        mn.set_edge_factor((3, 0), np.random.randn(k[3], k[0]))
        mn.create_matrices()
        return mn

    def test_comparison_to_slow_trbp(self):
        """Test that matrix TRBP infers the same marginals as loop-based TRBP"""
        mn = self.create_loop_model()

        probs = {(0, 1): 0.75, (1, 2): 0.75, (2, 3): 0.75, (0, 3): 0.75, (3, 4): 1.0}

        trbp_mat = MatrixTRBeliefPropagator(mn, probs)
        trbp_mat.infer(display='final')
        trbp_mat.load_beliefs()

        trbp = TreeReweightedBeliefPropagator(mn, probs)
        trbp.infer(display='final')
        trbp.compute_pairwise_beliefs()
        trbp.compute_beliefs()

        for i in mn.variables:
            print ("Slow TRBP unary marginal of %d: %s" % (i, repr(np.exp(trbp.var_beliefs[i]))))
            print ("Matrix TRBP unary marginal of %d: %s" % (i, repr(np.exp(trbp_mat.var_beliefs[i]))))
            assert np.allclose(np.exp(trbp.var_beliefs[i]), np.exp(trbp_mat.var_beliefs[i])), "unary beliefs don't match"

        print ("Slow TRBP pairwise marginal: " + repr(np.exp(trbp.pair_beliefs[(0, 1)])))
        print ("Matrix TRBP pairwise marginal: " + repr(np.exp(trbp_mat.pair_beliefs[(0, 1)])))

        assert np.allclose(trbp.pair_beliefs[(0, 1)], trbp_mat.pair_beliefs[(0, 1)]), "Pair beliefs don't match: " + \
            "\nTRBP:" + repr(np.exp(trbp.pair_beliefs[(0, 1)])) + "\nMatTRBP:" + repr(np.exp(trbp_mat.pair_beliefs[(0, 1)]))

        print ("TRBP matrix energy functional: %f" % trbp_mat.compute_energy_functional())
        print ("TRBP slow energy functional: %f" % trbp.compute_energy_functional())

        assert np.allclose(trbp_mat.compute_energy_functional(), trbp.compute_energy_functional()), \
            "Energy functional is not exact. Slow TRBP: %f, Matrix TRBP: %f" % (trbp.compute_energy_functional(),
                                                                                trbp_mat.compute_energy_functional())

    def test_tree_structured_model(self):
        """Test that TRBP infers the true marginals on tree-structured MRF."""
        mn = MarkovNet()

        # np.random.seed(1)

        k = [4, 3, 6, 2]
        # k = [4, 4, 4, 4]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))

        print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
        print("Neighbors of 1: " + repr(mn.get_neighbors(1)))

        edge_probabilities = dict()

        for edge in mn.edge_potentials:
            edge_probabilities[edge] = 1 # BP

        trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)

        trbp.infer(display='full')
        trbp.load_beliefs()

        trbp.compute_pairwise_beliefs()

        bf = BruteForce(mn)

        for i in range(2):
            print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unary_marginal(i))))
            print ("TRBP unary marginal of %d:        %s" % (i, repr(np.exp(trbp.var_beliefs[i]))))
            assert np.allclose(bf.unary_marginal(i), np.exp(trbp.var_beliefs[i])), "TRBP not close to true unary"

        print ("Brute force pairwise marginal: " + repr(bf.pairwise_marginal(0, 1)))
        print ("TRBP pairwise marginal:        " + repr(np.exp(trbp.pair_beliefs[(0, 1)])))

        assert np.allclose(bf.pairwise_marginal(0, 1), np.exp(trbp.pair_beliefs[(0, 1)])), \
            "TRBP not close to pair marginal"

        print ("Tree Bethe energy functional:       %f" % trbp.compute_energy_functional())
        print ("Brute force log partition function: %f" % np.log(bf.compute_z()))

        assert np.allclose(trbp.compute_energy_functional(), np.log(bf.compute_z()))

    def test_upper_bound(self):
        """Test that TRBP provides an upper bound on the true log-partition function."""
        trials = 5

        tr_diff = np.zeros(trials)
        bp_diff = np.zeros(trials)

        for trial in range(trials):

            mn = MarkovNet()

            width = 3
            height = 3

            k = 2

            for x in range(width):
                for y in range(height):
                    mn.set_unary_factor((x, y), np.random.random(k))

            for x in range(width - 1):
                for y in range(height - 1):
                    mn.set_edge_factor(((x, y), (x + 1, y)), np.random.random((k, k)))
                    mn.set_edge_factor(((y, x), (y, x + 1)), np.random.random((k, k)))

            bf = BruteForce(mn)

            edge_probabilities = dict()

            for edge in mn.edge_potentials:
                edge_probabilities[edge] = 0.5

            interior_prob = 0.5
            border_prob = 0.75

            for x in range(width):
                edge_probabilities[(x, 0)] = interior_prob
                edge_probabilities[(x, height - 1)] = interior_prob

            for y in range(height):
                edge_probabilities[(0, y)] = border_prob
                edge_probabilities[(width - 1, y)] = border_prob

            trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
            trbp.infer(display='off')

            trbp_z = trbp.compute_energy_functional()
            true_z = np.log(bf.compute_z())

            print("Tree Bethe energy functional:       %f" % trbp_z)
            print("Brute force log partition function: %f" % true_z)

            print("Is the TRBP energy functional an upper bound? %r" %
                  (trbp_z >= true_z))
            assert trbp_z >= true_z, "TRBP energy functional was lower than true log partition"

            tr_diff[trial] = trbp_z - true_z

            print("Difference range between variational Z and truth:")
            print("TRBP:  %f to %f" % (min(tr_diff[:trial + 1]), max(tr_diff[:trial + 1])))
            print("Average error. TRBP: %f" % np.mean(np.abs(tr_diff[:trial + 1])))

if __name__ == '__main__':
    unittest.main()
