"""Unit tests for convexified belief propagation"""
import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt


class TestConvexBP(unittest.TestCase):
    """
    Unit test class for convexified belief propagation
    """
    def create_q_model(self):
        """Create loop model with one variable hanging off the loop (forming a Q shape)."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))
        mn.set_unary_factor(4, np.random.randn(k[4]))

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((2, 3), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((0, 3), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((0, 4), np.random.randn(k[0], k[4]))
        mn.create_matrices()

        return mn

    def test_comparison_to_trbp(self):
        """
        Test that convex BP and tree-reweighted BP produce the same results when the convex BP counting numbers are
        set to the TRBP counting numbers. 
        """
        mn = self.create_q_model()

        probs = {(0, 1): 0.75, (1, 2): 0.75, (2, 3): 0.75, (0, 3): 0.75, (0, 4): 1.0}

        trbp_mat = MatrixTRBeliefPropagator(mn, probs)
        trbp_mat.infer(display='full')
        trbp_mat.load_beliefs()

        counting_numbers = probs.copy()
        counting_numbers[0] = 1.0 - 2.5
        counting_numbers[1] = 1.0 - 1.5
        counting_numbers[2] = 1.0 - 1.5
        counting_numbers[3] = 1.0 - 1.5
        counting_numbers[4] = 1.0 - 1.0

        cbp = ConvexBeliefPropagator(mn, counting_numbers)
        cbp.infer(display='full')
        cbp.load_beliefs()

        for i in mn.variables:
            print("Convex unary marginal of %d: %s" % (i, repr(np.exp(cbp.var_beliefs[i]))))
            print("Matrix TRBP unary marginal of %d: %s" % (i, repr(np.exp(trbp_mat.var_beliefs[i]))))
            assert np.allclose(np.exp(cbp.var_beliefs[i]), np.exp(trbp_mat.var_beliefs[i])), "unary beliefs don't match"

        print("Convex pairwise marginal: " + repr(np.exp(cbp.pair_beliefs[(0, 1)])))
        print("Matrix TRBP pairwise marginal: " + repr(np.exp(trbp_mat.pair_beliefs[(0, 1)])))

        print("Pairwise marginal error %f" %
               np.sum(np.abs(np.exp(cbp.pair_beliefs[(0, 1)]) - np.exp(trbp_mat.pair_beliefs[(0, 1)]))))

        # plt.subplot(211)
        # plt.imshow(cbp.pair_beliefs[(0, 1)], interpolation='nearest')
        # plt.xlabel('CBP')
        # plt.subplot(212)
        # plt.imshow(trbp_mat.pair_beliefs[(0, 1)], interpolation='nearest')
        # plt.xlabel('TRBP')
        # plt.show()

        assert np.allclose(cbp.pair_beliefs[(0, 1)], trbp_mat.pair_beliefs[(0, 1)]), "Pair beliefs don't match: " + \
                                                                                     "\nCBP:" + repr(
            np.exp(cbp.pair_beliefs[(0, 1)])) + "\nMatTRBP:" + repr(np.exp(trbp_mat.pair_beliefs[(0, 1)]))

        print("TRBP matrix energy functional: %f" % trbp_mat.compute_energy_functional())
        print("Convex energy functional: %f" % cbp.compute_energy_functional())

        assert np.allclose(trbp_mat.compute_energy_functional(), cbp.compute_energy_functional()), \
            "Energy functional is not exact. Convex: %f, Matrix TRBP: %f" % (cbp.compute_energy_functional(),
                                                                             trbp_mat.compute_energy_functional())

    def test_comparison_to_bethe(self):
        """
        Test that loopy belief propagation and convexified belief propagation output the same inferred marginals
        when the counting numbers are set to the Bethe counting numbers (which make convex BP no longer convex).
        :return: None
        """
        mn = self.create_q_model()

        bp = MatrixBeliefPropagator(mn)
        bp.infer(display='final')
        bp.load_beliefs()

        counting_numbers = {(0, 1): 1.0,
                            (1, 2): 1.0,
                            (2, 3): 1.0,
                            (0, 3): 1.0,
                            (0, 4): 1.0,
                            0: 1.0 - 3.0,
                            1: 1.0 - 2.0,
                            2: 1.0 - 2.0,
                            3: 1.0 - 2.0,
                            4: 1.0 - 1.0}

        cbp = ConvexBeliefPropagator(mn, counting_numbers)
        cbp.infer(display='full')
        cbp.load_beliefs()

        for i in mn.variables:
            print("Convex unary marginal of %d: %s" % (i, repr(np.exp(cbp.var_beliefs[i]))))
            print("Matrix BP unary marginal of %d: %s" % (i, repr(np.exp(bp.var_beliefs[i]))))
            assert np.allclose(np.exp(cbp.var_beliefs[i]), np.exp(bp.var_beliefs[i])), "unary beliefs don't match"

        print("Convex pairwise marginal: " + repr(np.exp(cbp.pair_beliefs[(0, 1)])))
        print("Matrix BP pairwise marginal: " + repr(np.exp(bp.pair_beliefs[(0, 1)])))

        assert np.allclose(cbp.pair_beliefs[(0, 1)], bp.pair_beliefs[(0, 1)]), "Pair beliefs don't match: " + \
                                                                                     "\nCBP:" + repr(
            np.exp(cbp.pair_beliefs[(0, 1)])) + "\nMatBP:" + repr(np.exp(bp.pair_beliefs[(0, 1)]))

        print("Bethe matrix energy functional: %f" % bp.compute_energy_functional())
        print("Convex energy functional: %f" % cbp.compute_energy_functional())

        assert np.allclose(bp.compute_energy_functional(), cbp.compute_energy_functional()), \
            "Energy functional is not exact. Convex: %f, BP: %f" % (cbp.compute_energy_functional(),
                                                                             bp.compute_energy_functional())

    def test_convexity(self):
        """Test that the convex BP objective is within numerical precision of being truly convex."""
        mn = self.create_q_model()

        edge_count = 0.1
        node_count = 0.1

        counting_numbers = {(0, 1): edge_count,
                            (1, 2): edge_count,
                            (2, 3): edge_count,
                            (0, 3): edge_count,
                            (0, 4): edge_count,
                            0: node_count,
                            1: node_count,
                            2: node_count,
                            3: node_count,
                            4: node_count}

        bp = ConvexBeliefPropagator(mn, counting_numbers)
        bp.infer(display="full")

        messages = bp.message_mat.copy()

        noise = 0.1 * np.random.randn(messages.shape[0], messages.shape[1])

        x = np.linspace(-1, 1, 21)
        y = np.zeros(21)
        z = np.zeros(21)

        for i in range(len(x)):
            mod_messages = messages + x[i] * noise
            bp.set_messages(mod_messages)
            y[i] = bp.compute_dual_objective()
            z[i] = bp.compute_inconsistency()

        bp.load_beliefs()
        print(np.exp(bp.var_beliefs[0]))
        print(np.exp(bp.pair_beliefs[(0, 1)]))

        print("Minimum dual objective: %f" % np.min(y))
        print("Inconsistency at argmin: %f" % z[np.argmin(y)])

        # plt.plot(x, y)
        # plt.show()

        assert np.allclose(y.min(), y[10]), "Minimum was not at converged messages"

        deriv = y[1:] - y[:-1]
        second_deriv = deriv[1:] - deriv[:-1]
        print(second_deriv)
        assert np.all(second_deriv >= 0), "Estimated second derivative was not non-negative"

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
        mn.create_matrices()

        return mn

    def test_exactness(self):
        """Test that convex BP with very small counting numbers produces the exact MAP solution on a chain model."""
        mn = self.create_chain_model()

        sharp_counting_numbers = dict()
        for var in mn.variables:
            sharp_counting_numbers[var] = 0.01
            for neighbor in mn.neighbors[var]:
                if var < neighbor:
                    sharp_counting_numbers[(var, neighbor)] = 0.01

        bp = ConvexBeliefPropagator(mn, sharp_counting_numbers)
        bp.infer(display='full')
        bp.load_beliefs()
        print(bp.var_beliefs)

        bf = BruteForce(mn)
        print(bf.map_inference())
        assert (np.allclose(np.exp(bp.belief_mat), np.exp(bf.map_inference()))), "beliefs are not exact"


if __name__ == '__main__':
    unittest.main()
