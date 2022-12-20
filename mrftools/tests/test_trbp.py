import unittest
from mrftools import *
import numpy as np


class TestTreeBeliefPropagator(unittest.TestCase):
    def set_up_tree_model(self):
        mn = MarkovNet()

        np.random.seed(1)

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
            edge_probabilities[edge] = 1  # BP

        return mn, edge_probabilities

    def test_consistency(self):
        mn, edge_probabilities = self.set_up_tree_model()

        bp = TreeReweightedBeliefPropagator(mn, edge_probabilities)
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
        mn, edge_probabilities = self.set_up_tree_model()

        trbp = TreeReweightedBeliefPropagator(mn, edge_probabilities)
        trbp.infer(display='full')

        trbp.compute_beliefs()
        trbp.compute_pairwise_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(trbp.var_beliefs[var])
            assert np.allclose(np.sum(unary_belief), 1.0), "unary belief is not normalized"
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.exp(trbp.pair_beliefs[(var, neighbor)])
                assert np.allclose(np.sum(pair_belief), 1.0), "pairwise belief is not normalize"

    def test_tree_structured_model(self):
        mn, edge_probabilities = self.set_up_tree_model()

        trbp = TreeReweightedBeliefPropagator(mn, edge_probabilities)
        bp = BeliefPropagator(mn)

        trbp.infer(display='full')

        bp.infer()

        trbp.compute_pairwise_beliefs()

        bf = BruteForce(mn)

        for i in range(2):
            print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unary_marginal(i))))
            print ("TRBP unary marginal of %d:        %s" % (i, repr(np.exp(trbp.var_beliefs[i]))))
            assert np.allclose(bf.unary_marginal(i), np.exp(trbp.var_beliefs[i])), "TRBP not close to true unary"

        print ("Brute force pairwise marginal: " + repr(bf.pairwise_marginal(0, 1)))
        print ("TRBP pairwise marginal:        " + repr(np.exp(trbp.pair_beliefs[(0, 1)])))

        assert np.allclose(np.exp(trbp.pair_beliefs[(0, 1)]), bf.pairwise_marginal(0, 1)), "Pair beliefs don't match: " + \
            "\nTRBP:" + repr(np.exp(trbp.pair_beliefs[(0, 1)])) + "\nTrue:" + repr(bf.pairwise_marginal(0, 1))


        assert np.allclose(bf.pairwise_marginal(0, 1), np.exp(trbp.pair_beliefs[(0, 1)])), \
            "TRBP not close to pair marginal"

        print ("Tree Bethe energy functional:       %f" % trbp.compute_energy_functional())
        print ("Bethe energy functional:            %f" % bp.compute_energy_functional())
        print ("Brute force log partition function: %f" % np.log(bf.compute_z()))

        assert np.allclose(trbp.compute_energy_functional(), np.log(bf.compute_z()))

    def test_upper_bound(self):

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

            trbp = TreeReweightedBeliefPropagator(mn, edge_probabilities)
            trbp.infer(display='off')

            trbp_z = trbp.compute_energy_functional()
            true_z = np.log(bf.compute_z())

            print ("Tree Bethe energy functional:       %f" % trbp_z)
            print ("Brute force log partition function: %f" % true_z)

            print ("Is the TRBP energy functional an upper bound? %r" %
                   (trbp_z >= true_z))
            assert trbp_z >= true_z, "TRBP energy functional was lower than true log partition"

            tr_diff[trial] = trbp_z - true_z

            print("Difference range between variational Z and truth:")
            print("TRBP:  %f to %f" % (min(tr_diff[:trial + 1]), max(tr_diff[:trial + 1])))
            print("Average error. TRBP: %f" % np.mean(np.abs(tr_diff[:trial + 1])))


if __name__ == '__main__':
    unittest.main()
