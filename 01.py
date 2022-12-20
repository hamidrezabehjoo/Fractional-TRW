import unittest
from mrftools import *
import numpy as np


print('Hi')
trials = 10

tr_diff = np.zeros(trials)
bp_diff = np.zeros(trials)

for trial in range(trials):

    mn = MarkovNet()

    width =  5
    height = 5

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
        edge_probabilities[edge] = 0.1

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
    #true_z = np.log(bf.compute_z())

    print ("Tree Bethe energy functional:       %f" % trbp_z)
    #print ("Brute force log partition function: %f" % true_z)

    #print ("Is the TRBP energy functional an upper bound? %r" %
    #       (trbp_z >= true_z))
    #assert trbp_z >= true_z, "TRBP energy functional was lower than true log partition"

    #tr_diff[trial] = trbp_z - true_z

  #  print("Difference range between variational Z and truth:")
 #   print("TRBP:  %f to %f" % (min(tr_diff[:trial + 1]), max(tr_diff[:trial + 1])))
#    print("Average error. TRBP: %f" % np.mean(np.abs(tr_diff[:trial + 1])))


#print(edge_probabilities)

