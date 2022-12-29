import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
np.random.seed(1)

#################### Generate planar grap of size nxn ##################################################################
grid_size = 4**2

k = 2 # alphabet size

mn = MarkovNet()

for i in range(grid_size):
    mn.set_unary_factor(i, np.ones(k))

for i in range(grid_size):
    for j in range(grid_size):
        if np.abs(i-j)==1 or np.abs(i-j)==3 :
           u = np.random.uniform(0, 1, 1)[0]
           mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )
           #mn.set_edge_factor((i, j), np.random.rand(k, k))

print(mn.variables)
print(mn.get_neighbors(4) )

######################## Exact Value of Partition Function ####################################################################
bf = BruteForce(mn)
true_z = np.log(bf.compute_z())
print(true_z)


####################### Assign Edge probabilities #############################################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    edge_probabilities[edge] = np.random.uniform(0,1,1)[0]


##################### Fractional TRW ############################################################################################# 


trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()

print ("TRBP pairwise marginal:\n   " + repr(np.exp(trbp.pair_beliefs[(5, 8)])))
print ("TRBP matrix energy functional: %f" % trbp.compute_energy_functional())
for i in range(grid_size):
    print ("TRBP unary marginal of %d: %s" % (i, repr(np.exp(trbp.var_beliefs[i]))))

for i in range(grid_size):
    for j in range(grid_size):
        if np.abs(i-j)==1 or np.abs(i-j)==3 :
          print( np.exp(trbp.pair_beliefs[(i,j)]) )
          print("\n")



