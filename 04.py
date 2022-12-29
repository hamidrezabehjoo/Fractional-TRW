import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
np.random.seed(1)

#################### Generate planar graph of size nxn #######################################################################
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

#print(mn.variables)
#print(mn.get_neighbors(4) )

######################## Exact Value of Partition Function ##################################################################
bf = BruteForce(mn)
#true_z = np.log(bf.compute_z())
#print(true_z)


####################### Assign Edge probabilities ###########################################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    edge_probabilities[edge] = np.random.uniform(0,1,1)[0]


##################### Fractional TRW ########################################################################################


trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()
"""
print ("TRBP pairwise marginal:\n   " + repr(np.exp(trbp.pair_beliefs[(5, 8)])))
print ("TRBP matrix energy functional: %f" % trbp.compute_energy_functional())
for i in range(grid_size):
    print ("TRBP unary marginal of %d: %s" % (i, repr(np.exp(trbp.var_beliefs[i]))))

for i in range(grid_size):
    for j in range(grid_size):
        if np.abs(i-j)==1 or np.abs(i-j)==3 :
          print( np.exp(trbp.pair_beliefs[(i,j)]) )
          print("\n")
"""
A = np.exp(trbp.pair_beliefs[(0,1)])
print(A.shape)



x = bernoulli.rvs(0.5, size=25)
#print(x)


den = 1

for i in range(grid_size):

    sum1 = 0
    s = mn.get_neighbors(i)
    for a in s:
        if a < i:
            #print(edge_probabilities[(a,i)])
            sum1 += edge_probabilities[(a,i)]
        else:
	        #print(edge_probabilities[(i,a)])
	        sum1 += edge_probabilities[(i,a)]

    den *= (0.5)** sum1
print("denumerator:\t", den)

num = 1
for edge, _ in edge_probabilities.items():
    B = np.exp(trbp.pair_beliefs[edge])
    if x[edge[0]]==x[edge[1]]:
       num *= B[0,0] ** edge_probabilities[edge]
    else:
       num *= B[0,1] ** edge_probabilities[edge]
    #print(edge)
    #print(type(edge))
    #print(edge[0], edge[1])
    #print(edge_probabilities[edge])

    #print(np.exp(trbp.pair_beliefs[edge]))
    #break
print("numerator:\t",num)

print(num/den)
