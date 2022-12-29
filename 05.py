import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
np.random.seed(0)

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
z_true = np.log(bf.compute_z())
print("z_true:\t", z_true)


####################### Assign Edge probabilities ###########################################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    edge_probabilities[edge] = np.random.uniform(0,1,1)[0]


##################### Fractional TRW ########################################################################################
trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()
z_trw = trbp.compute_energy_functional()
print("z_trw:\t", z_trw)


#################### Compute Correction Factor ##############################################################################
def B1(x, edge_probabilities, grid_size):
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
    return den



def B11(x, edge_probabilities, grid_size):
    num = 1
    for edge, _ in edge_probabilities.items():
        B = np.exp(trbp.pair_beliefs[edge])
        if x[edge[0]]==x[edge[1]]:
           num *= B[0,0] ** edge_probabilities[edge]
           #print("Hi")
        else:
           num *= B[0,1] ** edge_probabilities[edge]
           #print('Salam')
    return num
    


def corr_factor(n_samples):
    correction_factor = 0
    for k in range(n_samples):
        x = bernoulli.rvs(0.5, size=grid_size)
        a = B11(x, edge_probabilities, grid_size)
        b =  B1(x, edge_probabilities, grid_size)
        correction_factor += a/b
    return np.log(correction_factor/n_samples)


corr = corr_factor(1000)

print("Estim:\t", corr)

print("Exact:\t", z_true - z_trw)

############################################################################################################################################
cc = 0
n_mc = 10
for i in range(n_mc):
    cc += corr_factor(1000)

print(cc/n_mc)



