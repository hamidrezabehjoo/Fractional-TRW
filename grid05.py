import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
np.random.seed(0)

#################### Generate planar graph of size nxn #######################################################################
grid_size = 5**2

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
#bf = BruteForce(mn)
#z_true = np.log(bf.compute_z())
#print("z_true:\t", z_true)
z_true = 105.49893569871489

####################### Assign Edge probabilities ###########################################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    edge_probabilities[edge] = np.random.uniform(0,1,1)[0]


##################### Fractional TRW ########################################################################################
"""
trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()
z_trw = trbp.compute_energy_functional()
print("z_trw:\t", z_trw)
"""
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
    


def corr_factor(n_samples, n_MC):
    for i in range(n_MC):
        correction_factor = 0
        for k in range(n_samples):
            x = bernoulli.rvs(0.5, size=grid_size)
            a = B11(x, edge_probabilities, grid_size)
            b =  B1(x, edge_probabilities, grid_size)
            correction_factor += a/b
    return np.log(correction_factor/n_samples)

############################################################################################################
Z = []
C = []
tt = np.linspace(0, 1, 21)

for t in tt:

  for key, value in edge_probabilities.items():
      edge_probabilities[key] = value + t * (1-value)

  print(t)
  trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
  trbp.infer(display='off')
  trbp.load_beliefs()
  C.append(corr_factor(grid_size**4, 10))
  Z.append(trbp.compute_energy_functional())
  print ("TRBP matrix energy functional: %f" % trbp.compute_energy_functional())


plt.figure(0)
plt.plot(tt, C, 'bo', lw=2)
plt.plot(tt, 0 * np.ones(tt.shape), lw = 2)
#plt.xlim([1e-2, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$\log {C^{(\lambda)}}$')
#plt.grid()
plt.savefig("4C_FTRW.pdf")


plt.figure(1)
plt.plot(tt, Z, 'ro', lw=2)
plt.plot(tt, z_true * np.ones(tt.shape), lw = 2)
#plt.xlim([1e-2, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$\log {Z^{(\lambda)}}$')
#plt.grid()
plt.savefig("4Z_FTRW.pdf")

############################################################################################################################################




