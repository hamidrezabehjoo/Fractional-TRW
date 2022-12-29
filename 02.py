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
Z = []
tt = np.linspace(0, 1, 51)

for t in tt:

  for key, value in edge_probabilities.items():
      edge_probabilities[key] = value + t * (1-value)

  print(t)
  trbp_mat = MatrixTRBeliefPropagator(mn, edge_probabilities)
  trbp_mat.infer(display='off')
  trbp_mat.load_beliefs()

  Z.append(trbp_mat.compute_energy_functional())
  print ("TRBP matrix energy functional: %f" % trbp_mat.compute_energy_functional())

plt.plot(tt, Z, lw=2)
plt.plot(tt, true_z * np.ones(tt.shape), lw = 2)
plt.xlim([1e-2, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$\log {Z^{(\lambda)}}$')
#plt.grid()
plt.savefig("Z_FTRW.pdf")

