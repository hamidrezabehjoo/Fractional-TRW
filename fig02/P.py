import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
 
#np.random.seed(0)
#################### Generate planar graph of size nxn #######################################################################
def planar_graph(n): 
    grid_size = n**2
    k = 2 #alphabet size
    mn = MarkovNet()
    for i in range(grid_size):
        mn.set_unary_factor(i, np.ones(k))
        #mn.set_unary_factor(i, np.random.rand(k) )

    for i in range(grid_size):
        for j in range(grid_size):
            if  j-i ==1 and j%n !=0 :
                print(i,j)
                u = np.random.uniform(0, 1, 1)[0]
                mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )
            if j-i == n:
                print(i,j)
                u = np.random.uniform(0, 1, 1)[0]
                mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )

    return mn



def complete_graph(n): 
    grid_size = n**2
    k = 2 #alphabet size
    mn = MarkovNet()
    for i in range(grid_size):
        #mn.set_unary_factor(i, np.ones(k))
        mn.set_unary_factor(i, np.random.rand(k) )

    for i in range(grid_size):
        for j in range(grid_size):
            if i>j:
               print(i,j)
               u = np.random.uniform(0, 1, 1)[0]
               mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )
    return mn



n = 40
grid_size = n**2
mn =  planar_graph(n)

####################### Assign Edge probabilities ###########################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    #edge_probabilities[edge] = np.random.uniform(0,1,1)[0]
    #edge_probabilities[edge] = 2/grid_size # in complete graph
    edge_probabilities[edge] = (n+1)/(2*n)  # for planar graph
#############################################################################################################

Z = []
tt = np.linspace(0, 1, 21)

for t in tt:

  for key, value in edge_probabilities.items():
      edge_probabilities[key] = value + t * (1-value)

  print(t)
  #trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
  trbp = TreeReweightedBeliefPropagator(mn, edge_probabilities)
  trbp.infer(display='off')
  trbp.load_beliefs()
  Z.append(trbp.compute_energy_functional())
  print ("TRBP matrix energy functional: %f" % trbp.compute_energy_functional())



np.savetxt("out/Z.txt", np.array(Z))


plt.figure(1)
plt.plot(tt, Z, 'ro', lw=2)
plt.xlim([0, 1])
plt.xlabel('$\lambda$')
plt.ylabel('$\log {Z^{(\lambda)}}$')
plt.grid()
plt.savefig("out/Z.pdf")
#############################################################################################################
