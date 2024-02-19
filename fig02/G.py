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



n = 10
grid_size = n**2
mn =  planar_graph(n)
#mn =  complete_graph(n)


####################### Assign Edge probabilities ###########################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    #edge_probabilities[edge] = np.random.uniform(0,1,1)[0]
    #edge_probabilities[edge] = 2/grid_size # in complete graph
    edge_probabilities[edge] = (n+1)/(2*n)  # for planar graph
##################### BP ####################################################################################

#################### Compute Correction Factor ##############################################################
def B11(x, edge_probabilities, grid_size):
    num = 1
    for edge, _ in edge_probabilities.items():
        B = np.exp(trbp.pair_beliefs[edge])
        num *= B[x[edge[0]], x[edge[1]]] ** edge_probabilities[edge]
    return num


def B00(x, edge_probabilities, grid_size):
    den = 1
    for i in range(grid_size):
        sum1 = 0
        s = mn.get_neighbors(i)
        b = np.exp(trbp.var_beliefs[i])
        for a in s:
            if a < i:
               sum1 += edge_probabilities[(a,i)]
            else:
               sum1 += edge_probabilities[(i,a)]				
        den *= (b[x[i]])** sum1
    return den
   
def corr_factor(n_samples, n_MC):
    out = 0
    for i in range(n_MC):
        correction_factor = 0
        for k in range(n_samples):
            #x = bernoulli.rvs(0.5, size=grid_size)
            x = gen_samples(grid_size)
            a = B11(x, edge_probabilities, grid_size)
            b = B00(x, edge_probabilities, grid_size)
            correction_factor += a/b
        out += correction_factor /n_samples
    return np.log(out/n_MC)



def gen_samples(grid_size):
    x = np.zeros(grid_size) 
    for i in range(grid_size):
        p = np.exp(trbp.var_beliefs[i])
        x[i] = bernoulli.rvs(1-p[0], 0)
    return x.astype(int)



def grad(x, edge_probabilities):
    H_ab, H_a, H_b = 0, 0, 0
    for edge, weight in edge_probabilities.items():
        #print(weight)
        B = np.exp(trbp.pair_beliefs[edge])
        a = np.exp(trbp.var_beliefs[edge[0]])
        b = np.exp(trbp.var_beliefs[edge[1]])
        
        dummy = B[x[edge[0]], x[edge[1]]]
        cummy = a[x[edge[0]]]
        mummy = b[x[edge[1]]]
        H_ab += -dummy * np.log(dummy) *(1-weight)
        H_a  +=  cummy * np.log(cummy) *(1-weight)
        H_a  +=  mummy * np.log(mummy) *(1-weight)
    
    return H_ab + H_a + H_b
#############################################################################################################

Z = []
tt = np.linspace(0.05, 1, 21)

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
