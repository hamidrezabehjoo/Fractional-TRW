import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
path = "3x3"
np.random.seed(42)
bernoulli.random_state = np.random.RandomState(42)

#################### Generate planar graph of size nxn #######################################################################
n = 3
grid_size = n**2
k = 2 #alphabet size

mn = MarkovNet()

for i in range(grid_size):
    #mn.set_unary_factor(i, np.ones(k))
    mn.set_unary_factor(i, np.random.rand(k) )

for i in range(grid_size):
    for j in range(grid_size):
        if i>j:
           u = np.random.uniform(0, 1, 1)[0]
           mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )
           #mn.set_edge_factor((i, j), np.random.rand(k, k))

#print(mn.variables)
#print(mn.get_neighbors(0) )
####################### Assign Edge probabilities ###########################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    #edge_probabilities[edge] = np.random.uniform(0,1,1)[0]
    edge_probabilities[edge] = 2/grid_size # in complete graph
    #edge_probabilities[edge] = (n+1)/(2*n)  # for planar graph
##################### BP ####################################################################################
bf = BruteForce(mn)
z_true = np.log(bf.compute_z())
print("z_true:\t", z_true)

trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()
z_trw = trbp.compute_energy_functional()
print("z_trw:\t", z_trw)

bp = BeliefPropagator(mn)
bp.infer(display='off')
bp.load_beliefs()
z_bp = bp.compute_energy_functional()
print("z_bp:\t", z_bp)

#################### Compute Correction Factor ##############################################################
def B11(x, updated_edge_probabilities, grid_size):
    num = 1
    for edge, _ in updated_edge_probabilities.items():
        B = np.exp(trbp.pair_beliefs[edge])
        num *= B[x[edge[0]], x[edge[1]]] ** updated_edge_probabilities[edge]
    return num


def B00(x, updated_edge_probabilities, grid_size):
    den = 1
    for i in range(grid_size):
        sum1 = 0
        s = mn.get_neighbors(i)
        b = np.exp(trbp.var_beliefs[i])
        for a in s:
            if a < i:
               sum1 += updated_edge_probabilities[(a,i)]
            else:
               sum1 += updated_edge_probabilities[(i,a)]				
        den *= (b[x[i]])** sum1
    return den
   
def corr_factor(n_samples, n_MC):
    out = 0
    for i in range(n_MC):
        correction_factor = 0
        for k in range(n_samples):
            #x = bernoulli.rvs(0.5, size=grid_size)
            x = gen_samples(grid_size)
            a = B11(x, updated_edge_probabilities, grid_size)
            b = B00(x, updated_edge_probabilities, grid_size)
            correction_factor += a/b
        out += correction_factor /n_samples
    return np.log(out/n_MC)



def gen_samples(grid_size):
    x = np.zeros(grid_size) 
    for i in range(grid_size):
        p = np.exp(trbp.var_beliefs[i])
        x[i] = bernoulli.rvs(1-p[0], 0)
    return x.astype(int)



def grad(x, updated_edge_probabilities):
    H_ab, H_a, H_b = 0, 0, 0
    for edge, weight in updated_edge_probabilities.items():
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
C = []
G = []
tt = np.linspace(0, 1, 21)

updated_edge_probabilities = edge_probabilities.copy()

for t in tt:
   for key, value in edge_probabilities.items():
        updated_edge_probabilities[key] = value + t * (1 - value)

   trbp = MatrixTRBeliefPropagator(mn, updated_edge_probabilities)
   trbp.infer(display='off')
   trbp.load_beliefs()
   C.append(corr_factor(grid_size**6, 10))
   Z.append(trbp.compute_energy_functional())
   x = gen_samples(grid_size)
   G.append(grad(x, updated_edge_probabilities))
   print ("TRBP matrix energy functional: %f" % t, trbp.compute_energy_functional())





np.savetxt(path + "/Z.txt", np.array(Z))
np.savetxt(path + "/C.txt", np.array(C))
np.savetxt(path + "/G.txt", np.array(G))



plt.figure(1)
plt.plot(tt, Z, 'go', lw=2)
plt.plot(tt, z_true * np.ones(tt.shape),'--', lw = 1)
plt.xlim([0, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$\log {Z^{(\lambda)}}$')
plt.grid()
plt.savefig(path + "/Z_FTRW.pdf")


plt.figure(0)
plt.plot(tt, C, 'bo', lw=2)
plt.plot(tt, 0 * np.ones(tt.shape),'--', lw = 1)
plt.xlim([0, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$\log {C^{(\lambda)}}$')
plt.grid()
plt.savefig(path + "/C_FTRW.pdf")




plt.figure(2)
plt.plot(tt, G, 'ro', lw=2)
plt.plot(tt, -0.5 * np.ones(tt.shape),'--', lw = 1)
plt.xlim([0, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$G$')
plt.grid()
plt.savefig(path + "/G.pdf")
#############################################################################################################

