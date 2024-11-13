import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

np.random.seed(42)
bernoulli.random_state = np.random.RandomState(42)
#################### Generate planar graph of size nxn #######################################################################
n = 5
grid_size = n**2
k = 2 #alphabet size

mn = MarkovNet()

for i in range(grid_size):
    #mn.set_unary_factor(i, np.ones(k))
    mn.set_unary_factor(i, np.random.rand(k) )

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

#print(mn.variables)
#print(mn.get_neighbors(0) )
####################### Assign Edge probabilities ###########################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    #edge_probabilities[edge] = np.random.uniform(0,1,1)[0]
    #edge_probabilities[edge] = 2/grid_size # in complete graph
    edge_probabilities[edge] = (n+1)/(2*n)  # for planar graph
##################### BP ####################################################################################
#bf = BruteForce(mn)
#z_true = np.log(bf.compute_z())
#print("z_true:\t", z_true)

trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()
z_trw = trbp.compute_energy_functional()
print("z_trw:\t", z_trw)

#bp = BeliefPropagator(mn)
#bp.infer(display='off')
#bp.load_beliefs()
#z_bp = bp.compute_energy_functional()
#print("z_bp:\t", z_bp)

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
    h = []
    out = 0
    for i in range(n_MC):
        correction_factor = 0
        for k in range(n_samples):
            #x = bernoulli.rvs(0.5, size=grid_size)
            x = gen_samples(grid_size)
            a = B11(x, updated_edge_probabilities, grid_size)
            b = B00(x, updated_edge_probabilities, grid_size)
            correction_factor += a/b
        out = correction_factor /n_samples
        print(out)
        h.append(out)
    return h



def gen_samples(grid_size):
    x = np.zeros(grid_size)
    for i in range(grid_size):
        p = np.exp(trbp.var_beliefs[i])
        #p = np.exp(trbp.var_beliefs[i])
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

tt = np.linspace(0.6, 0.8, 21)

# Create a copy of the edge_probabilities dictionary
updated_edge_probabilities = edge_probabilities.copy()

for t in tt:
    for key, value in edge_probabilities.items():
        updated_edge_probabilities[key] = value + t * (1 - value)

# Now updated_edge_probabilities contains the modified values
# while edge_probabilities remains unchanged

    trbp = MatrixTRBeliefPropagator(mn, updated_edge_probabilities)
    trbp.infer(display='off')
    trbp.load_beliefs()
    z_trw = trbp.compute_energy_functional()
    print ("TRBP matrix energy functional: %f" % t, z_trw)

print('______________________print single value________________________')
t = 0.81  # exact value for 5x5 grid
#t = 0.67 # exact value for 4x4 grid 

for key, value in edge_probabilities.items():
    updated_edge_probabilities[key] = value + t * (1-value)

trbp = MatrixTRBeliefPropagator(mn, updated_edge_probabilities)
trbp.infer(display='off')
trbp.load_beliefs()
z_trw = trbp.compute_energy_functional()
print ("TRBP matrix energy functional: %f" % t, z_trw)

c = corr_factor(grid_size**3, 100)
#print(c)
cc = np.asarray(c)
cc = z_trw + np.log(cc)
mean, std = cc.mean(), cc.std()

print(mean, std)


