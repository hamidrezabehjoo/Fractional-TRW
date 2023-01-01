import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

#################### Generate planar graph of size nxn #######################################################################
n = 3
grid_size = n**2
k = 2 #alphabet size

mn = MarkovNet()

for i in range(grid_size):
    mn.set_unary_factor(i, np.ones(k))
    #mn.set_unary_factor(i, np.random.rand(k) )

for i in range(grid_size):
    for j in range(grid_size):
        if i>j:
           print(i,j)
           u = np.random.uniform(0, 1, 1)[0]
           mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )
           #mn.set_edge_factor((i, j), np.random.rand(k, k))

#print(mn.variables)
#print(mn.get_neighbors(0) )
