import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


m = 3
n = 3
grid_size = m * n

k = 2 # alphabet size

mn = MarkovNet()

for i in range(grid_size):
    for j in range(grid_size):
        if  j-i ==1 and j%n !=0 :
            print(i,j)
        if j-i == n:
            print("Second", i,j)

"""
print("Salam")         
for i in range(grid_size):
    for j in range(grid_size):
        if  j-i == n: 
            print(i,j)
"""
