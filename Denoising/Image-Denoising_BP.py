import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import cv2


############################# Load Image #############################################

# Load the image
img = cv2.imread('images/cameraman.jpg')
resized_img = cv2.resize(img, (256, 256))

# Convert to grayscale
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY)

cv2.imwrite('binary_image.png', binary_img*255)
binary_img.shape


### Add noise
def add_noise(img, flip_prob):
    ''' Adds noise to a given binary image. Value of each pixel is flipped with given probability
    # ARGUMENTS
    # =========
    # img (numpy array): image to which noise is to be added
    # flip_prob (float \in [0,1]): probability with which each pixel value is flipped
    '''
    img_noisy = np.copy(img)
    for i in range(img_noisy.shape[0]):
        for j in range(img_noisy.shape[1]):
            if np.random.rand() <= flip_prob:
                img_noisy[i][j] = 1 - img[i][j]
    return img_noisy


img_noise = add_noise(binary_img, flip_prob = 0.2)
cv2.imwrite('img_noise.png', img_noise *255)
x = binary_img.flatten()
#################### Generate planar graph of size nxn #######################################################################
n = img_noise.shape[0]
grid_size = n**2
y = img_noise.flatten()

h = 0.7
J = 0.3

k = 2 #alphabet size

mn = MarkovNet()

for i in range(grid_size):
    mn.set_unary_factor(i, h * np.array( [y[i], 1-y[i]] ))
    #mn.set_unary_factor(i, np.random.rand(k) )

#print(mn.variables)
#print( mn.unary_potentials )


for i in range(grid_size):
    for j in range(grid_size):
        if  j-i ==1 and j%n !=0 :
            #print(i,j)
            #u = np.random.uniform(0, 1, 1)[0]
            u = J
            mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )
        if j-i == n:
            #print(i,j)
            #u = np.random.uniform(0, 1, 1)[0]
            u = J
            mn.set_edge_factor((i, j), np.array([[ np.exp(u) , np.exp(-u)], [np.exp(-u), np.exp(u)]]) )

#print(mn.variables)
print(mn.get_neighbors(145) )


####################### Assign Edge probabilities ###########################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    #edge_probabilities[edge] = np.random.uniform(0,1,1)[0]
    #edge_probabilities[edge] = 2/grid_size # in complete graph
    edge_probabilities[edge] = (n+1)/(2*n)  # for planar graph
##################### BP ####################################################################################

bp = BeliefPropagator(mn)
bp.infer(display='off')
bp.load_beliefs()
z_bp = bp.compute_energy_functional()
print("z_bp:\t", z_bp)
print( np.exp(bp.var_beliefs[0]) ) 

for var in mn.variables:
    unary_belief = np.exp(bp.var_beliefs[var])
    if unary_belief[0] > unary_belief[1]:
        y[var] = 1.0
    else:
        y[var] = 0.0

a = np.reshape(y, (n,n))
cv2.imwrite('cameraBP.png', a*255)
print("Error BP \t", 1- (np.sum(x ==y)/ grid_size) )



