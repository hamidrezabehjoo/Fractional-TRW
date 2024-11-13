import unittest
from mrftools import *
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import cv2


############################# Load Image #############################################

# Load the image
img = cv2.imread('images/cameraman.jpg')
resized_img = cv2.resize(img, (16, 16))

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


img_noise = add_noise(binary_img, flip_prob = 0.1)
cv2.imwrite('img_noise.png', img_noise *255)
x = binary_img.flatten()
#################### Generate planar graph of size nxn #######################################################################
n = img_noise.shape[0]
grid_size = n**2
y = img_noise.flatten()

h = 1.1
J = 0.3

k = 2 #alphabet size

mn = MarkovNet()

for i in range(grid_size):
    mn.set_unary_factor(i, h * np.array( [y[i], 1-y[i] ] ))
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

print(mn.variables)
#print(mn.get_neighbors(145) )


####################### Assign Edge probabilities ###########################################################
edge_probabilities = dict()

for edge in mn.edge_potentials:
    #edge_probabilities[edge] = np.random.uniform(0,1,1)[0]
    #edge_probabilities[edge] = 2/grid_size # in complete graph
    edge_probabilities[edge] = (n+1)/(2*n)  # for planar graph

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
            #print(a, b)
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

C = []

tt = np.linspace(0, 1, 11)

for t in tt:

  for key, value in edge_probabilities.items():
      edge_probabilities[key] = value + t * (1-value)

  print(t)
  trbp = MatrixTRBeliefPropagator(mn, edge_probabilities)
  trbp.infer(display='off')
  trbp.load_beliefs()
  C.append(corr_factor(grid_size**2, 1))
  print(C)
  #Z.append(trbp.compute_energy_functional())
  x = gen_samples(grid_size)
  #G.append(grad(x, edge_probabilities))
  print ("TRBP matrix energy functional: %f" % trbp.compute_energy_functional())



np.savetxt("C.txt", np.array(C))


plt.figure(0)
plt.plot(tt, C, 'bo', lw=2)
plt.plot(tt, 0 * np.ones(tt.shape),'--', lw = 1)
plt.xlim([0, 1])
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('$\log {C^{(\lambda)}}$')
plt.grid()
plt.savefig("C_FTRW.pdf")


