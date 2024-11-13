import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class LoopyBPImageDenoiser():
    def __init__(self):
        pass
 
    def show_input(self):
        plt.imshow(self.img_in)
        
    def init_bethe_CG(self):
        self.init_nodes()
        self.init_node_potentials()
        self.init_edge_potentials()
        
    def init_nodes(self):
        '''Initialize Cluster Graph Node Names'''
        self.x_nodes = []
        self.y_nodes = []
        self.yx_nodes = []
        self.xx_nodes = []
        
        m = self.img_in.shape[0]
        n = self.img_in.shape[1]
        
        self.m = m
        self.n = n
        
        for i in range(m):
            for j in range(n):                    
                self.x_nodes.append('x' + str(i) + '.' + str(j))
                self.y_nodes.append('y' + str(i) + '.' + str(j))
                self.yx_nodes.append('y' + str(i) + '.' + str(j) + '_x' + str(i) + '.' + str(j))
                if i!=m-1:
                    self.xx_nodes.append('x' + str(i) + '.' + str(j) + '_x' + str(i+1) + '.' + str(j))
                if j!=n-1:
                    self.xx_nodes.append('x' + str(i) + '.' + str(j) + '_x' + str(i) + '.' + str(j+1))

                    
 
    def init_node_potentials(self):
        '''Initialize Cluster Graph Node Potentials as a dictionary'''
        self.node_potentials = {}
        for node in self.y_nodes:
            i, j = node[1:].split('.')
            pixel_val = self.img_in[int(i)][int(j)]
            node_potential = [abs(0.95 - pixel_val), abs(0.05 - pixel_val)]
            self.node_potentials[node] = np.array(node_potential)
            
        for node in self.x_nodes:
            node_potential = np.ones(2)
            self.node_potentials[node] = node_potential
            
        for node in self.xx_nodes:
            var1, var2 = node.split('_')
            node_potential = [[1 + self.theta, 1 - self.theta], [1 - self.theta, 1 + self.theta]]
            self.node_potentials[node] = np.array(node_potential)
        
        for node in self.yx_nodes:
            node_potential = [[1 + self.gamma, 1 - self.gamma], [1 - self.gamma, 1 + self.gamma]]
            self.node_potentials[node] = np.array(node_potential)
                        
                        
    def init_edge_potentials(self):
        '''Initialize Cluster Graph Edge Potentials as a defaultdict'''
        self.edge_potentials = defaultdict(lambda:np.ones(2))
        
    def denoise(self, img_in, theta, gamma, n_iters=3):
        self.gamma = gamma
        self.theta = theta
        self.img_in = img_in
        self.init_bethe_CG()
        for i in tqdm(range(n_iters)):
            self.single_pass()
        self.img_out = np.ones([m, n])
        for i in range(m):
            for j in range(n):
                key = 'x' + str(i) + '.' + str(j)
                self.img_out[i][j] = np.argmax(denoiser.node_potentials[key])
        return self.img_out
    
    def single_pass(self, verbose=False):
        for node in self.y_nodes:
            x_node = 'x' + node[1:]
            target_node = node + '_' + x_node
            self.send_message(node, target_node, verbose)
            
        for node in self.yx_nodes:
            target_node = node.split('_')[1]
            self.send_message(node, target_node, verbose)
            
        for node in self.x_nodes:
            i,j = node[1:].split('.')
            rnode = 'x' + i + '.' + str(int(j) + 1)
            lnode = 'x' + i + '.' + str(int(j) - 1)
            unode = 'x' + str(int(i) - 1) + '.' + j
            dnode = 'x' + str(int(i) + 1) + '.' + j
            
            if int(j) + 1 < self.m:
                target_node = node + '_' + rnode
                self.send_message(node, target_node, verbose)
            if int(j) - 1 >= 0:
                target_node =  lnode + '_'+ node
                self.send_message(node, target_node, verbose)
            if int(i) - 1>= 0:
                target_node = unode + '_' + node
                self.send_message(node, target_node, verbose)
            if int(i) + 1 < self.n:
                target_node = node + '_' + dnode  
                self.send_message(node, target_node, verbose)

        for node in self.xx_nodes:
            var1, var2 = node.split('_')
            self.send_message(node, var1, verbose)
            self.send_message(node, var2, verbose)
        
#         self.normalize()
        
    def normalize(self):
        m = self.m
        n = self.n
        n_nodes = 3*m*n + m*(n-1) + n*(m-1)
        normalizing_factor = 0
        for node in self.x_nodes:
            normalizing_factor += self.node_potentials[node].sum()
        for node in self.y_nodes:
            normalizing_factor += self.node_potentials[node].sum()
        for node in self.yx_nodes:
            normalizing_factor += self.node_potentials[node].sum()
        for node in self.xx_nodes:
            normalizing_factor += self.node_potentials[node].sum()
        
        normalizing_factor = normalizing_factor/n_nodes
        print('\nNormalizing Factor = ', normalizing_factor)
        
        for node in self.x_nodes:
            self.node_potentials[node] *= 1/normalizing_factor
        for node in self.y_nodes:
            self.node_potentials[node] *= 1/normalizing_factor
        for node in self.yx_nodes:
            self.node_potentials[node] *= 1/normalizing_factor
        for node in self.xx_nodes:
            self.node_potentials[node] *= 1/normalizing_factor
                
                
    def send_message(self, node1, node2, verbose=False):
        if not '_' in node1:
            sigma = self.node_potentials[node1]
        else:
            var1, var2 = node1.split('_')
            if node2 == var1:
                sigma = self.max_marginalize(node1, var2 )
            elif node2 == var2:
                sigma = self.max_marginalize(node1, var1)
        edge = node1 + node2
        initial_edge_potential = self.edge_potentials[edge]
        message = self.factor_div(sigma, self.edge_potentials[edge])
        initial_potential = self.node_potentials[node2]
        orientation = self.is_edge(node1, node2)
        self.node_potentials[node2] = self.factor_prod(message, self.node_potentials[node2], orientation)
        final_potential = self.node_potentials[node2]
        self.edge_potentials[edge] = sigma
        final_edge_potential = self.edge_potentials[edge]
        if verbose:
            print('\nmessage from: ', node1, '-->', node2)
            print('---------------------------------------')
            print('\nsigma: ', sigma)
            print('\nedge_potential = ', initial_edge_potential)
            print('\nmessage', node1,'-', node2,' = ', sigma, '/', initial_edge_potential, ' = ', message)
            print('\nInitial', node2, 'potential: ', initial_potential)
            print('\nFinal', node2, 'potential: ', final_potential)

    
    def is_edge(self, node1, node2):
        if '_' in node1:
            sep_set = node2
            l = len(node1)
            var1, var2 = node1.split('_')
        else:
            sep_set = node1
            var1, var2 = node2.split('_')
            
        if sep_set == var1:
            return -1
        elif sep_set == var2:
            return 1
        else:
            return 0
            
        
    def max_marginalize(self, node, var):
        ''' 'Max Marginalize' the node_potential of the input node over the input variable ''' 
        
        variables = node.split('_')
        if var == variables[0]:
            axis = 0
#             print('\nMarginalize Node Potential: ',node, ' ', self.node_potentials[node], ' on axis ', axis)
            return np.max(self.node_potentials[node], axis=axis)
        elif var == variables[1]:
            axis = 1
#             print('\nMarginalize Node Potential: ', node, ' ', self.node_potentials[node], ' on axis ', axis)
            return np.max(self.node_potentials[node], axis=axis)
        
    def factor_prod(self, f1, f2, orientation):
        if orientation==-1:
            return f1*f2
        elif orientation==1:
            return (f1*f2.T).T
         
    def factor_div(self, f1, f2):
        '''
        Divide factor f1 by factor f2
        
        Arguments:
        =========
        f1 (np.ndarray): Dividend
        f2 (np.ndarray): Divisor
        
        Constraints:
        f1.shape = (n,) , n \in N
        f2.shape = (n,)
        f2[i]=0 ==> f1[i]=0 \forall i in n
        '''
        f1_c = np.copy(f1)
        f2_c = np.copy(f2)
        for i in range(len(f2)):
            if f2[i] == 0:
                f2_c[i] = 1
                if f1[i] > 0:
                    print(f1[i],'/0 ','encountered')
                else:
                    f1_c[i] = 1                    
        return f1_c/f2_c
    
    def compare_with_original(self, target_img):
        ''' 
        Compare how close the output denoised image is to the target image. 
        Returns number of pixels where target image values differ from output image 
        
        Arguments:
        =========
        target_img (np.ndarray, mxn): Target Image 
        
        '''
        diff = np.sum(np.square(target_img - self.img_out))
        return diff
        

if __name__ == '__main__':
    print('Code execution started')
    # df = pd.read_csv('path//to//file')
    df = pd.read_csv('F:/Sanjeev/test_csv.csv')
    m = int(np.max(df[df.columns[0]])) + 1 
    n = int(np.max(df[df.columns[1]])) + 1
    img_in = np.ones([m,n])
    for index, row in df.iterrows():
        row_index = row[df.columns[0]]
        col_index = row[df.columns[1]]
        value = row[df.columns[2]]
        img_in[int(row_index) - 1][int(col_index) - 1] = value

    denoiser = LoopyBPImageDenoiser()
    gamma = 0.2
    theta = 0.1

    img_out = denoiser.denoise(img_in, theta, gamma)

    data = np.ones([m*n,3])

    for i in np.arange(m):
        for j in np.arange(n):
            data[m*i + j][0] = i
            data[m*i + j][1] = j
            data[m*i + j][2] = img_out[i][j]   

    df = pd.DataFrame(data, columns=['Row Index', 'Column Index', 'Value']) 

    df.to_csv('F:/Sanjeev/test_output.csv', index=False)    
    # df.to_csv('Output//Path//Name.csv', index=False)    
