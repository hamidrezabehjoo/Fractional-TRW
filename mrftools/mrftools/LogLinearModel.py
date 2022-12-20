"""Class to convert from log linear model to MRF"""
import numpy as np
from scipy.sparse import csr_matrix

from .MarkovNet import MarkovNet


class LogLinearModel(MarkovNet):
    """
    Log linear model class. Able to convert from log linear features to pairwise MRF.
    """

    def __init__(self):
        """Initialize a LogLinearModel. Create a Markov net."""
        super(LogLinearModel, self).__init__()
        self.unary_features = dict()
        self.unary_feature_weights = dict()
        self.edge_features = dict()
        self.num_features = dict()
        self.num_edge_features = dict()

        # matrix mode placeholders
        self.weight_dim = None
        self.max_states = None
        self.max_unary_features = None
        self.max_edge_features = None
        self.unary_weight_mat = None
        self.edge_weight_mat = None
        self.unary_feature_mat = None
        self.edge_feature_mat = None

    def set_edge_factor(self, edge, potential):
        """
        Set a factor by inputting the involved variables then the potential function. 
        The potential function should be a np matrix. 
        
        This method implicitly creates a single-dimensional edge feature set to value 1.0.

        :param edge: 2-tuple of the variables in the edge. Can be in any order.
        :param potential: k1 by k2 matrix of potential values for the joint state of the two variables
        :return: None
        """
        super(LogLinearModel, self).set_edge_factor(edge, potential)
        if edge not in self.edge_features:
            # set default edge feature
            self.set_edge_features(edge, np.array([1.0]))

    def set_unary_weights(self, var, weights):
        """
        Set the log-linear weights for the unary features of var. Used for non-matrix mode only.    
        
        :param var: variable whose weights to set.
        :param weights: ndarray weight matrix of shape (self.num_states[var], len(self.unary_features[var]))
        :return: None
        """
        assert isinstance(weights, np.ndarray)
        assert weights.shape[0] == self.num_states[var]
        self.unary_feature_weights[var] = weights

    def set_unary_features(self, var, values):
        """
        Set the log-linear features for a particular variable.  Used for non-matrix mode only.
        :param var: variable whose features to set
        :param values: ndarray feature vector describing the unary variable of any length
        :return: None
        """
        assert isinstance(values, np.ndarray)
        self.unary_features[var] = values

        self.num_features[var] = len(values)

    def set_edge_features(self, edge, values):
        """
        Set the log-linear feature for a particular edge.  Used for non-matrix mode only. Currently does not work.
        :param edge: pair of variables representing the edge being set
        :param values: ndarray of feature values describing the edge
        :return: None
        """
        reversed_edge = (edge[1], edge[0])
        self.edge_features[edge] = values
        self.num_edge_features[edge] = len(values)

        self.edge_features[reversed_edge] = values
        self.num_edge_features[reversed_edge] = len(values)

    def set_all_unary_factors(self):
        """
        Uses current weights and features to set the potential functions for all unary factors.
        Used for non-matrix mode only.
        
        :return: None
        """
        for var in self.variables:
            self.set_unary_factor(var, self.unary_feature_weights[var].dot(self.unary_features[var]))

    def set_feature_matrix(self, feature_mat):
        """
        Set matrix of features for all unary variables. Used for matrix mode.
        
        :param feature_mat: ndarray of shape (max_unary_features, len(variables)) with variables as ordered in self.variables.
                            Each jth column is the jth variable's feature vector.
        :return: None
        """
        assert (np.array_equal(self.unary_feature_mat.shape, feature_mat.shape))

        self.unary_feature_mat[:, :] = feature_mat

    def set_weights(self, weight_vector):
        """
        Set the unary and edge weight matrices by splitting and reshaping a weight vector. Useful for optimization when
        the optimizer is searching for a vector value. Used for matrix mode.
        
        :param weight_vector: real vector of length self.max_unary_features * self.max_state + 
                                self.max_edge_features * self.max_states ** 2
        :return: None
        """
        num_vars = len(self.variables)

        feature_size = self.max_unary_features * self.max_states
        feature_weights = weight_vector[:feature_size].reshape((self.max_unary_features, self.max_states))

        pairwise_weights = weight_vector[feature_size:].reshape((self.max_edge_features, self.max_states ** 2))

        self.set_unary_weight_matrix(feature_weights)
        self.set_edge_weight_matrix(pairwise_weights)

        self.update_unary_matrix()
        self.update_edge_tensor()

    def set_unary_weight_matrix(self, weight_mat):
        """
        Set unary weight matrix. Convenience method that also checks the shape of the new matrix. Used for matrix mode.
        
        Developer note: this method may not be necessary. It's meant to copy the matrix into the original weight matrix,
        memory but that may create problems for autodiff software in the future.
        
        :param weight_mat: ndarray of shape (self.max_unary_features, self.max_states)
        :type weight_mat: ndarray
        :return: None
        """
        assert (np.array_equal(self.unary_weight_mat.shape, weight_mat.shape))
        self.unary_weight_mat[:, :] = weight_mat

    def set_edge_weight_matrix(self, edge_weight_mat):
        """
        Set edge weight matrix. Used for matrix mode.
        
        See developer note in set_unary_weight_matrix.
        
        :param edge_weight_mat: ndarray of shape (self.max_edge_features, self.max_states ** 2)
        :type edge_weight_mat: ndarray
        :return: None
        """
        assert (np.array_equal(self.edge_weight_mat.shape, edge_weight_mat.shape))
        self.edge_weight_mat[:, :] = edge_weight_mat

    def update_unary_matrix(self):
        """
        Set the unary potential matrix by multiplying the feature matrix by the weight matrix.
        :return: None
        """
        self.set_unary_mat(self.unary_feature_mat.T.dot(self.unary_weight_mat).T)

    def update_edge_tensor(self):
        """
        Set the edge potential tensor by multiplying the edge feature matrix by the edge weight matrix, reshaping, and
        duplicating (to allow the tensor to contain the appropriate values for forward and backward messages.
        
        Used for matrix mode.
         
        :return: None
        """
        half_edge_tensor = self.edge_feature_mat.T.dot(self.edge_weight_mat).T.reshape(
            (self.max_states, self.max_states, self.num_edges))
        self.edge_pot_tensor[:, :, :] = np.concatenate((half_edge_tensor.transpose(1, 0, 2), half_edge_tensor), axis=2)

    def create_matrices(self):
        """
        Create matrix representations of the MRF structure and log-linear model to allow inference to be done via 
        matrix operations.
        :return: None
        """
        super(LogLinearModel, self).create_matrices()

        # create unary matrices
        self.max_unary_features = max([x for x in self.num_features.values()])
        self.unary_weight_mat = np.zeros((self.max_unary_features, self.max_states))
        self.unary_feature_mat = np.zeros((self.max_unary_features, len(self.variables)))

        for var in self.variables:
            index = self.var_index[var]
            self.unary_feature_mat[:, index] = self.unary_features[var]

        # create edge matrices
        self.max_edge_features = max([x for x in self.num_edge_features.values()] or [0])
        self.edge_weight_mat = np.zeros((self.max_edge_features, self.max_states ** 2))
        self.edge_feature_mat = np.zeros((self.max_edge_features, self.num_edges))

        for edge, i in self.message_index.items():
            self.edge_feature_mat[:, i] = self.edge_features[edge]

        self.weight_dim = self.max_states * self.max_unary_features + self.max_edge_features * self.max_states ** 2

    def create_indicator_model(self, markov_net):
        """
        Sets this object to be a log-linear model representation of a Markov Net to enable directly learning the 
        potential values. Each feature vector is an indicator vector, such that each dimension of the weights 
        corresponds to exactly one variable or edge.
         
        :param markov_net: Markov network to build the indicator model of
        :type markov_net: MarkovNet
        :return: None
        """
        n = len(markov_net.variables)

        # set unary variables
        for i, var in enumerate(markov_net.variables):
            self.declare_variable(var, num_states=markov_net.num_states[var])
            self.set_unary_factor(var, markov_net.unary_potentials[var])
            indicator_features = np.zeros(n)
            indicator_features[i] = 1.0
            self.set_unary_features(var, indicator_features)

        # count edges
        num_edges = 0
        for var in markov_net.variables:
            for neighbor in markov_net.get_neighbors(var):
                if var < neighbor:
                    num_edges += 1

        # create edge indicator features
        i = 0
        for var in markov_net.variables:
            for neighbor in markov_net.get_neighbors(var):
                if var < neighbor:
                    edge = (var, neighbor)
                    self.set_edge_factor(edge, markov_net.get_potential(edge))
                    indicator_features = np.zeros(num_edges)
                    indicator_features[i] = 1.0
                    self.set_edge_features(edge, indicator_features)
                    i += 1

        self.create_matrices()

        self.unary_feature_mat = csr_matrix(self.unary_feature_mat)
        self.edge_feature_mat = csr_matrix(self.edge_feature_mat)

        # load current unary potentials into unary_weight_mat
        for (var, i) in self.var_index.items():
            self.unary_weight_mat[i, :] = -np.inf
            potential = self.unary_potentials[var]
            self.unary_weight_mat[i, :len(potential)] = potential

        # load current edge potentials into edge_weight_mat
        for (edge, i) in self.message_index.items():
            padded_potential = -np.inf * np.ones((self.max_states, self.max_states))
            potential = self.get_potential(edge)
            padded_potential[:self.num_states[edge[0]], :self.num_states[edge[1]]] = potential
            self.edge_weight_mat[i, :] = padded_potential.ravel()

    def load_factors_from_matrices(self):
        """
        Load dictionary-based factors from current matrices. Since learning is often done by updating the matrix
        form, it's important to call this method before using the dictionary-based views of the potentials. This
        method will also properly undo the padding that makes all potential vectors and matrices the same shape (with 
        zero-probability states to pad the smaller cardinality variables).
        
        :return: None
        """
        self.update_unary_matrix()
        self.update_edge_tensor()

        for (var, i) in self.var_index.items():
            self.set_unary_factor(var, self.unary_mat[:self.num_states[var], i].ravel())

        for edge, i in self.message_index.items():
            self.set_edge_factor(edge,
                                 self.edge_pot_tensor[:self.num_states[edge[1]], :self.num_states[edge[0]],
                                 i].squeeze().T)
