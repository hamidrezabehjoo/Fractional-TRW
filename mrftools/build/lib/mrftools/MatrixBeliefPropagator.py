"""BeliefPropagator class."""
import numpy as np

from .Inference import Inference


class MatrixBeliefPropagator(Inference):
    """
    Object that can run belief propagation on a MarkovNet. Uses sparse matrices to encode the
    indexing underlying belief propagation. 
    """

    def __init__(self, markov_net):
        """
        Initialize belief propagator for markov_net.
        
        :param markov_net: Markov net
        :type markov_net: MarkovNet object encoding the probability distribution
        """
        self.mn = markov_net
        self.var_beliefs = dict()
        self.pair_beliefs = dict()

        # if the MarkovNet has not created its matrix representation of structure, create it
        if not self.mn.matrix_mode:
            self.mn.create_matrices()

        self.previously_initialized = False
        self.message_mat = None
        self.initialize_messages()

        self.belief_mat = np.zeros((self.mn.max_states, len(self.mn.variables)))
        self.pair_belief_tensor = np.zeros((self.mn.max_states, self.mn.max_states, self.mn.num_edges))

        self.max_iter = 500  # default maximum iterations

        # the augmented_mat is used to condition variables or for loss-augmented inference for max-margin learning
        self.augmented_mat = np.zeros((self.mn.max_states, len(self.mn.variables)))
        self.fully_conditioned = False  # true if every variable has been conditioned

        # conditioned stores the indices of variables that have been conditioned, initialized to all False
        self.conditioned = np.zeros(len(self.mn.variables), dtype=bool)

        # condition variables so they can't be in states greater than their cardinality
        self.disallow_impossible_states()

    def set_max_iter(self, max_iter):
        """
        Set the maximum iterations of belief propagation to run before early stopping
        :param max_iter: integer maximum iterations
        :return: None
        """
        self.max_iter = max_iter

    def initialize_messages(self):
        """
        Initialize messages to default initialization (set to zeros).

        :return: None
        """
        self.message_mat = np.zeros((self.mn.max_states, 2 * self.mn.num_edges))

    def augment_loss(self, var, state):
        """
        Adds a loss penalty to the MRF energy function. Used to create loss-augmented inference for max-margin learning
        
        :param var: variable to add loss to
        :type var: object
        :param state: state of variable in ground truth labels
        :type state: int
        :return: None
        """
        i = self.mn.var_index[var]
        self.augmented_mat[:, i] = 1
        self.augmented_mat[state, i] = 0

    def condition(self, var, state):
        """
        Condition a variable, usually because of statistical evidence, to be in a subset of its possible states
        
        :param var: variable to condition
        :type var: object
        :param state: state to condition to or array of states that the variable may be in
        :type state: int or array
        :return: None
        """
        i = self.mn.var_index[var]
        self.augmented_mat[:, i] = -np.inf
        self.augmented_mat[state, i] = 0
        if isinstance(state, int):
            # only if the variable is fully conditioned to be in a single state, mark that the variable is conditioned
            self.conditioned[i] = True

        if np.all(self.conditioned):
            # compute beliefs and set flag to never recompute them
            self.compute_beliefs()
            self.compute_pairwise_beliefs()
            self.fully_conditioned = True

    def disallow_impossible_states(self):
        """
        Force variables to only allow nonzero probability on their possible states.

        :return: None
        """
        for var, num_states in self.mn.num_states.items():
            self.condition(var, range(num_states))

    def compute_beliefs(self):
        """
        Compute unary log beliefs based on current messages and store them in belief_mat
        
        :return: None
        """
        if not self.fully_conditioned:
            self.belief_mat = self.mn.unary_mat + self.augmented_mat
            self.belief_mat += sparse_dot(self.message_mat, self.mn.message_to_map)

            self.belief_mat -= logsumexp(self.belief_mat, 0)

    def compute_pairwise_beliefs(self):
        """
        Compute pairwise log beliefs based on current messages, and stores them in pair_belief_tensor

        :return: None
        """
        if not self.fully_conditioned:
            adjusted_message_prod = self.belief_mat[:, self.mn.message_from] \
                                    - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                 self.message_mat[:, :self.mn.num_edges]))

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape(
                (self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape(
                (1, self.mn.max_states, self.mn.num_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs

    def update_messages(self):
        """
        Update all messages between variables and store them in message_mat 

        :return: the float change in messages from previous iteration.
        """
        self.compute_beliefs()

        # Using the beliefs as the sum of all incoming log messages, subtract the outgoing messages and add the edge
        # potential.
        adjusted_message_prod = self.mn.edge_pot_tensor + self.belief_mat[:, self.mn.message_from] \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                             self.message_mat[:, :self.mn.num_edges]))

        messages = np.squeeze(logsumexp(adjusted_message_prod, 1))
        messages = np.nan_to_num(messages - messages.max(0))

        with np.errstate(over='ignore'):
            change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

    def _compute_inconsistency_vector(self):
        """
        Compute the vector of inconsistencies between unary beliefs and pairwise beliefs
        :return: Vector of inconsistencies
        :rtype: array
        """
        expanded_beliefs = np.exp(self.belief_mat[:, self.mn.message_to])

        pairwise_beliefs = np.hstack((np.sum(np.exp(self.pair_belief_tensor), axis=0),
                                      np.sum(np.exp(self.pair_belief_tensor), axis=1)))

        return expanded_beliefs - pairwise_beliefs

    def compute_inconsistency(self):
        """
        Return the total disagreement between each unary belief and its pairwise beliefs. 
        When message passing converges, the inconsistency should be within numerical error.

        :return: the total absolute disagreement between each unary belief and its pairwise beliefs
        """
        disagreement = np.sum(np.abs(self._compute_inconsistency_vector()))

        return disagreement

    def infer(self, tolerance=1e-8, display='iter'):
        """
        Run belief propagation until messages change less than tolerance.

        :param tolerance: the minimum amount that the messages can change while message passing can be considered not
                            converged
        :param display: string parameter indicating how much to display. Options are 'full' and 'iter'
                        'full' prints the energy functional and dual objective each iteration,
                                which requires extra computation
                        'iter' prints just the change in messages each iteration
        :return: None
        """
        change = np.inf
        iteration = 0
        while change > tolerance and iteration < self.max_iter:
            change = self.update_messages()
            if display == "full":
                energy_func = self.compute_energy_functional()
                disagreement = self.compute_inconsistency()
                dual_obj = self.compute_dual_objective()
                print("Iteration %d, change in messages %f. Calibration disagreement: %f, "
                      "energy functional: %f, dual obj: %f" % (iteration, change, disagreement, energy_func, dual_obj))
            elif display == "iter":
                print("Iteration %d, change in messages %f." % (iteration, change))
            iteration += 1
        if display == 'final' or display == 'full' or display == 'iter':
            print("Belief propagation finished in %d iterations." % iteration)

    def load_beliefs(self):
        """
        Update the belief dictionaries var_beliefs and pair_beliefs using the current messages.

        :return: None
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        for (var, i) in self.mn.var_index.items():
            self.var_beliefs[var] = self.belief_mat[:len(self.mn.unary_potentials[var]), i]

        for edge, i in self.mn.message_index.items():
            (var, neighbor) = edge

            belief = self.pair_belief_tensor[:len(self.mn.unary_potentials[var]),
                     :len(self.mn.unary_potentials[neighbor]), i]

            self.pair_beliefs[(var, neighbor)] = belief

            self.pair_beliefs[(neighbor, var)] = belief.T

    def compute_bethe_entropy(self):
        """
        Compute Bethe entropy from current beliefs. 
        This method assumes that the beliefs have been computed and are fresh.
        
        :return: computed Bethe entropy
        """
        if self.fully_conditioned:
            entropy = 0
        else:
            entropy = - np.sum(np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor)) \
                      - np.sum((1 - self.mn.degrees) * (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))

        return entropy

    def compute_energy(self):
        """
        Compute the log-linear energy. Assume that the beliefs have been computed and are fresh.

        :return: computed energy
        """
        energy = np.sum(
            np.nan_to_num(self.mn.edge_pot_tensor[:, :, self.mn.num_edges:]) * np.exp(self.pair_belief_tensor)) + \
                 np.sum(np.nan_to_num(self.mn.unary_mat) * np.exp(self.belief_mat))

        return energy

    def get_feature_expectations(self):
        """
        Computes the feature expectations under the currently estimated marginal probabilities. Only works when the 
        model is a LogLinearModel class with features for edges. 

        :return: vector of the marginals in order of the flattened unary features first, then the flattened pairwise 
                    features
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        summed_features = self.mn.unary_feature_mat.dot(np.exp(self.belief_mat).T)

        summed_pair_features = self.mn.edge_feature_mat.dot(np.exp(self.pair_belief_tensor).reshape(
            (self.mn.max_states ** 2, self.mn.num_edges)).T)

        marginals = np.append(summed_features.reshape(-1), summed_pair_features.reshape(-1))

        return marginals

    def compute_energy_functional(self):
        """
        Compute the energy functional, which is the variational approximation of the log-partition function.

        :return: computed energy functional
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()
        return self.compute_energy() + self.compute_bethe_entropy()

    def compute_dual_objective(self):
        """
        Compute the value of the BP Lagrangian.

        :return: Lagrangian objective function
        """
        objective = self.compute_energy_functional() + \
                    np.sum(self.message_mat * self._compute_inconsistency_vector())

        return objective

    def set_messages(self, messages):
        """
        Set the message vector. Useful for warm-starting inference if a previously computed message matrix is available.
        
        :param messages: message matrix
        :type messages: ndarray
        :return: None
        """
        assert (np.all(self.message_mat.shape == messages.shape))
        self.message_mat = messages


def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.
    
    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of np.log(np.sum(np.exp(matrix), dim)))
    :rtype: ndarray
    """
    try:
        with np.errstate(over='raise', under='raise', divide='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val


def sparse_dot(full_matrix, sparse_matrix):
    """
    Convenience function to compute the dot product of a full matrix and a sparse matrix. 
    Useful to avoid writing code with a lot of transposes.
    
    :param full_matrix: dense matrix
    :type full_matrix: ndarray
    :param sparse_matrix: sparse matrix
    :type sparse_matrix: csc_matrix
    :return: full_matrix.dot(sparse_matrix)
    :rtype: ndarray
    """
    return sparse_matrix.T.dot(full_matrix.T).T
