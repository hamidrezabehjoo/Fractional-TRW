"""Class to do tree-reweighted belief propagation with matrix-based computation."""
import numpy as np

from .MatrixBeliefPropagator import MatrixBeliefPropagator, logsumexp, sparse_dot


class MatrixTRBeliefPropagator(MatrixBeliefPropagator):
    """
    Class to perform tree-reweighted belief propagation. 
    """
    def __init__(self, markov_net, tree_probabilities=None):
        """
        Initialize a TRBP object with a Markov net and a dictionary of tree probabilities
        
        :param markov_net: Markov net to perform inference on.
        :type markov_net: MarkovNet
        :param tree_probabilities: Edge appearance probabilities for spanning forest distribution. If this parameter is 
                                    not provided, this class assumes there are tree probabilities stored in the Markov
                                    net object. The probabilities should be provided as a dict with a key-value pair
                                    for each edge.
        :type tree_probabilities: dict
        """
        super(MatrixTRBeliefPropagator, self).__init__(markov_net)

        if tree_probabilities:
            self._set_tree_probabilities(tree_probabilities)
        else:
            self._set_tree_probabilities(markov_net.tree_probabilities)

    def _set_tree_probabilities(self, tree_probabilities):
        """
        Store the provided tree probabilities for later lookup as an array in order of the MarkovNet's internal edge 
        storage
        :param tree_probabilities: dict containing tree probabilities for all edges
        :type tree_probabilities: dict
        :return: 
        :rtype: 
        """
        self.tree_probabilities = np.zeros(2 * self.mn.num_edges)

        for edge, i in self.mn.message_index.items():
            reversed_edge = edge[::-1]
            if edge in tree_probabilities:
                self.tree_probabilities[i] = tree_probabilities[edge]
                self.tree_probabilities[i + self.mn.num_edges] = tree_probabilities[edge]
            elif reversed_edge in tree_probabilities:
                self.tree_probabilities[i] = tree_probabilities[reversed_edge]
                self.tree_probabilities[i + self.mn.num_edges] = tree_probabilities[reversed_edge]
            else:
                raise KeyError('Edge %s was not assigned a probability.' % repr(edge))

        self.expected_degrees = sparse_dot(self.tree_probabilities.T, self.mn.message_to_map).T

    def compute_bethe_entropy(self):
        if self.fully_conditioned:
            entropy = 0
        else:
            entropy = - np.sum(self.tree_probabilities[:self.mn.num_edges] *
                               np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor)) \
                      - np.sum((1 - self.expected_degrees) * (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))

        return entropy

    def update_messages(self):
        self.compute_beliefs()

        adjusted_message_prod = self.belief_mat[:, self.mn.message_from] \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                             self.message_mat[:, :self.mn.num_edges]))

        messages = np.squeeze(logsumexp(self.mn.edge_pot_tensor / self.tree_probabilities + adjusted_message_prod, 1))
        messages = np.nan_to_num(messages - messages.max(0))

        with np.errstate(over='ignore'):
            change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

    def compute_beliefs(self):
        if not self.fully_conditioned:
            self.belief_mat = self.mn.unary_mat + self.augmented_mat
            self.belief_mat += sparse_dot(self.message_mat * self.tree_probabilities, self.mn.message_to_map)

            log_z = logsumexp(self.belief_mat, 0)

            self.belief_mat = self.belief_mat - log_z

    def compute_pairwise_beliefs(self):
        if not self.fully_conditioned:
            adjusted_message_prod = self.belief_mat[:, self.mn.message_from] \
                                    - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                 self.message_mat[:, :self.mn.num_edges]))

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape(
                (self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape(
                (1, self.mn.max_states, self.mn.num_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] / self.tree_probabilities[self.mn.num_edges:] \
                      + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs
