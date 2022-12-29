"""Class to run max-product belief propagation."""
import numpy as np

from .MatrixBeliefPropagator import MatrixBeliefPropagator, sparse_dot, logsumexp


class MaxProductBeliefPropagator(MatrixBeliefPropagator):
    """
    Class to run inference of the most likely state via max-product belief propagation.
    """
    def __init__(self, markov_net):
        """
        Initialize a max-product belief propagator. 
        
        :param markov_net: MarkovNet object encoding the probability distribution
        :type markov_net: MarkovNet
        """
        super(MaxProductBeliefPropagator, self).__init__(markov_net)

    def compute_beliefs(self):
        if not self.fully_conditioned:
            max_marginals = self.mn.unary_mat + self.augmented_mat
            max_marginals += sparse_dot(self.message_mat, self.mn.message_to_map)

            states = max_marginals.argmax(0)
            self.belief_mat = -np.inf * np.ones(max_marginals.shape)
            self.belief_mat[states, range(self.belief_mat.shape[1])] = 0

    def compute_pairwise_beliefs(self):
        if not self.fully_conditioned:
            adjusted_message_prod = self.belief_mat[:, self.mn.message_from] \
                                    - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                 self.message_mat[:, :self.mn.num_edges]))

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape(
                (self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape(
                (1, self.mn.max_states, self.mn.num_edges))

            max_marginals = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] + to_messages + from_messages

            self.pair_belief_tensor = np.where(max_marginals == max_marginals.max((0, 1)), 0, -np.inf)

    def update_messages(self):
        belief_mat = self.mn.unary_mat + self.augmented_mat
        belief_mat += sparse_dot(self.message_mat, self.mn.message_to_map)

        belief_mat -= logsumexp(belief_mat, 0)

        adjusted_message_prod = self.mn.edge_pot_tensor - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                                     self.message_mat[:, :self.mn.num_edges]))
        adjusted_message_prod += belief_mat[:, self.mn.message_from]

        messages = np.squeeze(adjusted_message_prod.max(1))
        messages = np.nan_to_num(messages - messages.max(0))

        with np.errstate(over='ignore'):
            change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change
