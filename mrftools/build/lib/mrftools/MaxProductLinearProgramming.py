"""Class to run max-product linear programming for linear-programming MAP inference."""
import numpy as np

from .MatrixBeliefPropagator import sparse_dot
from .MaxProductBeliefPropagator import MaxProductBeliefPropagator


class MaxProductLinearProgramming(MaxProductBeliefPropagator):
    """
    Class to run max-product linear programming for linear-programming MAP inference.
    """
    def __init__(self, markov_net):
        """
        Initialize a max-product linear programming inference object. 
        
        :param markov_net: MarkovNet object encoding the probability distribution
        :type markov_net: MarkovNet
        """
        super(MaxProductLinearProgramming, self).__init__(markov_net)

    def update_messages(self):
        message_sum = sparse_dot(self.message_mat, self.mn.message_to_map)

        max_marginals = self.mn.unary_mat + self.augmented_mat
        max_marginals += message_sum

        adjusted_message_prod = self.mn.edge_pot_tensor - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                                     self.message_mat[:, :self.mn.num_edges]))
        adjusted_message_prod += max_marginals[:, self.mn.message_from]

        incoming_messages = np.squeeze(adjusted_message_prod.max(1))

        outgoing_messages = message_sum[:, self.mn.message_to] - self.message_mat
        messages = 0.5 * np.nan_to_num(incoming_messages - np.nan_to_num(outgoing_messages))

        messages = np.nan_to_num(messages - messages.max(0))

        with np.errstate(over='ignore'):
            change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change
