"""BeliefPropagator class."""
import numpy as np

from .Inference import Inference


class BeliefPropagator(Inference):
    """
    Object that can run belief propagation on a MarkovNet. Uses native python loops to compute message passing, so
    it can be slow. This class is mostly useful for reference, since the loop-based implementations are similar
    to how belief propagation is typically written in math.
    """

    def __init__(self, markov_net):
        """
        Initialize belief propagator for markov_net.
        
        :param markov_net: MarkovNet object encoding the probability distribution
        """
        self.mn = markov_net
        self.var_beliefs = dict()
        self.pair_beliefs = dict()
        self.messages = dict()
        self.init_messages()
        self.init_beliefs()
        self.max_iter = 500

    def set_max_iter(self, max_iter):
        """
        Set the maximum iterations of belief propagation to run before early stopping
        :param max_iter: integer maximum iterations
        :return: None
        """
        self.max_iter = max_iter

    def init_messages(self):
        """
        Initialize messages to default initialization (set to zeros).
        
        :return: None
        """
        for var in self.mn.variables:
            for neighbor in self.mn.get_neighbors(var):
                self.messages[(var, neighbor)] = np.zeros(self.mn.num_states[neighbor])

    def init_beliefs(self):
        """
        Initialize beliefs to be the normalized potential functions. 
        These beliefs will not be consistent until the are updated with message passing.
        :return: None
        """
        for var in self.mn.variables:
            belief = self.mn.unary_potentials[var]
            log_z = logsumexp(belief)
            belief = belief - log_z
            self.var_beliefs[var] = belief

        # Initialize pairwise beliefs

        for var in self.mn.variables:
            for neighbor in self.mn.get_neighbors(var):
                belief = self.mn.get_potential((var, neighbor))
                log_z = logsumexp(np.sum(belief))
                belief = belief - log_z
                self.pair_beliefs[(var, neighbor)] = belief

    def compute_beliefs(self):
        """
        Compute unary log beliefs based on current messages and store them in var_beliefs dict
        
        :return: None
        """
        for var in self.mn.variables:
            belief = self.mn.unary_potentials[var]
            for neighbor in self.mn.get_neighbors(var):
                belief = belief + self.messages[(neighbor, var)]
            log_z = logsumexp(belief)
            belief = belief - log_z
            self.var_beliefs[var] = belief

    def compute_pairwise_beliefs(self):
        """
        Compute pairwise log beliefs based on current messages, and stores them in pair_beliefs dict
        
        :return: None
        """
        for var in self.mn.variables:
            for neighbor in self.mn.get_neighbors(var):
                if var < neighbor:
                    belief = self.mn.get_potential((var, neighbor))

                    # compute product of all messages to var except from neighbor
                    var_message_product = self.var_beliefs[var] - self.messages[(neighbor, var)]
                    belief = (belief.T + var_message_product).T

                    # compute product of all messages to neighbor except from var
                    neighbor_message_product = self.var_beliefs[neighbor] - self.messages[(var, neighbor)]
                    belief = belief + neighbor_message_product

                    log_z = logsumexp(belief)
                    belief = belief - log_z
                    self.pair_beliefs[(var, neighbor)] = belief
                    self.pair_beliefs[(neighbor, var)] = belief.T

    def compute_message(self, var, neighbor):
        """
        Compute the message from var to factor.
        
        :param var: variable sending the message
        :param neighbor: neighbor receiving the message
        :return: message vector from var to neighbor
        """
        # compute the product of all messages coming into var except the one from neighbor
        adjusted_message_product = self.var_beliefs[var] - self.messages[(neighbor, var)]

        # partial log-sum-exp operation
        matrix = self.mn.get_potential((neighbor, var)) + adjusted_message_product
        # the dot product with ones is slightly faster than calling sum
        message = np.log(np.exp(matrix - matrix.max()).dot(np.ones(matrix.shape[1])))

        # pseudo-normalize message
        message -= np.max(message)

        return message

    def update_messages(self):
        """
        Update all messages between variables using belief division. 
        
        :return: the float change in messages from previous iteration.
        """
        change = 0.0
        self.compute_beliefs()
        new_messages = dict()
        for var in self.mn.variables:
            for neighbor in self.mn.get_neighbors(var):
                new_messages[(var, neighbor)] = self.compute_message(var, neighbor)
                change += np.sum(np.abs(new_messages[(var, neighbor)] - self.messages[(var, neighbor)]))
        self.messages = new_messages

        return change

    def compute_inconsistency(self):
        """
        Return the total disagreement between each unary belief and its pairwise beliefs. 
        When message passing converges, the inconsistency should be within numerical error.

        :return: the total absolute disagreement between each unary belief and its pairwise beliefs
        """
        disagreement = 0.0
        self.compute_beliefs()
        self.compute_pairwise_beliefs()
        for var in self.mn.variables:
            unary_belief = np.exp(self.var_beliefs[var])
            for neighbor in self.mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(self.pair_beliefs[(var, neighbor)]), 1)
                disagreement += np.sum(np.abs(unary_belief - pair_belief))
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
                disagreement = self.compute_inconsistency()
                energy_func = self.compute_energy_functional()
                dual_obj = self.compute_dual_objective()
                print("Iteration %d, change in messages %f. Calibration disagreement: %f, energy functional: %f, "
                      "dual obj: %f" % (iteration, change, disagreement, energy_func, dual_obj))
            elif display == "iter":
                print("Iteration %d, change in messages %f." % (iteration, change))
            iteration += 1
        if display == 'final' or display == 'full' or display == 'iter':
            print("Belief propagation finished in %d iterations." % iteration)

    def compute_bethe_entropy(self):
        """
        Compute Bethe entropy from current beliefs. 
        This method assumes that the beliefs have been computed and are fresh.
        
        :return: computed Bethe entropy
        """
        entropy = 0.0

        for var in self.mn.variables:
            neighbors = self.mn.get_neighbors(var)
            entropy -= (1 - len(neighbors)) * np.sum(
                np.exp(self.var_beliefs[var]) * np.nan_to_num(self.var_beliefs[var]))
            for neighbor in neighbors:
                if var < neighbor:
                    entropy -= np.sum(
                        np.exp(self.pair_beliefs[(var, neighbor)]) * np.nan_to_num(self.pair_beliefs[(var, neighbor)]))
        return entropy

    def compute_energy(self):
        """
        Compute the log-linear energy. Assume that the beliefs have been computed and are fresh.
        
        :return: computed energy
        """
        energy = 0.0

        for var in self.mn.variables:
            neighbors = self.mn.get_neighbors(var)
            energy += np.nan_to_num(self.mn.unary_potentials[var]).dot(np.exp(self.var_beliefs[var]))
            for neighbor in neighbors:
                if var < neighbor:
                    energy += np.sum(np.nan_to_num(
                        self.mn.get_potential((var, neighbor)) * np.exp(self.pair_beliefs[(var, neighbor)])))
        return energy

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
        objective = self.compute_energy_functional()
        for var in self.mn.variables:
            unary_belief = np.exp(self.var_beliefs[var])
            for neighbor in self.mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(self.pair_beliefs[(var, neighbor)]), 1)
                objective += self.messages[(neighbor, var)].dot(unary_belief - pair_belief)
        return objective

    def get_feature_expectations(self):
        """
        Computes the feature expectations under the currently estimated marginal probabilities.
        
        :return: vector of the marginals in order of the flattened unary features first, then the flattened pairwise 
                    features
        """
        self.infer(display='off')
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        # make vector form of marginals
        marginals = []
        for j in range(len(self.potentials)):
            if isinstance(self.potentials[j], tuple):
                # get pairwise belief
                table = np.exp(self.pair_beliefs[self.potentials[j]])
            else:
                # get unary belief and multiply by features
                var = self.potentials[j]
                table = np.outer(np.exp(self.var_beliefs[var]), self.mn.unaryFeatures[var])

            # flatten table and append
            marginals.extend(table.reshape((-1, 1)).tolist())
        return np.array(marginals)

    def load_beliefs(self):
        """
        Update the belief dictionaries var_beliefs and pair_beliefs using the current messages.
        
        :return: None
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()


def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.
    """
    max_val = matrix.max()
    return np.log(np.sum(np.exp(matrix - max_val), dim)) + max_val
