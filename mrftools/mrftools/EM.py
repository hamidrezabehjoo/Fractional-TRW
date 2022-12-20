"""EM learner class."""
from .Learner import Learner
from .opt import *

class EM(Learner):
    """
    Objects that perform expectation maximization for learning with latent variables.
    """
    def __init__(self, inference_type):
        super(EM, self).__init__(inference_type)

    def learn(self, weights, optimizer=ada_grad, callback=None, opt_args=None):
        """
        Fit model parameters by alternating inference of latent variables and learning the best parameters
        to fit all variables. This method implements the variational expectation-maximization concept.
        
        :param weights: Initial weight vector. Can be used to warm start from a previous solution.
        :param optimizer: gradient-based optimization function, as defined in opt.py
        :param callback: callback function run during each iteration of the optimizer. The function receives the 
                        weights as input. Can be useful for diagnostics, live plotting, storing records, etc.
        :param opt_args: optimization arguments. Usually a dictionary of parameter values
        :return: learned weights
        """
        old_weights = np.inf
        new_weights = weights
        self.start_time = time.time()
        while not np.allclose(old_weights, new_weights, rtol=1e-4, atol=1e-5):
            old_weights = new_weights
            self.e_step(new_weights)
            new_weights = self.m_step(new_weights, optimizer, callback, opt_args)

        return new_weights

    def e_step(self, weights):
        self.label_expectations = self.calculate_expectations(weights, self.conditioned_belief_propagators, True)

    def m_step(self, weights, optimizer=ada_grad, callback=None, opt_args=None):
        res = optimizer(self.objective, self.gradient, weights, args=opt_args, callback=callback)
        return res
