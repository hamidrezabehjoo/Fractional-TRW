"""
Paired dual learner class
"""
from .Learner import Learner
from .opt import *


class PairedDual(Learner):
    """
    Objects that learn with paired inner dual optimization.
    """
    def __init__(self, inference_type, bp_iter=2, warm_up=5):
        """
        Initialize paired dual learner object
        
        :param inference_type: Inference class used to estimate feature expectations
        :param bp_iter: maximum number of iterations each inference is allowed to run before performing a 
                        weight-learning step
        :param warm_up: number of inference iterations to run before starting learning
        """
        super(PairedDual, self).__init__(inference_type)
        self.bp_iter = bp_iter
        self.warm_up = warm_up

    def learn(self, weights, optimizer=ada_grad, callback=None, opt_args=None):
        """
        Fit model parameters my jointly solving the full dual saddle-point objective that includes optimization over
        estimated expectations of output variables and latent variables as well as weight optimization. 
        
        :param weights: Initial weight vector. Can be used to warm start from a previous solution.
        :param optimizer: gradient-based optimization function, as defined in opt.py
        :param callback: callback function run during each iteration of the optimizer. The function receives the 
                        weights as input. Can be useful for diagnostics, live plotting, storing records, etc.
        :param opt_args: optimization arguments. Usually a dictionary of parameter values
        :return: learned weights
        """
        for bp in self.belief_propagators + self.conditioned_belief_propagators:
            bp.set_max_iter(self.bp_iter)
            for i in range(self.warm_up):
                bp.update_messages()

        self.start_time = time.time()
        new_weights = optimizer(self.dual_obj, self.subgrad_grad, weights, args=opt_args, callback=callback)

        return new_weights
