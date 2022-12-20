"""Primal dual weight learning class"""
from .PairedDual import PairedDual
from .opt import *


class PrimalDual(PairedDual):
    """
    Objects that learn with inner dual optimization interleaved with full inference of latent variables
    """
    def __init__(self, inference_type, bp_iter=300, dual_bp_iter=1):
        """
        Initialize primal dual learner.
        
        :param inference_type: Inference class used to estimate feature expectations
        :param bp_iter: maximum number of iterations the latent-variable inference is allowed to run before exiting
        :param dual_bp_iter: maximum number of iterations the main inferences are allowed to run before performing a 
                            weight-learning step
        """
        super(PrimalDual, self).__init__(inference_type)
        self.bp_iter = bp_iter
        self.dual_bp_iter = dual_bp_iter

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
        for bp in self.conditioned_belief_propagators:
            bp.set_max_iter(self.bp_iter)

        for bp in self.belief_propagators:
            bp.set_max_iter(self.dual_bp_iter)

        self.start_time = time.time()
        res = optimizer(self.dual_obj, self.subgrad_grad, weights, args=opt_args, callback=callback)
        new_weights = res
        return new_weights
