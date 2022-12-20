"""Main learner class for log-linear model parameter learning. """
import copy

from .ConvexBeliefPropagator import ConvexBeliefPropagator
from .opt import *
from .MatrixBeliefPropagator import MatrixBeliefPropagator


class Learner(object):
    """
    Learner class for log-lienar model parameter learning. This class contains methods for calculating various 
    objective functions and gradients, and implements a subgradient optimization for the variational likelihood.
    """
    def __init__(self, inference_type=MatrixBeliefPropagator):
        """
        Initialize a learner by setting the inference method for computing variational likelihood approximations.
        
        :param inference_type: Inference class for computing feature expectations used in variational likelihood
        """
        self.label_expectations = None
        self.inferred_expectations = None
        self.inference_type = inference_type
        self.num_examples = 0
        self.models = []
        self.conditioned_models = []
        self.conditioned_belief_propagators = []
        self.belief_propagators = []
        self.l1_regularization = 0.00
        self.l2_regularization = 1
        self.weight_dim = None
        self.fully_observed = True
        self.initialization_flag = False
        self.loss_augmented = False
        self.inference_instantiator = None
        self.start_time = 0
        self.max_time = np.inf
        self.display = 'off'

    def set_regularization(self, l1, l2):
        """
        Set the regularization parameters.
        
        :param l1: l1 regularization parameter
        :param l2: l2 regularization parameter
        :return: 
        """
        self.l1_regularization = l1
        self.l2_regularization = l2

    def add_data(self, labels, model):
        """
        Add data example to training set. The states variable should be a dictionary containing all the states of the
         unary variables. 
        
        :param labels: dict containing true states of all labeled variables
        :param model: LogLinearModel object containing features for each pairwise and unary potential
        :return: 
        """
        self.models.append(model)
        if self.inference_instantiator:
            # if a custom inference instantiation function is provided, use that instead of default constructor
            bp = self.inference_instantiator(model)
        else:
            bp = self.inference_type(model)

        if self.loss_augmented:
            # if we are using augmented loss for max-margin learning, add loss-augmented potentials to inference
            for (var, state) in labels.items():
                bp.augment_loss(var, state)

        self.belief_propagators.append(bp)

        if self.weight_dim is None:
            self.weight_dim = model.weight_dim
        else:
            assert self.weight_dim == model.weight_dim, "Parameter dimensionality did not match"

        # create inference objects to extract the feature expectations based on either (1) reading the variable states
        # or (2) inferring the latent variables then reading the inferred expectations
        self.conditioned_models.append(model)

        if self.inference_instantiator:
            conditioned_bp = self.inference_instantiator(model)
        else:
            conditioned_bp = self.inference_type(model)

        for (var, state) in labels.items():
            conditioned_bp.condition(var, state)

        for var in model.variables:
            if var not in labels.keys():
                self.fully_observed = False

        self.conditioned_belief_propagators.append(conditioned_bp)

        self.num_examples += 1

    def _set_initialize_every_iter(self, flag):
        """
        Force learner to reinitialize inference objects before each objetive and gradient computation instead of warm
        starting from the previous iteration. Doing so will make learning slower, but may protect against getting
        stuck in a local optimum.
        
        :param flag: Boolean value of whether to initialize
        :return: None:
        """
        self.initialization_flag = flag

    def do_inference(self, belief_propagators):
        """
        Perform inference on all stored models.
        
        :param belief_propagators: iterable of inference objects
        :return: None
        """
        for bp in belief_propagators:
            if self.initialization_flag:
                bp.initialize_messages()
            bp.infer(display=self.display)

    def set_inference_truncation(self, bp_iter):
        """
        Set maximum number of iterations for inference. Useful for faster learning or inner-dual learning to stop
        inference before they run to convergence. 
        :param bp_iter: maximum iterations each belief propagator can run for each gradient/objective computation
        :return: NOne
        """
        for bp in self.belief_propagators + self.conditioned_belief_propagators:
            bp.set_max_iter(bp_iter)

    def get_feature_expectations(self, belief_propagators):
        """
        Run inference and return the marginal in vector form using the order of self.potentials.
        
        :param belief_propagators: iterable of inference objects to use to get feature expectations
        :return: vector of feature expectations
        """
        marginal_sum = 0
        for bp in belief_propagators:
            marginal_sum += np.true_divide(bp.get_feature_expectations(), len(bp.mn.variables))

        return marginal_sum / len(belief_propagators)

    def get_bethe_entropy(self, belief_propagators):
        """
        Compute the average Bethe entropy of all inference objects
        :param belief_propagators: iterable of inference objects
        :return: average Bethe entropy of all objectives
        """
        bethe = 0
        for bp in belief_propagators:
            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()
            bethe += bp.compute_bethe_entropy()

        bethe = bethe / self.num_examples
        return bethe

    def subgrad_obj(self, weights, options=None, do_inference=True):
        """
        Compute the variational negative log likelihood. Performs inference on latent variables in the labeled 
        inference objects before calling the EM objective
        
        :param weights: Weight vector containing the same number of entries as all weights for this model
        :param do_inference: Boolean value indicating whether or not to run inference. Defaults to True.
        :return: objective value (float)
        """
        if self.label_expectations is None or not self.fully_observed:
            self.label_expectations = self.calculate_expectations(weights, self.conditioned_belief_propagators,
                                                                  do_inference)
        return self.objective(weights)

    def subgrad_grad(self, weights, options=None, do_inference=False):
        """
        Compute the gradient of the variational negative log likelihood. 
        
        :param weights: Weight vector containing the same number of entries as all weights for this model
        :param do_inference: Boolean value indicating whether or not to run inference. Defaults to False because 
                            typically the objective function was called immediately before, which does inference.
        :return: gradient with respect to weights
        """
        if self.label_expectations is None or not self.fully_observed:
            self.label_expectations = self.calculate_expectations(weights, self.conditioned_belief_propagators,
                                                                  do_inference)
        return self.gradient(weights)

    def learn(self, weights, optimizer=ada_grad, callback=None, opt_args=None):
        """
        Fit model parameters my maximizing the variational likelihood
        :param weights: Initial weight vector. Can be used to warm start from a previous solution.
        :param optimizer: gradient-based optimization function, as defined in opt.py
        :param callback: callback function run during each iteration of the optimizer. The function receives the 
                        weights as input. Can be useful for diagnostics, live plotting, storing records, etc.
        :param opt_args: optimization arguments. Usually a dictionary of parameter values
        :return: learned weights
        """
        self.start_time = time.time()
        res = optimizer(self.subgrad_obj, self.subgrad_grad, weights, opt_args, callback=callback)
        new_weights = res

        return new_weights

    def set_weights(self, weight_vector, belief_propagators):
        """
        Set weights of Markov net from vector using the order in self.potentials.
        :param weight_vector: weight vector containing weights for all potentials
        :param belief_propagators: iterable of belief propagators whose models should be updated with the weights
        :return: None
        """
        for bp in belief_propagators:
            bp.mn.set_weights(weight_vector)

    def calculate_expectations(self, weights, belief_propagators, should_infer=True):
        """
        Calculate the feature expectations given the provided model weights.
        
        :param weights: weight vector containing weights for all potentials
        :param belief_propagators: iterable of belief propagators whose models should be updated with the weights
        :param should_infer: Boolean value of whether to run inference. This value should usually only be False when
                            inference has already been run for this particular weight vector, i.e., if this function
                            is being called immediately after it has been called with the same weights.
        :return: feature expectation vector
        """
        self.set_weights(weights, belief_propagators)
        if should_infer:
            self.do_inference(belief_propagators)

        return self.get_feature_expectations(belief_propagators)

    def objective(self, weights, options=None):
        """
        Return the primal regularized negative variational log likelihood 
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: objective value
        """
        self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, True)

        term_p = sum([np.true_divide(x.compute_energy_functional(), len(x.mn.variables)) for x in
                      self.belief_propagators]) / len(self.belief_propagators)

        if not self.fully_observed:
            # recompute energy functional for label distributions only in latent variable case
            self.set_weights(weights, self.conditioned_belief_propagators)
            term_q = sum([np.true_divide(x.compute_energy_functional(), len(x.mn.variables)) for x in
                          self.conditioned_belief_propagators]) / len(self.conditioned_belief_propagators)
        else:
            term_q = np.dot(self.label_expectations, weights)

        self.term_q_p = term_p - term_q

        objective = 0.0
        # add regularization penalties
        objective += self.l1_regularization * np.sum(np.abs(weights))
        objective += 0.5 * self.l2_regularization * weights.dot(weights)
        objective += self.term_q_p

        return objective

    def gradient(self, weights, options=None):
        """
        Return the gradient of the regularized negative variational log likelihood 
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: gradient vector
        """
        if self.start_time != 0 and time.time() - self.start_time > self.max_time:
            if self.display == 'full':
                print('more than %d seconds...' % self.max_time)
            grad = np.zeros(len(weights))
            return grad
        else:
            self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, False)

            grad = np.zeros(len(weights))

            # add regularization penalties
            grad += self.l1_regularization * np.sign(weights)
            grad += self.l2_regularization * weights

            grad -= np.squeeze(self.label_expectations)
            grad += np.squeeze(self.inferred_expectations)

            return grad

    def dual_obj(self, weights, options=None):
        """
        Return the dual regularized negative variational log likelihood including Lagrangian penalty terms for 
        local inconsistencies of estimated marginals (i.e., beliefs)
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: dual objective value
        """
        if self.label_expectations is None or not self.fully_observed:
            self.label_expectations = self.calculate_expectations(weights, self.conditioned_belief_propagators, True)
        self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, True)
        term_p = sum(
            [np.true_divide(x.compute_dual_objective(), len(x.mn.variables)) for x in self.belief_propagators]) / len(
            self.belief_propagators)
        if not self.fully_observed:
            # recompute energy functional for label distributions only in latent variable case
            self.set_weights(weights, self.conditioned_belief_propagators)
            term_q = sum([np.true_divide(x.compute_dual_objective(), len(x.mn.variables)) for x in
                          self.conditioned_belief_propagators]) / len(self.conditioned_belief_propagators)
        else:
            term_q = np.dot(self.label_expectations, weights)

        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        return objec
