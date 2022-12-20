"""Class to do generative learning directly on MRF parameters."""
from copy import deepcopy

from .Learner import Learner
from .LogLinearModel import LogLinearModel
from .MatrixBeliefPropagator import MatrixBeliefPropagator


class ApproxMaxLikelihood(Learner):
    """
    Object that runs approximate maximum likelihood parameter training.
    This method creates an indicator model where every feature is an indicator function, also known as an 
    overcomplete representation.
    """

    def __init__(self, markov_net, inference_type=MatrixBeliefPropagator):
        """
        Initialize the learner with the Markov network whose parameters are to be learned.
        
        :param markov_net: MarkovNet object whose parameters are to be learned
        :type markov_net: MarkovNet
        :param inference_type: Inference method to use for estimating the feature expectations during learning
        :type inference_type: Inference
        """
        super(ApproxMaxLikelihood, self).__init__(inference_type)
        self.base_model = LogLinearModel()
        self.base_model.create_indicator_model(markov_net)

    def add_data(self, labels):
        """
        Add observed training data
        
        :param labels: dictionary containing an integer state value for each observed variable 
        :type labels: dict
        :return: None
        """
        model = deepcopy(self.base_model)
        super(ApproxMaxLikelihood, self).add_data(labels, model)

        # as a hack to save time, since these models don't condition on anything, make all belief propagators equal
        self.belief_propagators = [self.belief_propagators[0]]
