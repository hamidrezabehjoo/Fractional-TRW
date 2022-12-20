"""Miscellaneous utility functions and classes"""
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from .ConvexBeliefPropagator import ConvexBeliefPropagator


def instantiate_convex_map(self, model):
    """
    Instantiate convex belief propagators with temperatures that force numerical MAP inference
    
    :param model: MarkovNet model to perform inference on
    :return: ConvexBeliefPropagator with low counting numbers
    """
    default_counting_numbers = dict()
    for var in model.variables:
        default_counting_numbers[var] = 0.1
        for neighbor in model.neighbors[var]:
            if var < neighbor:
                default_counting_numbers[(var, neighbor)] = 0.1

    bp = ConvexBeliefPropagator(model, default_counting_numbers)
