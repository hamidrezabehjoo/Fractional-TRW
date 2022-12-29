"""Gibbs sampling class"""
from __future__ import division

import random
from collections import Counter

import numpy as np
from .MatrixBeliefPropagator import logsumexp


class GibbsSampler(object):
    """Object that can run Gibbs sampling on a MarkovNet"""

    def __init__(self, markov_net):
        """Initialize belief propagator for markov_net."""
        self.mn = markov_net
        self.states = dict()
        self.unary_weights = dict()
        self.samples = list()

    @staticmethod
    def generate_state(weight):
        """Generate state according to the given weight"""
        r = random.uniform(0, 1)
        # Sum = sum(weight.values())
        Sum = sum(weight)
        rnd = r * Sum
        for i in range(len(weight)):
            rnd = rnd - weight[i]
            if rnd < 0:
                return i

    def init_states(self, seed=None):
        """
        Initialize the state of each node.
        
        :param seed: random seed
        """
        if seed is not None:
            np.random.seed(seed)

        for var in self.mn.variables:
            weight = self.mn.unary_potentials[var]
            weight = np.exp(weight - logsumexp(weight))
            self.unary_weights[var] = weight
            self.states[var] = self.generate_state(self.unary_weights[var])

    def update_states(self):
        """Update the state of each node based on neighbor states."""
        for var in self.mn.variables:
            weight = self.mn.unary_potentials[var]
            for neighbor in self.mn.neighbors[var]:
                weight = weight + self.mn.get_potential((var, neighbor))[:, self.states[neighbor]]
            weight = np.exp(weight - logsumexp(weight))
            self.states[var] = self.generate_state(weight)

    def burn_in(self, iters):
        """
        Run the state update procedure until mixed. 
        :param iters: number of iterations for mixing
        """
        for i in range(0, iters):
            self.update_states()

    def sampling(self, num):
        """
        Run sampling
        
        :param num: number of samples to collect
        """
        for i in range(0, num):
            self.update_states()
            self.samples.append(self.states.copy())
            # for i in range(0, s-1):
            #     self.update_states()

    def gibbs_sampling(self, burn_in, num):
        """
        Run Gibbs sampling 
        
        :param burn_in: number of burn-in samples to discard
        :type burn_in: int
        :param num: number of samples to collect once burn-in phase is done
        :type num: int
        """
        self.burn_in(burn_in)
        self.sampling(num)

    def count_occurrences(self, var):
        """
        Count the number of times in our samples the variable was in each state.
        
        :param var: variable to count the states of
        :type var: object
        :return: count array of state occurrences
        :rtype: arraylike
        """
        counts = Counter([sample[var] for sample in self.samples])
        count_array = np.asarray([counts[x] for x in range(self.mn.num_states[var])])
        return count_array
