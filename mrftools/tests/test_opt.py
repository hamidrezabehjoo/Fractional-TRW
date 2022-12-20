"""Test class for optimizers"""
import unittest
from mrftools import *
import numpy as np
import itertools

class TestOpt(unittest.TestCase):
    """Test class for optimizers"""

    def test_solutions(self):
        """Test that optimizers lead to similar solutions."""
        optimizers = [sgd, ada_grad, rms_prop, adam, lbfgs]
        solutions = {}
        start = np.zeros(3)
        for optimizer in optimizers:
            solutions[optimizer] = optimizer(objective, gradient, start)

        for opt1, opt2 in itertools.combinations(optimizers, 2):
            print("*****\n%s: " % opt1.__name__)
            print(solutions[opt1])
            print("%s: " % opt2.__name__)
            print(solutions[opt2])
            assert np.allclose(solutions[opt1], solutions[opt2], rtol=1.e-1, atol=1.e-1), \
                "%s and %s did not get similar solutions." % (opt1.__name__, opt2.__name__)

    def test_callbacks(self):
        """Test that all methods properly use callback function"""
        self.did_call = False

        def simple_callback(x):
            self.did_call = True

        optimizers = [sgd, ada_grad, rms_prop, adam, lbfgs]
        start = np.zeros(3)
        for optimizer in optimizers:
            self.did_call = False
            optimizer(objective, gradient, start, callback=simple_callback)
            assert self.did_call, "Callback was not used"

    def test_args(self):
        """Test that optimizers pass arguments to objective"""

        def simple_obj(x, args=None):
            print(args)
            if args and args['hello'] is 'world':
                self.did_receive_obj_args = True
            else:
                self.did_receive_obj_args = False
            return 0

        def simple_grad(x, args=None):
            if args and args['hello'] is 'world':
                self.did_receive_grad_args = True
            else:
                self.did_receive_grad_args = False
            return 0 * x

        optimizers = [sgd, ada_grad, rms_prop, adam, lbfgs]
        start = np.zeros(3)
        dummy_args = {'hello': 'world'}
        for optimizer in optimizers:
            self.did_receive_obj_args = True
            self.did_receive_grad_args = True
            optimizer(simple_obj, simple_grad, start, args=dummy_args)
            assert self.did_receive_grad_args and self.did_receive_obj_args, \
                "Args were not properly passed for %s" % optimizer.__name__


def objective(x, args=None):
    """
    Simple function to optimize.
    :param x: input vector
    :type x: array
    :param args: unused
    :return: output of function
    :rtype: float
    """
    return 0.5 * x.dot(x) + np.sum(4 * x) + x[0] * x[-1]


def gradient(x, args=None):
    """
    Gradient of objective
    :param x: input vector
    :type x: array
    :param args: unused
    :return: gradient of function
    :rtype: array
    """
    grad = x + 4 + np.exp(x)
    grad[0] += x[-1]
    grad[-1] += x[0]

    return grad


if __name__ == '__main__':
    unittest.main()
