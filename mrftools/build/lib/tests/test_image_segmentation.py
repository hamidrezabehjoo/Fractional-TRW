"""Test class for learning and inference tasks on synthetic image segmentation"""
import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
import itertools


class TestImageSegmentation(unittest.TestCase):
    """Test class for learning and inference tasks on synthetic image segmentation with latent variables"""

    # create shared data as tuples of dictionaries: (training_labels, features, true_labels)
    data = [({(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): None},
             {(0, 0): np.array([1, 0, 0]),
              (0, 1): np.array([0, 1, 0]),
              (1, 0): np.array([1, 0, 0]),
              (1, 1): np.array([0, 0, 1])},
             {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}),
            ({(0, 0): 0, (0, 1): 0, (1, 0): None, (1, 1): 2},
             {(0, 0): np.array([1, 0, 0]),
              (0, 1): np.array([1, 0, 0]),
              (1, 0): np.array([0, 1, 0]),
              (1, 1): np.array([0, 0, 1])},
             {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 2})]

    def get_average_accuracy(self, weight_records):
        """
        Compute the average accuracy across all data for each weight vector in the weight record.
        
        :param iters: Number of iterations run by optimizer
        :type iters: int
        :param weight_records: matrix of stacked weight vectors
        :type weight_records: ndarray
        :return: array of average accuracy
        :rtype: array
        """
        d = 3
        width = 2
        height = 2
        num_states = 3
        accuracy_ave_train = np.zeros(weight_records.shape[0])
        counter = 0

        for i in range(len(self.data)):
            data = self.data[i]
            counter += 1
            accuracy = []
            for j in range(weight_records.shape[0]):
                w = weight_records[j, :]
                w_unary = np.reshape(w[0:d * num_states], (d, num_states)).T
                w_pair = np.reshape(w[num_states * d:], (num_states, num_states))
                w_unary = np.array(w_unary, dtype=float)
                w_pair = np.array(w_pair, dtype=float)

                mn = self.create_markov_net(2, 2, w_unary, w_pair, data[1])

                # perform inference
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display="off")
                bp.compute_beliefs()
                bp.compute_pairwise_beliefs()
                bp.load_beliefs()

                # measure predictions
                prediction = []
                correct = 0
                for x in range(width):
                    for y in range(height):
                        pixel_id = (x, y)
                        prediction.append(np.argmax(bp.var_beliefs[pixel_id]))
                        if data[2][pixel_id] == prediction[-1]:
                            correct += 1

                accuracy.append(np.true_divide(correct, width * height))

            accuracy_ave_train = np.add(accuracy_ave_train, accuracy)

        accuracy_ave_train = np.true_divide(accuracy_ave_train, counter)

        return accuracy_ave_train

    def get_all_edges(self, width, height):
        """
        Get all edges in a width by height grid
        :param width: width of grid
        :type width: int
        :param height: height of grid
        :type height: int
        :return: list of edges in grid graph
        :rtype: list
        """
        edges = []

        # add horizontal edge_index
        for x in range(width - 1):
            for y in range(height):
                edge = ((x, y), (x + 1, y))
                edges.append(edge)

        # add vertical edge_index
        for x in range(width):
            for y in range(height - 1):
                edge = ((x, y), (x, y + 1))
                edges.append(edge)

        return edges

    def create_markov_net(self, height, width, w_unary, w_pair, pixels):
        """
        Generates a grid Markov net with the provided size, unary weights and pixel data
        
        :param height: number of rows of the MRF
        :param width: number of columns of the MRF
        :param w_unary: linear weights to generate unary potentials from pixel features
        :param w_pair: pairwise potential function (table)
        :param pixels: pixel data dictionary
        :return: constructed Markov net object
        """
        mn = MarkovNet()
        edges = self.get_all_edges(width, height)

        # Set unary factor
        for i in range(width):
            for j in range(height):
                mn.set_unary_factor((i, j), np.dot(w_unary, pixels[(i, j)]))

        for edge in edges:
            mn.set_edge_factor(edge, w_pair)

        return mn

    def create_model(self, num_states, width, height, d, data):
        """
        Create LogLinearModel of pixel-based image segmentation model with random weights.
        :param num_states: number of classes to segment
        :type num_states: int
        :param width: width of grid
        :type width: int
        :param height: height of grid
        :type height: int
        :param d: number of features for each unary potential
        :type d: int
        :param data: dictionary of labels
        :type data: 
        :return: LogLinearModel for inferring pixel labels
        :rtype: LogLinearModel
        """
        model = LogLinearModel()

        for key in data.keys():
            model.declare_variable(key, num_states)
            model.set_unary_weights(key, np.random.randn(num_states, d))

        for key, value in data.items():
            model.set_unary_features(key, value)

        model.set_all_unary_factors()

        for edge in self.get_all_edges(width, height):
            model.set_edge_factor(edge, np.zeros((num_states, num_states)))

        edge_probabilities = dict()

        for edge in model.edge_potentials:
            edge_probabilities[edge] = 0.75

        model.tree_probabilities = edge_probabilities

        return model

    def set_up_learner(self, learner):
        """
        Add training data to learner and set regularizer
        :param learner: learner object to prepare for learning
        :return: None
        """
        data_dim = 3
        num_states = 3

        models = []
        labels = []
        for i in range(len(self.data)):
            m = self.create_model(num_states, 2, 2, data_dim, self.data[i][1])
            models.append(m)

            label_dict = self.data[i][0]
            # remove observed (latent) pixels
            for key in list(label_dict.keys()):
                if label_dict[key] is None:
                    del label_dict[key]

            labels.append(label_dict)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        learner.set_regularization(0, 0.1)

    def test_subgradient_obj(self):
        """Test that the subgradient learner (Learner) reduces the objective as it optimizes."""
        np.random.seed(0)

        weights = np.zeros(9 + 9)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        # initialize the callback utility that saves weights during optimization
        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        num_iters = weight_record.shape[0]

        # check that the objective value gets smaller with each iteration
        old_obj = np.Inf
        for i in range(num_iters):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert (new_obj <= old_obj + 1e-8), "subgradient objective did not decrease" + repr((new_obj, old_obj))
            old_obj = new_obj

    def test_EM_obj(self):
        """Test that the EM learner reduces the objective as it optimizes."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        old_obj = learner.subgrad_obj(weight_record[0, :])
        new_obj = learner.subgrad_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "EM objective did not decrease"

    def test_paired_dual_obj(self):
        """Test that the paired-dual learner reduces the objective as it optimizes."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record

        old_obj = learner.dual_obj(weight_record[0, :])
        new_obj = learner.dual_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "paired dual objective did not decrease"

    def test_primal_dual_obj(self):
        """Test that the primal-dual learner reduces the objective as it optimizes."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = PrimalDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record

        old_obj = learner.dual_obj(weight_record[0, :])
        new_obj = learner.dual_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "primal dual objective did not decrease"

    def test_subgradient_training_accuracy(self):
        """Test that the subgradient learner improves its accuracy when training."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        subgrad_weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record

        subgrad_accuracy_ave_train = self.get_average_accuracy(subgrad_weight_record)

        for i in range(len(subgrad_accuracy_ave_train)):
            print("iter %d: accuracy %e" % (i, subgrad_accuracy_ave_train[i]))
            if i != len(subgrad_accuracy_ave_train) - 1:
                assert (subgrad_accuracy_ave_train[i] <=
                        subgrad_accuracy_ave_train[i + 1]), "subgradient accuracy is not increasing"

        return subgrad_accuracy_ave_train

    def test_EM_training_accuracy(self):
        """Test that the EM learner improves its accuracy when training."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        em_weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record

        em_accuracy_ave_train = self.get_average_accuracy(em_weight_record)

        for i in range(len(em_accuracy_ave_train)):
            print("iter %d: accuracy %e" % (i, em_accuracy_ave_train[i]))
            if i != len(em_accuracy_ave_train) - 1:
                assert (em_accuracy_ave_train[i] <= em_accuracy_ave_train[i + 1]), \
                    "EM accuracy is not increasing"

        return em_accuracy_ave_train

    def test_paired_dual_accuracy(self):
        """Test that the paired-dual learner improves its accuracy when training."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record

        paired_dual_accuracy_ave_train = self.get_average_accuracy(weight_record)

        for i in range(len(paired_dual_accuracy_ave_train)):
            print("iter %d: accuracy %e" % (i, paired_dual_accuracy_ave_train[i]))
            if i != len(paired_dual_accuracy_ave_train) - 1:
                assert (paired_dual_accuracy_ave_train[i] <= paired_dual_accuracy_ave_train[
                    i + 1]), "paired dual accuracy is not increasing"

        return paired_dual_accuracy_ave_train

    def test_primal_dual_accuracy(self):
        """Test that the paired-dual learner improves its accuracy when training."""
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = PrimalDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, callback=wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        t = time_record[0]

        primal_dual_accuracy_train = self.get_average_accuracy(weight_record)

        for i in range(len(primal_dual_accuracy_train)):
            print("iter %d: accuracy %e" % (i, primal_dual_accuracy_train[i]))
            if i != len(primal_dual_accuracy_train) - 1:
                assert (primal_dual_accuracy_train[i] <= primal_dual_accuracy_train[
                    i + 1]), "primal dual accuracy is not increasing"

        return primal_dual_accuracy_train


if __name__ == '__main__':
    unittest.main()