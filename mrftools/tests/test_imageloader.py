"""Test class for image loader utility"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os
from mrftools import *
from scipy.optimize import minimize, check_grad


class TestImageLoader(unittest.TestCase):
    """Unit test class for image loader utility"""
    def test_load(self):
        """Test that loading images and resizing them leads to correct size data."""
        loader = ImageLoader()

        train_dir = os.path.join(os.path.dirname(__file__), 'train_data')

        images, models, labels, names = loader.load_all_images_and_labels(train_dir, 2)
        files = [x for x in os.listdir(train_dir) if x.endswith(".jpg") or x.endswith('.png')]
        for i, filename in enumerate(files):
            full_name = os.path.join(train_dir, filename)
            img = Image.open(full_name)
            features = models[i].unary_features
            edge_features = models[i].edge_features
            edges = ImageLoader.get_all_edges(img)
            assert len(labels[i]) == img.width * img.height, "the size of labels is wrong"
            assert len(features) == img.width * img.height, "the size of features is wrong"
            assert len(edge_features) / 2 == len(edges), "the size of edge features is wrong"

            for x in range(img.width):
                for y in range(img.height):
                    var = (x, y)
                    assert len(features[var]) == 65, "Unary features were the wrong size"

            for x in range(img.width):
                for y in range(img.height - 1):
                    edge = ((x, y), (x, y + 1))
                    assert len(models[i].edge_features[edge]) == 11, "Edge features were the wrong size"

            for x in range(img.width - 1):
                for y in range(img.height):
                    edge = ((x, y), (x + 1, y))
                    assert len(models[i].edge_features[edge]) == 11, "Edge features were the wrong size"

    def test_unary_only(self):
        """
        Test accuracy of learned model using unary features only (i.e., just predicting each pixel independently.)
        Implemented via a separate logistic regression.
        """
        num_features = 65
        num_states = 2

        all_pixel, all_label = load_all_images_and_labels(os.path.join(os.path.dirname(__file__), 'train_data'), num_features, 3)

        initial_w = np.zeros(num_features * num_states)
        res = minimize(objective, initial_w, method="L-BFGS-B", args=(all_pixel, all_label, num_features, num_states),
                       jac=gradient)
        weights = res.x

        accuracy_training = accuracy(weights, all_pixel, all_label, num_features, num_states)
        print ("accuracy on training set: %f" % accuracy_training)
        assert (accuracy_training >= 0.9), "Unary classification accuracy on training data is less than 0.9"

        all_pixel, all_label = load_all_images_and_labels(os.path.join(os.path.dirname(__file__), 'test_data'), num_features, 1)
        accuracy_testing = accuracy(weights, all_pixel, all_label, num_features, num_states)
        print ("accuracy on testing set: %f" % accuracy_testing)
        assert (accuracy_testing >= 0.7), "Unary classification accuracy on testing data is less than 0.7"

    def test_tree_probability_calculation(self):
        """Test the spanning-tree edge appearance probability computations (used for TRBP) are correct."""
        height = 3
        width = 3
        tree_prob = ImageLoader.calculate_tree_probabilities_snake_shape(width, height)
        assert (tree_prob[(0, 0), (0, 1)] == 0.75), "side edge probability does not equal to 0.75"
        assert (tree_prob[(0, 1), (0, 0)] == 0.75), "side edge probability does not equal to 0.75"
        assert (tree_prob[(1, 1), (1, 0)] == 0.5), "center edge probability does not equal to 0.5"

        side_edge_count = 0
        center_edge_count = 0
        for keys in tree_prob:
            if tree_prob[keys] == 0.75:
                side_edge_count += 1
            else:
                center_edge_count += 1

        assert (side_edge_count == 16), "number of side edges not correct: %d" % (side_edge_count)
        assert (center_edge_count == 8), "number of center edges not correct"

    def test_model_matrix_structure(self):
        """Test that the loaded model has the correct matrix structure."""
        loader = ImageLoader(10, 10)

        train_dir = os.path.join(os.path.dirname(__file__), 'train_data')

        images, models, labels, names = loader.load_all_images_and_labels(train_dir, 2, 1)

        model = models[0]

        model.create_matrices()

        for edge, i in model.message_index.items():
            from_index = model.var_index[edge[0]]
            to_index = model.var_index[edge[1]]
            assert model.message_from[i] == from_index, "Message sender index is wrong"
            assert model.message_to[i] == to_index, "Message receiver index is wrong"
            assert model.message_to_map.getrow(i).getcol(to_index) == 1, "Message receiver matrix map is wrong"

        assert np.all(np.sum(model.message_to_map.todense(), axis=1) == 1), \
            "Message sender map has a row that doesn't sum to 1.0"

        assert np.allclose(model.edge_pot_tensor[:, :, :model.num_edges],
                           model.edge_pot_tensor[:, :, model.num_edges:]), "Edge tensor structure is wrong"
        assert np.allclose(model.edge_pot_tensor[:, :, :model.num_edges],
                           model.edge_pot_tensor[:, :, model.num_edges:].transpose(1, 0, 2)), \
            "Edge tensor is not symmetric"


def softmax(x):
    """Softmax function used for logistic regression."""
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted), 1, keepdims=True)


def objective(weights, features, label_vec, num_features, num_states):
    """Objective for logistic regression."""
    probabilities = np.dot(features, weights.reshape((num_features, num_states)))
    probabilities = softmax(probabilities)

    label_mat = np.zeros((probabilities.shape))

    for i in range(num_states):
        label_mat[:, i] = np.where(label_vec == i, 1, 0)

    return -np.sum(label_mat * np.nan_to_num(np.log(probabilities))) + np.dot(weights.ravel(), weights.ravel())
    # return -np.sum(label_mat * np.nan_to_num(np.log(probabilities)))


def gradient(weights, features, label_vec, num_features, num_states):
    """Gradient for logistic regression."""
    probabilities = np.dot(features, weights.reshape((num_features, num_states)))
    probabilities = softmax(probabilities)

    label_mat = np.zeros((probabilities.shape))

    for i in range(num_states):
        label_mat[:, i] = np.where(label_vec == i, 1, 0)

    g = -features.T.dot(label_mat - probabilities).ravel() + 2 * weights.ravel()
    # g = -features.T.dot(label_mat - probabilities).ravel()
    return g


def accuracy(weights, features, label_vec, num_features, num_states):
    """
    Compute the accuracy for logistic regression
    :param weights: model weights for pixel-based logistic regression
    :type weights: ndarray
    :param features: input data
    :type features: ndarray
    :param label_vec: ground truth values
    :type label_vec: array
    :param num_features: number of features
    :type num_features: int
    :param num_states: number of possible classes
    :type num_states: int
    :return: accuracy of logistic regression on the pixel-labeling task
    :rtype: float
    """
    total_error = 0
    scores = np.dot(features, weights.reshape((num_features, num_states)))
    positive_vec = np.argmax(scores, axis=1)
    error = np.sum(np.abs(positive_vec - label_vec))
    num_pixels = np.shape(label_vec)

    total_error = total_error + error
    accuracy = 1 - (total_error / num_pixels)

    return accuracy


def load_all_images_and_labels(path, num_features, num_images):
    """
    Load all images and labels from a directory as just pixel variables
    :param path: directory to load from
    :type path: string
    :param num_features: number of features for each pixel
    :type num_features: int
    :param num_images: number of images to load
    :type num_images: int
    :return: tuple containing a matrix of all pixels and an array of labels
    :rtype: tuple
    """
    loader = ImageLoader()
    all_pixel = np.zeros((0, num_features))
    all_label = []

    # files = [x for x in os.listdir('./train') if x.endswith(".jpg") or x.endswith('.png')]
    files = [x for x in os.listdir(path) if x.endswith(".jpg") or x.endswith('.png')]
    for i, filename in enumerate(files):
        if i < num_images:
            full_name = os.path.join(path, filename)
            # full_name = os.path.join('./train', filename)
            img = Image.open(full_name)
            height = img.size[1]
            width = img.size[0]

            features, edge_features = ImageLoader.compute_features(img)
            pixel = np.asarray(list(features.values()))
            all_pixel = np.concatenate((all_pixel, pixel), axis=0)

            label_dict = loader.load_label_dict(full_name)
            label_vec = np.asarray(list(label_dict.values()))
            all_label = np.concatenate((all_label, label_vec), axis=0)

    return all_pixel, all_label


if __name__ == '__main__':
    unittest.main()