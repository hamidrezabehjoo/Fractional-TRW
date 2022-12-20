"""Tests of integration of image loading, learning, and inference"""
from mrftools import *
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
import copy


class TestIntegration(unittest.TestCase):
    """Test class to test interface between image loading, learning, and inference"""
    def test_loading_and_learning(self):
        """Test loading of a full image-segmentation training set, learning, and inference."""
        loader = ImageLoader(10, 10)

        images, models, labels, names = loader.load_all_images_and_labels(os.path.join(os.path.dirname(__file__),
                                                                                       'train_data'), 2, 3)

        learner = Learner(MatrixBeliefPropagator)

        learner.set_regularization(0.0, 0.00001)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        d_unary = 65
        num_states = 2
        d_edge = 11

        weights = np.zeros(d_unary * num_states + d_edge * num_states**2)

        args = {'max_iter': 200}

        new_weights = learner.learn(weights, opt_args=args)

        unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
        pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states**2))
        print("Unary weights:\n" + repr(unary_mat))
        print("Pairwise weights:\n" + repr(pair_mat))

        # test inference with weights

        i = 1

        models[i].set_weights(new_weights)
        bp = MatrixBeliefPropagator(models[i])
        bp.infer(display='final')
        bp.load_beliefs()

        beliefs = np.zeros((images[i].height, images[i].width))
        label_img = np.zeros((images[i].height, images[i].width))
        errors = 0
        baseline = 0

        for x in range(images[i].width):
            for y in range(images[i].height):
                beliefs[y, x] = np.exp(bp.var_beliefs[(x, y)][1])
                label_img[y, x] = labels[i][(x, y)]
                errors += np.abs(labels[i][(x, y)] - np.round(beliefs[y, x]))
                baseline += labels[i][(x, y)]

        # # uncomment this to plot the beliefs
        # plt.subplot(131)
        # plt.imshow(images[i], interpolation="nearest")
        # plt.subplot(132)
        # plt.imshow(label_img, interpolation="nearest")
        # plt.subplot(133)
        # plt.imshow(beliefs, interpolation="nearest")
        # plt.show()

        print("Error rate: %f" % np.true_divide(errors, images[i].width * images[i].height))
        print("Baseline from guessing all background: %f" % np.true_divide(baseline, images[i].width * images[i].height))
        assert errors < baseline, "Learned model did no better than guessing all background."

    def test_consistency(self):
        """Test consistency and correctness of inference on an image-segmentation MRF"""
        loader = ImageLoader(1, 4)
        np.random.seed(0)

        images, models, labels, names = loader.load_all_images_and_labels(
            os.path.join(os.path.dirname(__file__), 'train_data'), 2, 1)
        i = 0

        d_unary = 65
        num_states = 2
        d_edge = 11

        new_weights = 0.1 * np.random.randn(d_unary * num_states + d_edge * num_states ** 2)

        models[i].set_weights(new_weights)
        models[i].load_factors_from_matrices()

        for inference_type in [BeliefPropagator, MatrixBeliefPropagator]:

            bp = inference_type(models[i])
            bp.infer(display='full')
            bp.load_beliefs()

            bf = BruteForce(models[i])

            # check unary marginal agreement with brute force
            for var in sorted(bp.mn.variables):
                unary_belief = np.exp(bp.var_beliefs[var])
                assert np.allclose(np.sum(unary_belief), 1.0), "Unary belief not normalized"
                unary_error = np.sum(np.abs(bf.unary_marginal(var) - unary_belief))
                print("Unary marginal for %s. Error compared to brute force: %e" % (repr(var), unary_error))
                assert unary_error < 1e-3, "Unary error was too large compared to brute force"

            # check pairwise marginal agreement with brute force
            for var in sorted(bp.mn.variables):
                for neighbor in sorted(bp.mn.get_neighbors(var)):
                    edge_error = np.sum(
                        np.abs(bf.pairwise_marginal(var, neighbor) - np.exp(bp.pair_beliefs[(var, neighbor)])))
                    print("Pair %s marginal error compared to brute force: %e" % (repr((var, neighbor)), edge_error))
                    assert edge_error < 1e-3, "Pairwise error was too large compared to brute force"

            # check consistency
            for var in sorted(bp.mn.variables):
                unary_belief = np.exp(bp.var_beliefs[var])

                for neighbor in sorted(bp.mn.get_neighbors(var)):
                    pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                    assert np.allclose(np.sum(pair_belief), 1.0), "Pair belief not normalized"

                    print(pair_belief, unary_belief)
                    assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"

            print("Finished and passed tests for " + repr(inference_type))

    def test_belief_propagators(self):
        """Compare belief propagator implementations on image-segmentation MRFs"""
        loader = ImageLoader(4, 4)
        np.random.seed(0)

        images, models, labels, names = loader.load_all_images_and_labels(
            os.path.join(os.path.dirname(__file__), 'train_data'), 2, 1)
        i = 0

        d_unary = 65
        num_states = 2
        d_edge = 11

        new_weights = 0.1 * np.random.randn(d_unary * num_states + d_edge * num_states ** 2)

        models[i].set_weights(new_weights)
        models[i].load_factors_from_matrices()

        model = models[i]

        bp = BeliefPropagator(model)
        bp.load_beliefs()

        mat_bp = MatrixBeliefPropagator(model)
        mat_bp.load_beliefs()

        for i in range(4):
            for var in sorted(bp.mn.variables):
                for neighbor in sorted(bp.mn.get_neighbors(var)):
                    edge = (var, neighbor)
                    bp_message = bp.messages[edge]

                    if edge in mat_bp.mn.message_index:
                        edge_index = mat_bp.mn.message_index[edge]
                    else:
                        edge_index = mat_bp.mn.message_index[(edge[1], edge[0])] + mat_bp.mn.num_edges

                    mat_bp_message = mat_bp.message_mat[:, edge_index].ravel()

                    assert np.allclose(bp_message, mat_bp_message), \
                        "BP and matBP did not agree on message for edge %s in iter %d" % (repr(edge), i) \
                        + "\nBP: " + repr(bp_message) + "\nmatBP: " + repr(mat_bp_message)

                    # print "Message %s is OK" % repr(edge)

                    assert np.allclose(bp.pair_beliefs[edge], mat_bp.pair_beliefs[edge]), \
                        "BP and matBP did not agree on pair beliefs after %d message updates" % i

                assert np.allclose(bp.var_beliefs[var], mat_bp.var_beliefs[var]), \
                    "BP and matBP did not agree on unary beliefs after %d message updates" % i

            bp.update_messages()
            bp.load_beliefs()
            mat_bp.update_messages()
            mat_bp.load_beliefs()

    def test_dual_learner_speed(self):
        """Test the speed of inner-dual learner against primal learner"""
        d_unary = 65
        num_states = 2
        d_edge = 11

        weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

        loader = ImageLoader(10, 10)

        images, models, labels, names = loader.load_all_images_and_labels(
            os.path.join(os.path.dirname(__file__), 'train_data'), 2, 3)

        learner = Learner(MatrixBeliefPropagator)

        learner.set_regularization(0.0, 1.0)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        args = {'max_iter': 100}

        start = time.time()
        subgrad_weights = learner.learn(weights, optimizer=ada_grad, opt_args=args)
        subgrad_time = time.time() - start
        print("Learner took %f seconds" % subgrad_time)

        learner = PairedDual(MatrixBeliefPropagator)
        learner.set_regularization(0.0, 1.0)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        args = {'max_iter': 500}

        start = time.time()
        paired_weights = learner.learn(weights, optimizer=ada_grad, opt_args=args)
        pd_time = time.time() - start
        print("PD took %f seconds" % pd_time)

        assert pd_time < subgrad_time, "Paired dual learning took longer than subgradient"

        print(learner.subgrad_obj(subgrad_weights), learner.subgrad_obj(paired_weights))

        assert learner.subgrad_obj(subgrad_weights) >= learner.subgrad_obj(paired_weights), \
            "subgrad reached lower minimum than paired dual"

    def test_optimizer(self):
        """Test that, once the optimizer outputs and optimum, that it does not find a better optimum on a second call"""
        d_unary = 65
        num_states = 2
        d_edge = 11

        learner_type = PairedDual
        inference_type = MatrixBeliefPropagator

        weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

        image_size = 6

        loader = ImageLoader(image_size, image_size)

        images, models, labels, names = loader.load_all_images_and_labels(
            os.path.join(os.path.dirname(__file__), 'train_data'), 2, 2)

        # make latent variable

        for label in labels:
            # print "Number of labels: %d" % len(label)
            for x in range(int(image_size / 2)):
                for y in range(int(image_size / 2)):
                    del label[(x, y)]
            # print "Number of labels after removing quadrant: %d" % len(label)

        learner = learner_type(inference_type)
        learner.set_regularization(0.0, 1.0)

        eval_learner = learner_type(inference_type)
        eval_learner.set_regularization(0.0, 1.0)

        for model, states in zip(models, labels):
            learner.add_data(states, model)
            eval_learner.add_data(states.copy(), copy.copy(model))

        objectives = []

        op = ObjectivePlotter(eval_learner.subgrad_obj, eval_learner.subgrad_grad)
        # op = ObjectivePlotter(learner.dual_obj, eval_learner.subgrad_grad)

        for i in range(4):
            prev_weights = weights
            start = time.time()

            weights = learner.learn(prev_weights, callback=op.callback)
            subgrad_time = time.time() - start
            print("Learner took %f seconds" % subgrad_time)

            objectives.append(learner.subgrad_obj(weights))

            print("After i %d of optimization, objective was %e." % (i, objectives[i]))

        for i in range(len(objectives) - 1):
            assert objectives[i] - objectives[i + 1] < 1e-2, \
                "Optimizer improved after supposedly reaching optimum"


if __name__ == '__main__':
    unittest.main()