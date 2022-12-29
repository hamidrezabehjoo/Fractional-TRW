"""Utility class for loading images into grid MRF structures for image segmentation"""
import itertools
import os
import time

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .LogLinearModel import LogLinearModel


class ImageLoader(object):
    """
    Image loading class that reads images from disk and creates grid-structured CRFs for image segmentation
    """
    def __init__(self, max_width=0, max_height=0):
        """
        Initialize an ImageLoader
        
        :param max_width: maximum width of image to load. This object will resize any images that are wider than this.
        :type max_width: int
        :param max_height: maximum height of image to load. This object will resize any images that are taller.
        :type max_height: int
        """
        self.max_width = max_width
        self.max_height = max_height

    def load_image(self, path):
        """
        Load image at path and resize according to our maximum size parameters
        :param path: location of image in file system
        :type path: string
        :return: PIL image object
        """
        img = Image.open(path)
        img1 = img

        if self.max_width > 0 and self.max_height > 0:
            img = img.resize((self.max_width, self.max_height), resample=PIL.Image.BICUBIC)

        return img

    def load_label_img(self, image_name):
        """
        Load the semantic segmentation labels, assuming they are stored in a space-delimited text table of the same
        size as the image. This method will resize the label image according to this object's maximum size parameters.
        The labels should be integer values starting from 0.
        
        The label file must have the same name as the associated image, except instead of the image type extension,
        it has '_label.txt' as its suffix. For example the image './picture1.jpg' should have label file
        './picture1_label.txt'
        
        :param image_name: full path to the image file (this method will remove the extension and append '_label.txt'
        :type image_name: string
        :return: PIL image object with the labels as the greyscale intensity value.
        :rtype: 
        """
        label_file = os.path.splitext(image_name)[0] + '_label.txt'
        label_mat = np.loadtxt(label_file)

        label_img = Image.fromarray(label_mat.astype(np.uint8))

        if self.max_width > 0 and self.max_height > 0:
            label_img = label_img.resize((self.max_width, self.max_height), resample=PIL.Image.NEAREST)

        return label_img

    def load_label_dict(self, image_name):
        """
        Create a dictionary of label values for an image and label file.
        
        :param image_name: full path to the image file (this method will remove the extension and append '_label.txt'
        :type image_name: string
        :return: dictionary with pixel names as keys (pixel_width, pixel_height) and integer class as values
        :rtype: dict
        """
        label_img = self.load_label_img(image_name)

        label_pixels = label_img.load()
        label_dict = dict()
        for x in range(label_img.width):
            for y in range(label_img.height):
                label_dict[(x, y)] = label_pixels[x, y]

        return label_dict

    def draw_image_and_label(self, name):
        """
        Draw an image and its ground truth label.
        
        :param name: path to image file
        :type name: string
        :return: None
        """
        img = self.load_image(name)
        labels = self.load_label_img(name)
        features = ImageLoader.compute_features(img)

        plt.subplot(121)
        plt.imshow(img, interpolation='nearest')
        plt.xlabel('Original Image')
        plt.subplot(122)
        plt.imshow(labels, interpolation='nearest')
        plt.xlabel("Labels")
        plt.show()

    def load_all_images_and_labels(self, directory, num_states, num_images=np.inf):
        """
        Load all jpg or png images from a directory.
        
        :param directory: path to directory
        :type directory: string
        :param num_states: number of possible classes for segmentation
        :type num_states: int
        :param num_images: maximum number of images to load
        :type num_images: int
        :return: tuple containing the images, LogLinearModels, the labels, and the names of the images
        :rtype: tuple
        """
        images = []
        models = []
        labels = []
        names = []
        files = [x for x in os.listdir(directory) if x.endswith(".jpg") or x.endswith('.png')]
        num_images = min(len(files), num_images)
        start = time.time()
        for i, filename in enumerate(files):
            if i < num_images:
                full_name = os.path.join(directory, filename)
                img = self.load_image(full_name)
                model = ImageLoader.create_model(img, num_states)
                label_vec = self.load_label_dict(full_name)

                names.append(filename)
                images.append(img)
                models.append(model)

                labels.append(label_vec)

                if i % 10 == 0 or i == num_images - 1:
                    elapsed = time.time() - start
                    eta = np.true_divide(elapsed, i + 1) * (len(files) - i - 1)
                    print("Loaded %d of %d. Time elapsed: %f. ETA: %f" % (i + 1, num_images, elapsed, eta))

        return images, models, labels, names

    @staticmethod
    def create_model(img, num_states):
        """
        Create a log-linear model for the image
        
        :param img: PIL image object
        :type img: image
        :param num_states: number of labels possible for each image
        :type num_states: int
        :return: LogLinearModel representing the image with variables for each pixel 
        :rtype: LogLinearModel 
        """
        model = LogLinearModel()

        # generate spanning tree probabilities for TRBP
        tree_prob = ImageLoader.calculate_tree_probabilities_snake_shape(img.width, img.height)
        model.tree_probabilities = tree_prob

        # create features
        feature_dict, edge_feature_dict = ImageLoader.compute_features(img)

        # create pixel variables
        for pixel, feature_vec in feature_dict.items():
            model.declare_variable(pixel, num_states)
            model.set_unary_features(pixel, feature_vec)
            model.set_unary_factor(pixel, np.zeros(num_states))

        # create edge variables
        for edge, edge_feature_vec in edge_feature_dict.items():
            model.set_edge_features(edge, edge_feature_vec)
            model.set_edge_factor(edge, np.eye(num_states))

        model.create_matrices()

        return model

    @staticmethod
    def show_images(images):
        """
        Draw images onscreen.
    
        :param images: iterable of images
        :type images: iterable
        :return: None
        """
        plt.clf()
        total = len(images)

        rows = np.ceil(np.sqrt(total))
        cols = rows

        for i, img in enumerate(images):
            plt.clf()
            plt.imshow(img, interpolation='nearest')
            plt.pause(1e-10)

    @staticmethod
    def get_all_edges(img):
        """
        Create a list of all edges in a grid structured graph of the same width and height as the image.
        :param img: image object with a width and height 
        :type img: image
        :return: list of edges
        :rtype: list
        """
        edges = []

        # add horizontal edges
        for x in range(img.width - 1):
            for y in range(img.height):
                edge = ((x, y), (x + 1, y))
                edges.append(edge)

        # add vertical edges
        for x in range(img.width):
            for y in range(img.height - 1):
                edge = ((x, y), (x, y + 1))
                edges.append(edge)

        return edges

    @staticmethod
    def compute_features(img):
        """
        Generate pixel and edge features based on Fourier expansion. 
        Method ported from https://arxiv.org/abs/1301.3193 by Justin Domke.
        
        :param img: image to compute features of
        :type img: image
        :return: tuple of two dictionaries: (1) a dictionary of pixel features, (2) a dictionary of edge features
        :rtype: tuple
        """
        pixels = img.load()

        base_features = np.zeros((img.width * img.height, 5))
        pixel_ids = []
        nthresh = 10
        edge_ids = []

        i = 0
        for x in range(img.width):
            for y in range(img.height):
                base_features[i, :3] = pixels[x, y]
                base_features[i, 3:] = (x, y)

                pixel_ids.append((x, y))

                i += 1

        base_features /= [255, 255, 255, img.width, img.height]

        # perform fourier expansion

        coeffs = list(itertools.product([0, 1], repeat=5))
        coeffs = np.column_stack(coeffs)

        prod = base_features.dot(coeffs)
        feature_mat = np.hstack((np.sin(prod), np.cos(prod), np.ones((img.width * img.height, 1))))

        if img.mode == 'RGB':
            channels = 3
        elif img.mode == 'L':
            channels = 1
        else:
            print("Unknown mode: %s" % img.mode)

        edges = ImageLoader.get_all_edges(img)

        edge_feature_mat = np.zeros((len(edges), nthresh + 1))

        for j, edge in enumerate(edges):
            diff = 0
            edge_feats_vec = np.zeros(nthresh + 1)
            for z in range(channels):
                diff += np.true_divide((pixels[edge[0]][z] - pixels[edge[1]][z]), 255) ** 2

            diff = np.sqrt(diff)
            for n in range(nthresh):
                thresh = .5 * n / nthresh
                edge_feats_vec[n] = 1 * (diff > thresh)
            edge_feats_vec[-1] = 1.0  # add bias feature
            edge_feature_mat[j, :] = edge_feats_vec

        # package up feature matrix as feature dictionary
        feature_vectors = [np.array(x) for x in feature_mat.tolist()]
        feature_dict = dict(zip(pixel_ids, feature_vectors))

        edge_feature_vectors = [np.array(x) for x in edge_feature_mat.tolist()]
        edge_feature_dict = dict(zip(edges, edge_feature_vectors))

        return feature_dict, edge_feature_dict

    @staticmethod
    def calculate_tree_probabilities_snake_shape(width, height):
        """
        Calculate spanning-tree edge appearance probabilities by considering two "snakes" that cover the graph going 
        north-to-south and east-to-west.
        
        :param width: width of grid MRF
        :type width: int
        :param height: height of grid MRF
        :type height: int
        :return: dictionary of edge appearance probabilities under the two-snake spanning tree distribution
        :rtype: dict
        """
        tree_prob = dict()
        for x in range(width):
            for y in range(height - 1):
                if x == 0 or x == width - 1:
                    tree_prob[((x, y), (x, y + 1))] = 0.75
                    tree_prob[((x, y + 1), (x, y))] = 0.75
                else:
                    tree_prob[((x, y), (x, y + 1))] = 0.5
                    tree_prob[((x, y + 1), (x, y))] = 0.5

        for x in range(width - 1):
            for y in range(height):
                if y == 0 or y == height - 1:
                    tree_prob[((x, y), (x + 1, y))] = 0.75
                    tree_prob[((x + 1, y), (x, y))] = 0.75
                else:
                    tree_prob[((x, y), (x + 1, y))] = 0.5
                    tree_prob[((x + 1, y), (x, y))] = 0.5

        return tree_prob
