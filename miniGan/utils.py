import torch
from torch.autograd import Variable


def noise(size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    n = Variable(torch.randn(size, 100))
    return n


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)  # before 1


def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    return data
