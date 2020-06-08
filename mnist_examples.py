import logging
import os
import pickle
import sys
from functools import partial

import jax
import numpy as onp
import scipy.io

from examples import (find_starting_point,
                      find_starting_point_likely_class_strategy,
                      find_starting_point_simple_strategy)
from staxmod import Conv, Dense, Flatten, MaxPool, Relu, serial
from utils import is_device_array


def MadryCNN():
    return serial(Conv(32, (5, 5), padding='SAME'), Relu,
                  MaxPool((2, 2), strides=(2, 2), padding='VALID'),
                  Conv(64, (5, 5), padding='SAME'), Relu,
                  MaxPool((2, 2), strides=(2, 2), padding='VALID'), Flatten,
                  Dense(1024), Relu, Dense(10))


def load_madry_params(load_from):
    mapping = [
        ("A0", "A1"),
        (),
        (),
        ("A2", "A3"),
        (),
        (),
        (),
        ("A4", "A5"),
        (),
        ("A6", "A7"),
    ]
    w = scipy.io.loadmat(load_from)
    params = []
    for map_sublist in mapping:
        if len(map_sublist) == 0:
            params.append(())
        else:
            params.append((w[map_sublist[0]], w[map_sublist[1]]))
    return jax.tree_map(jax.device_put, params)


def get_madry_example(name):
    weights = {
        'mnist_madry_plain': './models_mnist/mnist_weights_plain.mat',
        'mnist_madry_linf': './models_mnist/mnist_weights_linf.mat',
        'mnist_madry_l2': './models_mnist/mnist_weights_l2.mat',
    }
    init, predict = MadryCNN()
    output_shape, _ = init((-1, 32, 32, 3))
    params = load_madry_params(weights[name])
    n_classes = output_shape[-1]
    return n_classes, predict, params
