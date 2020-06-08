#!/usr/bin/env python3
import argparse
import logging
import os
import pickle

from lr_attack import run
from mnist_data import load_mnist
from mnist_examples import get_example


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help='name of the model to attack')
    parser.add_argument('--working_dir', help='working directory')
    parser.add_argument('--image', type=int, default=0)
    parser.add_argument('--accuracy', action='store_true', help='first determines the accuracy of the model')
    parser.add_argument('--save', type=str, default=None, help='filename to save result to')

    # hyperparameters
    parser.add_argument('--regions', type=int, default=400)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--gamma', type=int, default=6, help='hyperparam of region selection')
    parser.add_argument('--misc-factor', type=float, default=75.)

    # advanced control over certain aspects (only if you know what you are doing)
    parser.add_argument('--nth-likely-class-starting-point', type=int, default=None)
    parser.add_argument('--no-line-search', action='store_true')
    parser.add_argument('--max-other-classes', type=int, default=None)
    parser.add_argument('--no-normalization', action='store_true')

    args = parser.parse_args()

    # get logger
    logger = logging.getLogger()
    [logger.removeHandler(handler) for handler in logger.handlers]
    file_hndl = FileHandler(os.path.join(FLAGS.working_dir, 'tensorflow.log'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)
    cmd_hndl = StreamHandler()
    cmd_hndl.setLevel(logging.INFO)
    cmd_hndl.setFormatter(Formatter('%(message)s'))
    logger.addHandler(cmd_hndl)
    logger.setLevel(logging.DEBUG)

    if args.save is not None:
        if os.path.exists(args.save):
            logging.warning(f'not runnning because results already exist: {args.save}')
            return

    n_classes, predict, params = get_example(args.model)
    _, _, test_ds = load_mnist()
    test_images, test_labels = test_ds
    result = run(n_classes, predict, params, test_images, test_labels, args=args)

    if args.save is not None:
        directory = os.path.dirname(args.save)
        if len(directory) > 0 and not os.path.exists(directory):
            os.makedirs(directory)
        with open(args.save, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    main()
