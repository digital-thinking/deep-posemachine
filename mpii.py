#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import gzip
import random
import numpy
from six.moves import urllib, range

from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['Mpii']

""" This file is mostly copied from tensorflow example """

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        logger.info("Downloading mnist data to {}...".format(filepath))
        download(SOURCE_URL + filename, work_directory)
    return filepath

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
               'Invalid magic number %d in MNIST image file: %s' %
               (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
              'Invalid magic number %d in MNIST label file: %s' %
              (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        """Construct a DataSet. """
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

class Mpii(RNGDataFlow):
    """
    Return [image, label],
        image is 28x28 in the range [0,1]
    """
    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test: string either 'train' or 'test'
        """
        #dir = 'data/mpii'
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

        local_file = maybe_download(TRAIN_IMAGES, dir)
        train_images = extract_images(local_file)

        local_file = maybe_download(TRAIN_LABELS, dir)
        train_labels = extract_labels(local_file)

        local_file = maybe_download(TEST_IMAGES, dir)
        test_images = extract_images(local_file)

        local_file = maybe_download(TEST_LABELS, dir)
        test_labels = extract_labels(local_file)

        self.train = DataSet(train_images, train_labels)
        self.test = DataSet(test_images, test_labels)

    def size(self):
        ds = self.train if self.train_or_test == 'train' else self.test
        return ds.num_examples

    def get_data(self):
        ds = self.train if self.train_or_test == 'train' else self.test
        idxs = list(range(ds.num_examples))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = ds.images[k].reshape((28, 28))
            label = ds.labels[k]
            yield [img, label]

if __name__ == '__main__':
    ds = Mpii('train')
    for (img, label) in ds.get_data():
        from IPython import embed; embed()
        break
