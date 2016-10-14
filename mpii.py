#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

# import hdf5storage


from os.path import join

import cv2
import numpy as np

# import h5py

from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['Mpii']

""" This file is mostly copied from tensorflow example """

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

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
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
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

        #np.set_printoptions(threshold=np.nan)

        #dir = 'data/mpii'
        self.train_or_test = train_or_test
        self.shuffle = shuffle
        self.image_dir = join(dir, 'images')

        self.image_paths = []
        self.labels = []

        csv_file = 'train_joints.csv' if train_or_test == 'train' else 'test_joints.csv'

        path = join(dir, csv_file)
        with open(path, 'r') as f:
            for line in f.readlines():
                splitted = line.split(',')
                file_name = splitted[0]
                ptx = float(splitted[1])
                pty = float(splitted[2])
                self.image_paths.append(file_name)
                self.labels.append((ptx, pty))

        print self.image_paths
        print self.labels

        self.reset_state()
        # mat = hdf5storage.loadmat(path)
        #mat = h5py.File(path, 'r')

        #    print '\n'

        # self.train = DataSet(train_images, train_labels)
        #self.test = DataSet(test_images, test_labels)

    def size(self):
        return len(self.image_paths)

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_path = join(self.image_dir, self.image_paths[k])
            image = cv2.imread(img_path)
            old_size = image.shape
            warped_image = cv2.resize(image, (128, 128))
            label = self.labels[k]
            label_x = label[0] * 128.0 / old_size[1]
            label_y = label[1] * 128.0 / old_size[0]
            out_label = np.array([label_x, label_y], dtype=np.float32)
            yield [warped_image, out_label]

if __name__ == '__main__':
    ds = Mpii('train', dir='mpii')
    for (img, label) in ds.get_data():
        coord = (int(label[0]), int(label[1]))
        cv2.circle(img, coord, 10, [255, 0, 0])
        cv2.imshow('test', img)

        print img.shape
        print label

        cv2.waitKey(1000)
