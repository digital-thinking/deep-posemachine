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


def calcBoundingBox(points):
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    return np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])


def scaleBB(bb, scale):
    centerX = (bb[0][0] + bb[1][0]) / 2
    centerY = (bb[0][1] + bb[2][1]) / 2
    center = (centerX, centerY)
    scl_center = (centerX * scale[0], centerY * scale[1])

    p1 = scale * (bb[0] - center) + scl_center
    p2 = scale * (bb[1] - center) + scl_center
    p3 = scale * (bb[2] - center) + scl_center
    p4 = scale * (bb[3] - center) + scl_center

    return np.array([p1, p2, p3, p4])


def expandBB(scaledBB, size):
    bbw = np.abs(scaledBB[0][0] - scaledBB[1][0])
    bbh = np.abs(scaledBB[0][1] - scaledBB[2][1])

    expandX = (size[0] - bbw) / 2
    expandY = (size[1] - bbh) / 2

    p1 = scaledBB[0] + (-expandX, -expandY)
    p2 = scaledBB[1] + (+expandX, -expandY)
    p3 = scaledBB[2] + (+expandX, +expandY)
    p4 = scaledBB[3] + (+expandX, +expandY)

    return np.array([p1, p2, p3, p4])


class Mpii(RNGDataFlow):
    """
    Return [image, label],
        image is 28x28 in the range [0,1]
    """
    def __init__(self, train_or_test, shuffle=True, dir=None):

        self.imageDimension = 512
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
        self.boundigBoxes = []

        csv_file = 'train_joints.csv' if train_or_test == 'train' else 'test_joints.csv'

        path = join(dir, csv_file)
        with open(path, 'r') as f:
            for line in f.readlines():
                splitted = line.split(',')
                file_name = splitted[0]

                # 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis,
                # 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist, 12 - r shoulder,
                # 13 - l shoulder, 14 - l elbow, 15 - l wrist

                pointlist = [float(x) for x in splitted[1:]]
                points = np.array(pointlist, dtype=np.int32).reshape((16, 2))
                self.image_paths.append(file_name)
                self.labels.append(points[9])
                self.boundigBoxes.append(calcBoundingBox(points))


        self.reset_state()
        # mat = hdf5storage.loadmat(path)
        #mat = h5py.File(path, 'r')

        #    print '\n'

        # self.train = DataSet(train_images, train_labels)
        #self.test = DataSet(test_images, test_labels)

    def size(self):
        return len(self.image_paths)

    def cropAndResizeImage(self, idx):
        img_path = join(self.image_dir, self.image_paths[idx])
        # downscale
        image = cv2.imread(img_path)
        orgSize = image.shape[:2]
        label = self.labels[idx]
        bb = self.boundigBoxes[idx]
        dim = self.imageDimension / 2
        # define the target height of the bounding box
        targetHeight = 300.0
        w = np.abs(bb[0][0] - bb[1][0])
        h = np.abs(bb[0][1] - bb[2][1])
        targetScale = targetHeight / h

        scaledImage = cv2.resize(image, (0, 0), fx=targetScale, fy=targetScale)
        scaledBB = scaleBB(bb, (targetScale, targetScale))
        cropRegion = expandBB(scaledBB, (self.imageDimension, self.imageDimension))

        startX = cropRegion[0][0] + dim
        startY = cropRegion[0][1] + dim
        endX = cropRegion[2][0] + dim
        endY = cropRegion[2][1] + dim

        padded_image = np.pad(scaledImage, ((dim, dim), (dim, dim), (0, 0)), mode='constant')
        croppedImage = padded_image[int(startY):int(endY), int(startX):int(endX)]

        # new label
        out_labelX = int((label[0] * targetScale - cropRegion[0][0]))
        out_labelY = int((label[1] * targetScale - cropRegion[0][1]))
        out_label = (out_labelX, out_labelY)

        # debug

        cv2.circle(croppedImage, out_label, 10, [255, 255, 255])

        bbp1 = (int(scaledBB[0][0]), int(scaledBB[0][1]))
        bbp2 = (int(scaledBB[2][0]), int(scaledBB[2][1]))

        crop1 = (int(cropRegion[0][0]), int(cropRegion[0][1]))
        crop2 = (int(cropRegion[2][0]), int(cropRegion[2][1]))

        cv2.rectangle(scaledImage, bbp1, bbp2, [255, 255, 255])
        cv2.rectangle(scaledImage, crop1, crop2, [255, 0, 0])

        #
        # # scaleds up the ROI within the image
        # bbscaled = scaleBB(bb, targetScale)
        #
        # #  ROI dimension
        # bbw = np.abs(bbscaled[0][0] - bbscaled[1][0])
        # bbh = np.abs(bbscaled[0][1] - bbscaled[2][1])
        #
        # print bbw, bbh
        #
        # # add padding
        # startX = bbscaled[0][0] + dim
        # startY = bbscaled[0][1] + dim
        # endX = bbscaled[2][0] + dim
        # endY = bbscaled[2][1] + dim
        #
        # # new label
        # out_labelX = int((label[0] - bbscaled[0][0]))
        # out_labelY = int((label[1] - bbscaled[0][1]))
        # out_label = (out_labelX, out_labelY)
        #
        # # debug draw
        # bbcp1 = (int(bbscaled[0][0]), int(bbscaled[0][1]))
        # bbcp2 = (int(bbscaled[2][0]), int(bbscaled[2][1]))
        # bbp1 = (int(bb[0][0]), int(bb[0][1]))
        # bbp2 = (int(bb[2][0]), int(bb[2][1]))
        #
        # cv2.circle(scaledImage, (label[0], label[1]), 10, [255, 255, 255])
        # cv2.rectangle(scaledImage, bbp1, bbp2, [255, 0, 0])
        # cv2.rectangle(scaledImage, bbcp1, bbcp2, [255, 255, 0])
        #
        # padded_image = np.pad(scaledImage, ((dim, dim), (dim, dim), (0, 0)), mode='constant')
        # croppedImage = padded_image[int(startY):int(endY), int(startX):int(endX)]
        #
        # cv2.circle(scaledImage, out_label, 10, [0, 0, 255])
        # # np.lib.pad(croppedImage, (int(self.imageDimension), int(self.imageDimension)), 'constant', constant_values=(0))


        return [croppedImage, out_label]

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.cropAndResizeImage(k)

if __name__ == '__main__':
    ds = Mpii('train', dir='mpii')
    for (img, label) in ds.get_data():
        coord = (int(label[0]), int(label[1]))
        #cv2.circle(img, coord, 10, [255, 0, 0])
        cv2.imshow('test', img)
        cv2.waitKey(1000)
