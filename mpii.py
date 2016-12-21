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

        self.imageDimension = 368
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

        csv_file = 'train_joints.csv' if train_or_test == 'train' else 'test_joints.csv'  # test

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

    def size(self):
        return len(self.image_paths)
        # return 1

    def cropAndResizeImage(self, idx):
        path = self.image_paths[idx]
        img_path = join(self.image_dir, self.image_paths[idx])
        # downscale
        image = cv2.imread(img_path)
        orgSize = image.shape[:2]
        label = self.labels[idx]
        bb = self.boundigBoxes[idx]
        dim = self.imageDimension / 2
        # define the target height of the bounding box
        targetHeight = 200.0
        w = np.abs(bb[0][0] - bb[1][0])
        h = np.abs(bb[0][1] - bb[2][1])
        targetScale = targetHeight / h

        scaledImage = cv2.resize(image, (0, 0), fx=targetScale, fy=targetScale)
        scaledBB = scaleBB(bb, (targetScale, targetScale))
        cropRegion = expandBB(scaledBB, (self.imageDimension, self.imageDimension))

        startX = int(cropRegion[0][0] + dim)
        startY = int(cropRegion[0][1] + dim)
        endX = startX + self.imageDimension  # cropRegion[2][0] + dim
        endY = startY + self.imageDimension  #cropRegion[2][1] + dim

        padded_image = np.pad(scaledImage, ((dim, dim), (dim, dim), (0, 0)), mode='constant')
        croppedImage = padded_image[startY:endY, startX:endX]

        # new label
        out_labelX = int((label[0] * targetScale - cropRegion[0][0]))
        out_labelY = int((label[1] * targetScale - cropRegion[0][1]))
        out_label = np.array([out_labelY, out_labelX])

        # debug

        #cv2.circle(croppedImage, (out_label[1], out_label[0]), 10, [255, 255, 255])

        bbp1 = (int(scaledBB[0][0]), int(scaledBB[0][1]))
        bbp2 = (int(scaledBB[2][0]), int(scaledBB[2][1]))

        crop1 = (int(cropRegion[0][0]), int(cropRegion[0][1]))
        crop2 = (int(cropRegion[2][0]), int(cropRegion[2][1]))

        # cv2.rectangle(scaledImage, bbp1, bbp2, [255, 255, 255])
        #cv2.rectangle(scaledImage, crop1, crop2, [255, 0, 0])
        # print croppedImage.shape
        # print out_label.shape
        result_img = 2.0 * croppedImage / 255.0 - 1.0
        return [result_img.astype(np.float32), out_label]

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
        cv2.circle(img, coord, 10, [255, 0, 0])
        cv2.imshow('test', img)
        cv2.waitKey(1000)
