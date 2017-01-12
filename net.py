import numpy as np
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import *

sigma = 32.0
BATCH_SIZE = None
BODY_PART_COUNT = 16


def pdf_debug_img(name, float_image, sigma):
    # max_val = 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi))
    max_val = tf.reduce_max(float_image)
    float_image = tf.maximum(float_image, np.float(0))
    debug = tf.cast(255 * (float_image / max_val), tf.uint8)
    return tf.image_summary(name, debug, max_images=5)


def gaussian_image(label):
    label = tf.reshape(label, [-1, 2])
    indices = np.indices([368, 368])[:, ::8, ::8].astype(np.float32)
    coords = tf.constant(indices)
    stretch = tf.reshape(tf.to_float(label), [-1, 2, 1, 1])
    stretch = tf.tile(stretch, [1, 1, 46, 46])
    # pdf = 1.0/(np.sqrt(2*(sigma**2)*np.pi)) * tf.exp(-tf.pow(coords-stretch,2)/(2*sigma**2))
    pdf = tf.pow(coords - stretch, 2) / (2 * sigma ** 2)
    pdf = tf.reduce_sum(pdf, [1])
    # pdf = tf.reduce_prod(pdf,[1])
    # print debug
    pdf = tf.expand_dims(pdf, 3)
    debug = tf.exp(-pdf)  # 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi)) *
    pdf_debug_img('super', debug, sigma)

    return debug


class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()

    def _get_input_vars(self):
        return [InputVar(tf.float32, [BATCH_SIZE, 368, 368, 3], 'input'),
                InputVar(tf.int32, [BATCH_SIZE, BODY_PART_COUNT, 2], 'label')
                ]

    def _build_graph(self, input_vars, is_training):
        image, label = input_vars
        if is_training:
            tf.image_summary("train_image", image, 10)

            # shared = LinearWrap(image) \
            # .Conv2D('conv1.1', out_channel=64, kernel_shape=9) \
            # .MaxPooling('pool1', 2, stride=2, padding='SAME') \
            # .Conv2D('conv2.1', out_channel=128, kernel_shape=9) \
            # .MaxPooling('pool2', 2, stride=2, padding='SAME') \
            # .Conv2D('conv3.1', out_channel=256, kernel_shape=9) \
            # .MaxPooling('pool3', 2, stride=2, padding='SAME') \
            # .Conv2D('conv4.1', out_channel=512, kernel_shape=5) \
            # .Conv2D('conv4.2', out_channel=512, kernel_shape=9) \
            # .Conv2D('conv4.3', out_channel=512, kernel_shape=1) \
            # .Conv2D('conv4.4', out_channel=BODY_PART_COUNT, kernel_shape=1, nl=tf.identity)()

            shared = (LinearWrap(image)
                      .Conv2D('conv1_1', 64, kernel_shape=3)
                      .Conv2D('conv1_2', 64, kernel_shape=3)
                      .MaxPooling('pool1', 2)
                      # 184
                      .Conv2D('conv2_1', 128, kernel_shape=3)
                      .Conv2D('conv2_2', 128, kernel_shape=3)
                      .MaxPooling('pool2', 2)
                      # 92
                      .Conv2D('conv3_1', 256, kernel_shape=3)
                      .Conv2D('conv3_2', 256, kernel_shape=3)
                      .Conv2D('conv3_3', 256, kernel_shape=3)
                      .Conv2D('conv3_4', 256, kernel_shape=3)
                      .MaxPooling('pool3', 2)
                      # 46
                      .Conv2D('conv4_1', 512, kernel_shape=3)
                      .Conv2D('conv4_2', 512, kernel_shape=3)
                      .Conv2D('conv4_3_CPM', 256, kernel_shape=3)
                      .Conv2D('conv4_4_CPM', 256, kernel_shape=3)
                      .Conv2D('conv4_5_CPM', 256, kernel_shape=3)
                      .Conv2D('conv4_6_CPM', 256, kernel_shape=3)
                      .Conv2D('conv4_7_CPM', 128, kernel_shape=3)())

            def add_stage(stage, l):
                l = tf.concat(0, [l, shared])
                for i in range(1, 6):
                    l = Conv2D('Mconv{}_stage{}'.format(i, stage), l, 128, kernel_shape=7)
                    l = Conv2D('Mconv6_stage{}'.format(stage), l, 128, kernel_shape=1)
                    l = Conv2D('Mconv7_stage{}'.format(stage), l, BODY_PART_COUNT, kernel_shape=1, nl=tf.identity)
                return l

            out1 = (LinearWrap(shared)
                    .Conv2D('conv5_1_CPM', 512, kernel_shape=1)
                    .Conv2D('conv5_2_CPM', 15, kernel_shape=1, nl=tf.identity)())
            out2 = add_stage(2, out1)
            out3 = add_stage(3, out2)
            out4 = add_stage(4, out3)
            out5 = add_stage(5, out4)
            out6 = add_stage(6, out4)
            pred = tf.image.resize_bilinear(out6, [368, 368], name='resized_map')

        # debug_pred = 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi)) * tf.exp(-pred)
        # pred = tf.reshape(tf.nn.softmax(tf.reshape(pred,[5,64*64])),[5,64,64,1])
        belief_maps_output = tf.identity(pred, "belief_maps_output")
        pred = tf.transpose(pred, perm=[0, 3, 1, 2])
        pred = tf.reshape(pred, [-1, 46, 46, 1])
        # pdf_debug_img('pred', pred, sigma)


        gaussian = gaussian_image(label)
        # diff = (pred - gaussian)
        # dbg = tf.reduce_sum(tf.to_float(tf.is_nan(gaussian)))
        cost = tf.squared_difference(pred, gaussian, name='l2_norm')
        # pdf_debug_img('cost', cost, sigma)

        cost = tf.reduce_mean(cost, name='mse')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = tf.identity(cost, name='wrong')
        # monitor training error
        # add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.000001,
                         regularize_cost('conv.*/W', tf.nn.l2_loss),
                         name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
