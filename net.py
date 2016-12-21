import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack import *
from tensorpack.callbacks.dump import DumpParamAsImage
from tensorpack.tfutils.summary import *
import tensorflow as tf
import numpy as np

sigma = 32.0
BATCH_SIZE = None


def pdf_debug_img(name, float_image, sigma):
    # max_val = 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi))
    max_val = tf.reduce_max(float_image)
    float_image = tf.maximum(float_image, np.float(0))
    debug = tf.cast(255 * (float_image / max_val), tf.uint8)
    return tf.image_summary(name, debug, max_images=5)


def gaussian_image(label):
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
                InputVar(tf.int32, [BATCH_SIZE, 2], 'label')
                ]

    def _build_graph(self, input_vars, is_training):
        image, label = input_vars
        if is_training:
            tf.image_summary("train_image", image, 10)

        pred = LinearWrap(image) \
            .Conv2D('conv1.1', out_channel=64, kernel_shape=9) \
            .MaxPooling('pool1', 2, stride=2, padding='SAME') \
            .Conv2D('conv2.1', out_channel=128, kernel_shape=9) \
            .MaxPooling('pool2', 2, stride=2, padding='SAME') \
            .Conv2D('conv3.1', out_channel=256, kernel_shape=9) \
            .MaxPooling('pool3', 2, stride=2, padding='SAME') \
            .Conv2D('conv4.1', out_channel=512, kernel_shape=5) \
            .Conv2D('conv4.2', out_channel=512, kernel_shape=9) \
            .Conv2D('conv4.3', out_channel=512, kernel_shape=1) \
            .Conv2D('conv4.4', out_channel=1, kernel_shape=1, nl=tf.identity)()

        """
        pred = LinearWrap(image) \
            .MaxPooling('pool1', 2, stride=2, padding='SAME') \
            .MaxPooling('pool2', 2, stride=2, padding='SAME') \
            .MaxPooling('pool3', 2, stride=2, padding='SAME') \
            .Conv2D('conv4.4', out_channel=1, kernel_shape=1, nl=tf.identity)()
        """

        # debug_pred = 1.0 / (np.sqrt(2 * (sigma ** 2) * np.pi)) * tf.exp(-pred)
        # pred = tf.reshape(tf.nn.softmax(tf.reshape(pred,[5,64*64])),[5,64,64,1])
        belief_maps_output = tf.identity(pred, "belief_maps_output")
        pdf_debug_img('pred', pred, sigma)

        gaussian = gaussian_image(label)
        # diff = (pred - gaussian)
        # dbg = tf.reduce_sum(tf.to_float(tf.is_nan(gaussian)))
        cost = tf.squared_difference(pred, gaussian, name='l2_norm')
        pdf_debug_img('cost', cost, sigma)

        cost = tf.reduce_mean(cost, name='mse')


        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbf.rms(gaussian - pred, name='wrong')
        # monitor training error
        #add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.000001,
                         regularize_cost('conv.*/W', tf.nn.l2_loss),
                         name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
