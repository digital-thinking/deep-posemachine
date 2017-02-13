import numpy as np
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

        # tf.image_summary("train_image", image, 10)
        gaussian = gaussian_image(label)

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
            l = tf.concat(3, [l, shared])
            for i in range(1, 6):
                l = Conv2D('Mconv{}_stage{}'.format(i, stage), l, 128, kernel_shape=7)
            l = Conv2D('Mconv6_stage{}'.format(stage), l, 128, kernel_shape=1)
            l = Conv2D('Mconv7_stage{}'.format(stage), l, BODY_PART_COUNT, kernel_shape=1, nl=tf.identity)
            pred = tf.transpose(l, perm=[0, 3, 1, 2])
            pred = tf.reshape(pred, [-1, 46, 46, 1])
            error = tf.squared_difference(pred, gaussian, name='se_{}'.format(stage))
            return l, error

        belief = (LinearWrap(shared)
                  .Conv2D('conv5_1_CPM', 512, kernel_shape=1)
                  .Conv2D('conv5_2_CPM', BODY_PART_COUNT, kernel_shape=1, nl=tf.identity)())
        transposed = tf.transpose(belief, perm=[0, 3, 1, 2])

        se_calc = tf.reshape(transposed, [-1, 46, 46, 1])
        error = tf.squared_difference(se_calc, gaussian, name='se_{}'.format(1))

        for i in range(2, 7):
            belief, e = add_stage(i, belief)
            error = error + e

        belief = tf.image.resize_bilinear(belief, [368, 368], name='resized_map')

        # validation error
        pred_collapse = tf.reshape(se_calc, [-1, 46 * 46])

        flatIndex = tf.argmax(pred_collapse, 1, name="flatIndex")
        predCordsX = tf.reshape((flatIndex % 46) * 8, [-1, 1])
        predCordsY = tf.reshape((flatIndex / 46) * 8, [-1, 1])
        predCordsYX = tf.concat(1, [predCordsY, predCordsX])
        predCords = tf.cast(tf.reshape(predCordsYX, [-1, 16, 2]), tf.int32, name='debug_cords')

        euclid_distance = tf.sqrt(tf.cast(tf.reduce_sum(tf.square(
            tf.sub(predCords, label)), 2), dtype=tf.float32), name="euclid_distance")

        minradius = tf.constant(25.0, dtype=tf.float32)
        incircle = 1 - tf.sign(tf.cast(euclid_distance / minradius, tf.int32))
        pcp = tf.reduce_mean(tf.cast(incircle, tf.float32), name="train_pcp")
        add_moving_summary(pcp)

        belief_maps_output = tf.identity(belief, "belief_maps_output")
        cost = tf.reduce_mean(error, name='mse')

        wrong = tf.identity(1 - pcp, 'error')


        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.000001,
                         regularize_cost('conv.*/W', tf.nn.l2_loss),
                         name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
