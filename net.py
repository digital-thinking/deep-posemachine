import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack import *
from tensorpack.tfutils.summary import *


class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 128, 128, 3], 'input'),
                InputVar(tf.float32, [None, 2], 'label')
                ]

    def _build_graph(self, input_vars, is_training):
        image, label = input_vars
        if is_training:
            tf.image_summary("train_image", image, 10)

        with argscope(Conv2D, nl=BNReLU(is_training), use_bias=False, kernel_shape=3):
            pred = LinearWrap(image) \
                .Conv2D('conv1.1', out_channel=64) \
                .Conv2D('conv1.2', out_channel=64) \
                .MaxPooling('pool1', 3, stride=2, padding='SAME') \
                .Conv2D('conv2.1', out_channel=128) \
                .Conv2D('conv2.2', out_channel=128) \
                .MaxPooling('pool2', 3, stride=2, padding='SAME') \
                .Conv2D('conv3.1', out_channel=256) \
                .Conv2D('conv3.2', out_channel=256) \
                .MaxPooling('pool3', 3, stride=2, padding='SAME') \
                .Conv2D('conv4.1', out_channel=512) \
                .Conv2D('conv4.2', out_channel=512) \
                .FullyConnected('fc0', 512,
                                b_init=tf.constant_initializer(0.1)) \
                .FullyConnected('fc1', 512,
                                b_init=tf.constant_initializer(0.1)) \
                .FullyConnected('linear', out_dim=2, nl=tf.identity)()

        # cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.squared_difference(pred, label, name='squared_difference')
        cost = tf.reduce_mean(cost, name='mse')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbf.rms(label - pred, name='wrong')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.004,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
