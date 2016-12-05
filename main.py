import argparse

from tensorpack import *

from mpii import Mpii
from net import Model

BATCH_SIZE = 1


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = Mpii(train_or_test, dir='mpii', shuffle=False)

    if isTrain:
        augmentors = [
            imgaug.RandomCrop((30, 30)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((30, 30)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    # ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    # if isTrain:
    #    ds = PrefetchData(ds, 3, 2)
    return ds


def get_config():
    # logger.auto_set_dir('d')
    # overwrite existing
    logger.set_logger_dir('mpii/log', 'k')

    # prepare dataset
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    # lr = tf.train.exponential_decay(
    #    learning_rate=1e-2,
    #    global_step=get_global_step_var(),
    #    decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
    #    decay_rate=0.5, staircase=True, name='learning_rate')

    # tf.train.
    lr = tf.Variable(1E-3, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test, ClassificationError()),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 1E-3), (30, 1E-4), (60, 1E-5)])
            #DumpParamAsImage('pdf_label')

        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=250,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        # QueueInputTrainer(config).train()
        SimpleTrainer(config).train()
