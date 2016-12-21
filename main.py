import argparse

from tensorpack import *

from mpii import Mpii
from net import Model
import cv2
import numpy as np

BATCH_SIZE = 10


def get_data(train_or_test, batch_size=10):
    isTrain = train_or_test == 'train'
    ds = Mpii(train_or_test, dir='mpii', shuffle=True)

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
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_config():
    # logger.auto_set_dir('d')
    # overwrite existing
    logger.set_logger_dir('mpii/log', 'k')

    # prepare dataset
    dataset_train = get_data('train', BATCH_SIZE)
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test', BATCH_SIZE)

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
                                      [(1, 1E-3), (100, 1E-4), (200, 1E-5)])
            #DumpParamAsImage('pdf_label')

        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=250,
    )


def get_pred_config(weights_path, input_var_names=['input', 'label'], output_names=['belief_maps_output']):
    loaded_model = SaverRestore(weights_path)
    config = PredictConfig(
        model=Model(),
        output_var_names=output_names,
        input_var_names=input_var_names,  #
        session_init=loaded_model,
        return_input=True
    )
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--test', help='Run in test mode?', default='0')
    args = parser.parse_args()

    test = args.test == '1'

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

        if not test:
            QueueInputTrainer(config).train()
        else:
            config = get_pred_config(args.load, input_var_names=["input", "label"])  # 'label'
            pred = SimpleDatasetPredictor(config, get_data('test', 1))
            i = 0
            for input, output in pred.get_result():
                raw_img = input[0][0]
                label = input[1][0]
                believe = output[0][0, :, :, 0]
                believe = cv2.resize(believe, (0, 0), fx=8, fy=8)

                believe = believe - believe.min()
                believe = believe / believe.max()
                believe = np.uint8(255 * believe)
                believe = np.dstack([believe, believe, believe])

                # (raw_img, label)
                img = 255.0 * ((raw_img + 1) / 2.0)
                img = np.uint8(img)
                coord = (int(label[1]), int(label[0]))
                cv2.circle(believe, coord, 10, [255, 0, 0])
                cv2.imwrite('mpii/results/in_%d.png' % i, img)
                cv2.imwrite('mpii/results/out_%d.png' % i, believe)
                i = i + 1
                cv2.waitKey(1000)
                # SimpleTrainer(config).train()
