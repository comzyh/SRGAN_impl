import os.path
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops
from tensorflow.python.framework.errors_impl import OutOfRangeError

from model import SRResNet
from model import SRGAN_discriminator
from model import srresnet_preprocess
from model import srresnet_postprocess



def parse_record(record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
    return image


def get_dataset(tfrecord_dir, setname):
    # filenames = tf.placeholder(tf.string, shape=[None])
    filenames = [os.path.join(tfrecord_dir, setname)]
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(parse_record, num_parallel_calls=8)
    dataset = dataset.filter(lambda image: tf.reduce_all(tf.greater_equal(tf.shape(image), (96, 96, 3))))

    def croper(image):
        hr_images = tf.random_crop(image, (96, 96, 3))
        lr_images = tf.image.resize_images(
            hr_images, (48, 48),
            method=tf.image.ResizeMethod.BICUBIC)  # ResizeMethod.BILINEAR
        return hr_images, lr_images
    dataset = dataset.map(croper)
    return dataset


def evalue(datapath):
    dataset = get_dataset(datapath, 'validation')
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        # def get_next_figure():
        #     hr_image, lr_image = sess.run(next_element)

        #     def get_input_fn(lr_image):
        #         def input_fn():
        #             return {'lr_images': tf.constant(np.expand_dims(lr_image, 0), dtype=tf.float32)}
        #         return input_fn
        #     result = next(srresnet.predict(input_fn=get_input_fn(lr_image)))
        #     # result_image = np.clip(result['sr_images'], 0, 255).astype(np.uint8)
        #     result_image = result['sr_images']
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(hr_image)
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(result_image)
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(np.clip(lr_image, 0, 255).astype(np.uint8))
        #     return hr_image, lr_image, result['sr_images']
        import IPython
        IPython.embed()


def main():
    parser = argparse.ArgumentParser(description='train SRGAN')
    parser.add_argument('--datapath', type=str, required=True, help='location of SR dataset tfrecords')
    parser.add_argument('--model_dir', type=str, default='/tmp/SRResNet', help='directory to save model')
    parser.add_argument('--evalue', action='store_true', help='evalue model')
    parser.add_argument('--bs', type=int, default=64, help='batch size')

    args = parser.parse_args()

    batch_size = args.bs

    if args.evalue:
        return evalue()

    train_dataset = get_dataset(args.datapath, 'train')  # .shuffle(1000)
    train_dataset = train_dataset.batch(batch_size).prefetch(2)
    train_iterator = train_dataset.make_initializable_iterator()

    hr_images, lr_images = train_iterator.get_next()

    # build graph

    # hr_images_ph = tf.placeholder(dtype=tf.uint8, shape=(None, 96, 96, 3))
    # lr_images_ph = tf.placeholder(dtype=tf.float32, shape=(None, 48, 48, 3))

    hr_images = srresnet_preprocess(hr_images)
    lr_images = srresnet_preprocess(lr_images)

    # SRResNet
    sr_images = SRResNet(lr_images, training=True, reuse=True)

    # Discriminator
    d_logits_fake, _ = SRGAN_discriminator(sr_images, training=True, reuse=False)
    d_logits_real, _ = SRGAN_discriminator(hr_images, training=True, reuse=True)

    # loss to train discriminator
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_logits_fake), logits=d_logits_fake,
        name='d_loss_fake'
    )
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_logits_real), logits=d_logits_real,
        name='d_loss_real'
    )

    d_loss = d_loss_fake + d_loss_real

    # loss to train SRResNet
    g_mse_loss = tf.losses.mean_squared_error(labels=hr_images, predictions=sr_images)

    g_gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        # we want gentrator generate images that looks realistic
        # so labels = 1
        labels=tf.ones_like(d_logits_fake), logits=d_logits_fake,
        name='g_gan_loss'
    )
    g_loss = g_mse_loss + 0.001 * g_gan_loss

    # varlist
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRResNet')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_D')
    p_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='SRResNet')
    g_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='SRResNet')
    d_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='SRGAN_D')

    # optimize
    lr_all = tf.constant(0.00001, dtype=tf.float32)
    p_optim = tf.train.AdamOptimizer(lr_all, beta1=0.9).minimize(g_mse_loss, var_list=g_vars)
    g_optim = tf.train.AdamOptimizer(lr_all, beta1=0.9).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_all, beta1=0.9).minimize(d_loss, var_list=d_vars)

    saver = tf.train.Saver()
    model_path = os.path.join(args.model_dir, 'model')

    # start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=config) as sess:
        # sess.run()

        try:
            saver.restore(sess, model_path)
        except Exception:
            print('initialize parameters')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        # import IPython
        # IPython.embed()

        # sess.run(sr_images)

        for epoch in range(40):
            print("epoch {}".format(epoch))
            sess.run(train_iterator.initializer)
            while True:
                try:
                    a = sess.run([p_optim, g_mse_loss, tf.shape(lr_images)])
                    print(a)
                except OutOfRangeError:
                    break
            saver.save(sess, model_path, global_step=tf.train.get_global_step())


if __name__ == '__main__':
    main()
