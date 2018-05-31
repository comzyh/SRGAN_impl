import os.path
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

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
    # filenames = [os.path.join(tfrecord_dir, setname)]
    filenames = tf.data.TFRecordDataset.list_files(os.path.join(tfrecord_dir, setname + '*'))
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
    import matplotlib.pyplot as plt
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


def get_psnr_from_mse(mse, max_val, name='psnr'):
    psnr = math_ops.subtract(
        20 * math_ops.log(max_val) / math_ops.log(10.0),
        np.float32(10 / np.log(10)) * math_ops.log(mse),
        name=name)
    return psnr


def main():
    parser = argparse.ArgumentParser(description='train SRGAN')
    parser.add_argument('--datapath', type=str, required=True, help='location of SR dataset tfrecords')
    parser.add_argument('--model_dir', type=str, default='/tmp/SRResNet', help='directory to save model')
    parser.add_argument('--evalue', action='store_true', help='evalue model')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--repeat', type=int, default=10, help='train repeat time')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--gan', action='store_true', help='using GAN')

    args = parser.parse_args()

    batch_size = args.bs

    if args.evalue:
        return evalue()

    train_dataset = get_dataset(args.datapath, 'train').repeat(count=args.repeat)  # .shuffle(1000)
    train_dataset = train_dataset.batch(batch_size).prefetch(2)

    valid_dataset = get_dataset(args.datapath, 'validation')
    valid_dataset = valid_dataset.batch(batch_size).prefetch(2)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    hr_images, lr_images = iterator.get_next()

    train_iterator = train_dataset.make_initializable_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()

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
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_logits_fake), logits=d_logits_fake,
        name='d_loss_fake'
    ))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_logits_real), logits=d_logits_real,
        name='d_loss_real'
    ))

    d_loss = d_loss_fake + d_loss_real

    # loss to train SRResNet
    g_mse_loss = tf.losses.mean_squared_error(labels=hr_images, predictions=sr_images)

    g_gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        # we want gentrator generate images that looks realistic
        # so labels = 1
        labels=tf.ones_like(d_logits_fake), logits=d_logits_fake,
        name='g_gan_loss'
    ))
    g_loss = g_mse_loss + 0.001 * g_gan_loss

    # lr
    lr_all = tf.constant(args.lr, dtype=tf.float32)

    # summary
    # currnet_bs = tf.shape(lr_images)[0]  # this batch size
    # metric_g_mse_loss = tf.metric.mean(g_mse_loss, weights=currnet_bs)
    # metric_psnr = tf.metric.mean(, weights=currnet_bs)
    psnr = get_psnr_from_mse(g_mse_loss, 2.0)
    tf.summary.scalar('mse_loss', g_mse_loss, collections=['gan', 'train', 'valid'])
    tf.summary.scalar('psnr', psnr, collections=['gan', 'train', 'valid'])
    tf.summary.scalar('lr', lr_all, collections=['gan', 'train'])
    tf.summary.scalar('g_gan_loss', g_gan_loss, collections=['gan'])
    tf.summary.scalar('g_loss', g_loss, collections=['gan'])
    tf.summary.scalar('d_loss_real', d_loss_real, collections=['gan'])
    tf.summary.scalar('d_loss_fake', d_loss_fake, collections=['gan'])
    tf.summary.scalar('d_loss', d_loss, collections=['gan'])

    # varlist
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRResNet')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_D')
    p_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='SRResNet')
    g_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='SRResNet')
    d_train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='SRGAN_D')

    # global_step
    global_step = tf.train.get_or_create_global_step()

    # optimize
    p_optim = tf.train.AdamOptimizer(lr_all).minimize(g_mse_loss, var_list=g_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(lr_all).minimize(g_loss, var_list=g_vars, global_step=global_step)
    d_optim = tf.train.AdamOptimizer(lr_all).minimize(d_loss, var_list=d_vars, global_step=global_step)

    p_train_op.append(p_optim)
    g_train_op.append(g_optim)
    d_train_op.append(d_optim)

    p_train_op = tf.group(*p_train_op)
    g_train_op = tf.group(*g_train_op)
    d_train_op = tf.group(*d_train_op)
    # saver
    saver = tf.train.Saver(max_to_keep=3)
    model_path = os.path.join(args.model_dir, 'model')

    # start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        using_gan = args.gan
        # summary writer
        if using_gan:
            train_summary = tf.summary.merge_all('gan')
        else:
            train_summary = tf.summary.merge_all('train')

        vaild_summary = tf.summary.merge_all('valid')

        train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'train'), sess.graph)
        vaild_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'vaild'))

        # train and vaild handle
        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())

        # resore or initialize weights
        model_file = tf.train.latest_checkpoint(args.model_dir)
        if model_file:
            saver.restore(sess, model_file)
        else:
            print('initialize parameters')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        # train dict
        train_dict = {
            'global_step': global_step,
            'g_mse_loss': g_mse_loss,
            'train_summary': train_summary,
            'psnr': psnr,
        }
        if using_gan:
            train_dict.update({
                'g_train_op': g_train_op,
                'd_train_op': d_train_op,
            })
        else:
            train_dict.update({
                'p_train_op': p_train_op,
            })
        for epoch in range(args.epoch):

            print("epoch {}, using_gan: {}".format(epoch, using_gan))
            # train
            sess.run(train_iterator.initializer)
            while True:
                try:
                    result = sess.run(train_dict, feed_dict={handle: train_handle})
                    train_writer.add_summary(result['train_summary'], result['global_step'])
                    print(result['g_mse_loss'], result['psnr'])
                except tf.errors.OutOfRangeError:
                    break
            vaild_writer.flush()
            saver.save(sess, model_path, global_step=global_step)

            # vaildation
            sess.run(valid_iterator.initializer)
            while True:
                try:
                    result = sess.run({
                        'g_mse_loss': g_mse_loss,
                        'vaild_summary': vaild_summary,
                        'global_step': global_step,

                    }, feed_dict={handle: valid_handle})
                    vaild_writer.add_summary(result['vaild_summary'], result['global_step'])

                    print('valid', result['g_mse_loss'])
                except tf.errors.OutOfRangeError:
                    break
            vaild_writer.flush()


if __name__ == '__main__':
    main()
