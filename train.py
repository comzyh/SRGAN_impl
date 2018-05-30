import os.path
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops


from model import SRResNet


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


def model_fn(features, labels, mode):
    trainig = (mode == tf.estimator.ModeKeys.TRAIN)
    lr_images = features['lr_images']
    lr_images = tf.to_float(lr_images) / 127.5 - 1

    x = SRResNet(lr_images, training=trainig)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"sr_images": tf.saturate_cast((x + 1) * 127.5, dtype=tf.uint8)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    hr_images = features['hr_images']
    hr_images = tf.to_float(hr_images) / 127.5 - 1
    mse = tf.losses.mean_squared_error(labels=hr_images, predictions=x)
    loss = mse
    # PSNR
    max_value = 2.0
    psnr = math_ops.subtract(
        20 * math_ops.log(max_value) / math_ops.log(10.0),
        np.float32(10 / np.log(10)) * math_ops.log(mse),
        name='psnr')
    psnr = tf.metrics.mean(psnr)

    tf.summary.scalar('psnr', psnr[1])

    if trainig:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001, name='Adam')
        train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch_normalization
        train_op.append(optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()))
        train_op = tf.group(*train_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        "psnr": psnr,
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    parser = argparse.ArgumentParser(description='train SRGAN')
    parser.add_argument('--datapath', type=str, required=True, help='location of SR dataset tfrecords')
    parser.add_argument('--model_dir', type=str, default='/tmp/SRResNet', help='directory to save model')
    parser.add_argument('--evalue', action='store_true', help='evalue model')

    args = parser.parse_args()

    config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                    save_summary_steps=1,
                                    save_checkpoints_steps=100,
                                    keep_checkpoint_max=10,
                                    log_step_count_steps=10)

    srresnet = tf.estimator.Estimator(model_fn=model_fn, config=config)
    print(srresnet.latest_checkpoint())
    batch_size = 64

    def input_fn_factory(setname):
        def input_fn():
            dataset = get_dataset(args.datapath, setname)
            dataset = dataset.repeat(count=10)
            if setname == 'train':
                dataset = dataset.shuffle(1000)
            if setname != 'evalue':
                dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(2)
            iterator = dataset.make_one_shot_iterator()
            hr_images, lr_images = iterator.get_next()
            return {'hr_images': hr_images, 'lr_images': lr_images}

        return input_fn

    if args.evalue:
        dataset = get_dataset(args.datapath, 'validation')
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            def get_next_figure():
                hr_image, lr_image = sess.run(next_element)

                def get_input_fn(lr_image):
                    def input_fn():
                        return {'lr_images': tf.constant(np.expand_dims(lr_image, 0), dtype=tf.float32)}
                    return input_fn
                result = next(srresnet.predict(input_fn=get_input_fn(lr_image)))
                # result_image = np.clip(result['sr_images'], 0, 255).astype(np.uint8)
                result_image = result['sr_images']
                plt.subplot(1, 3, 1)
                plt.imshow(hr_image)
                plt.subplot(1, 3, 2)
                plt.imshow(result_image)
                plt.subplot(1, 3, 3)
                plt.imshow(np.clip(lr_image, 0, 255).astype(np.uint8))
                return hr_image, lr_image, result['sr_images']
            import IPython
            IPython.embed()
        return

    for epoch in range(40):
        print('Epoch {}'.format(epoch))
        srresnet.train(input_fn=input_fn_factory('train'))
        eval_results = srresnet.evaluate(input_fn=input_fn_factory('validation'))
        print(eval_results)


if __name__ == '__main__':
    main()
