import os.path
import argparse
import tensorflow as tf

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

    def parser(record):
        images = parse_record(record)
        hr_images = tf.random_crop(images, (64, 64, 3))
        lr_images = tf.image.resize_images(hr_images, (32, 32))
        return hr_images, lr_images

    dataset = dataset.map(parser, num_parallel_calls=8)
    return dataset


def model_fn(features, labels, mode):
    trainig = mode == tf.estimator.ModeKeys.TRAIN
    lr_images = features['lr_images']
    lr_images = tf.to_float(lr_images) / 255.0

    x = SRResNet(lr_images, training=trainig)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"sr_images": x}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    hr_images = features['hr_images']
    hr_images = tf.to_float(hr_images) / 255.0
    loss = tf.losses.mean_squared_error(labels=hr_images, predictions=x)
    if trainig:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam')
        train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch_normalization
        train_op.append(optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()))
        train_op = tf.group(*train_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)


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
    batch_size = 100

    def input_fn_factory(setname):
        def input_fn():
            dataset = get_dataset(args.datapath, setname)
            dataset.repeat(count=10)
            if setname == 'train':
                dataset = dataset.shuffle(1000)
            if setname != 'evalue':
                dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(2)
            iterator = dataset.make_one_shot_iterator()
            hr_images, lr_images = iterator.get_next()
            return {'hr_images': hr_images, 'lr_images': lr_images}

        return input_fn
    for epoch in range(40):
        print('Epoch {}'.format(epoch))
        srresnet.train(input_fn=input_fn_factory('train'))
        eval_results = srresnet.evaluate(input_fn=input_fn_factory('validation'))
        print(eval_results)

if __name__ == '__main__':
    main()
