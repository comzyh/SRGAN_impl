import argparse
import os
import tensorflow as tf
import numpy as np

from PIL import Image
from model import SRResNet
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='SRGAN prediction tools')
    parser.add_argument('testing_dir', type=str, help='directory to load images')
    parser.add_argument('output_dir', type=str, help='directory to save results')
    parser.add_argument('--model_dir', type=str, default='/tmp/SRResNet', help='directory of model')
    parser.add_argument('--bicubic', action='store_true', help='using bicubic for compare')

    args = parser.parse_args()

    images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    prediction = SRResNet(images, training=False, reuse=True)
    prediction = tf.saturate_cast(prediction * 255.0, dtype=tf.uint8)

    model_file = tf.train.latest_checkpoint(args.model_dir)
    saver = tf.train.Saver()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        for filename in os.listdir(args.testing_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img_path = os.path.join(args.testing_dir, filename)
                img = Image.open(img_path)
                img_lr = img.resize((img.size[0] // 2, img.size[1] // 2), Image.ANTIALIAS)
                if args.bicubic:
                    img_sr = img_lr.resize(img.size, Image.BICUBIC)
                else:
                    img_lr = np.asarray(img_lr, dtype=np.float32) / 255.0
                    result = sess.run(prediction, feed_dict={images: np.expand_dims(img_lr, axis=0)})
                    img_sr = Image.fromarray(result[0])
                # save as PNG
                name, _ = os.path.splitext(filename)
                img_sr.save(os.path.join(args.output_dir, name + '.png'))
                print(name)


if __name__ == '__main__':
    main()
