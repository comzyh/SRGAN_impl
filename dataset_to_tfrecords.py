import argparse
import random
import os
import os.path

import tensorflow as tf
from tqdm import tqdm


def split_train_vaildation(dataset_path, train_writer, vaild_writer, train_ratio=0.9):
    # data from http://vllab.ucmerced.edu/wlai24/LapSRN/
    total_num = 0
    train_num = 0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        print("scanning {}".format(root))
        for filename in tqdm(files):
            if not filename.endswith('.png'):
                continue
            with open(os.path.join(root, filename), 'rb') as f:
                record = tf.train.Example(features=tf.train.Features(feature={
                    "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f.read()])),
                }))
            if random.random() <= train_ratio:
                train_num += 1
                train_writer.write(record.SerializeToString())
            else:
                vaild_writer.write(record.SerializeToString())
            total_num += 1
    return train_num


def main():
    parser = argparse.ArgumentParser(description='Preprocess super resolution dataset')
    parser.add_argument('--datapath', type=str, required=True, help='location of SR dataset')
    parser.add_argument('--output_path', type=str, help='location of tf_records')
    parser.add_argument('--train_ratio', type=float, help='ratio of train', default=0.8)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with tf.python_io.TFRecordWriter(os.path.join(args.output_path, 'train')) as train_writer:
        with tf.python_io.TFRecordWriter(os.path.join(args.output_path, 'validation')) as vaild_writer:
            split_train_vaildation(
                dataset_path=args.datapath,
                train_writer=train_writer,
                vaild_writer=vaild_writer,
                train_ratio=args.train_ratio
            )


if __name__ == '__main__':
    main()
