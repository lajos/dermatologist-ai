from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import math

import_paths = ['../models/research/slim']
for import_path in import_paths:
    if import_path not in sys.path:
        sys.path.append(import_path)

import tensorflow as tf

from datasets import dataset_utils

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = '%s_%05d-of-%05d.tfrecord' % (split_name, shard_id,
                                                    _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = get_dataset_filename(dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard,len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' %(i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i],'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width,class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def get_dataset_paths(dataset_root):
    dataset_paths = []
    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            dataset_paths.append(path)
    return dataset_paths

def run(dataset_root):
    dataset_paths = get_dataset_paths(dataset_root)
    class_names = None
    class_names_to_ids = None
    for dataset_path in dataset_paths:
        dataset_name = os.path.split(dataset_path)[-1]
        print('dataset:', dataset_name)
        image_filenames, _class_names = get_filenames_and_classes(dataset_path)
        _class_names_to_ids = dict(zip(_class_names, range(len(_class_names))))

        if class_names is None:
            class_names = _class_names
            class_names_to_ids = _class_names_to_ids
        else:
            if not class_names == _class_names:
                print("ERROR: class names don't match:", class_names, _class_names)
                sys.exit(1)

        convert_dataset(dataset_name, image_filenames, class_names_to_ids, dataset_root)

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_root)

if __name__=='__main__':
    run('data_small')
