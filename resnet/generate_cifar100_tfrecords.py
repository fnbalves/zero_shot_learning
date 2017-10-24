# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
#Partialy based on https://github.com/tensorflow/models

Read CIFAR-100 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-100 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import os

import tarfile
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

CIFAR_FILENAME = 'cifar-100-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-100-python'


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['train']
  file_names['validation'] = ['train']
  file_names['eval'] = ['test']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data_dict = u.load()
  return data_dict


def convert_to_tfrecord(input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)

      if "train" in output_file:
          data = data_dict['data']
          data = data[0 : int(len(data)* 0.8)]
          coarse_labels = data_dict['coarse_labels']
          coarse_labels = coarse_labels[0 : int(len(coarse_labels)* 0.8)]
          fine_labels = data_dict['fine_labels']
          fine_labels = fine_labels[0 : int(len(fine_labels)* 0.8)]
          num_entries_in_batch = len(fine_labels)
      elif "validation" in output_file:
          data = data_dict['data']
          data = data[int(len(data) * 0.8) : ]
          coarse_labels = data_dict['coarse_labels']
          coarse_labels = coarse_labels[int(len(coarse_labels) * 0.8) : ]
          fine_labels = data_dict['fine_labels']
          fine_labels = fine_labels[int(len(fine_labels) * 0.8) : ]
          num_entries_in_batch = len(fine_labels)
      else:
          data = data_dict['data']
          coarse_labels = data_dict['coarse_labels']
          fine_labels = data_dict['fine_labels']
          num_entries_in_batch = len(fine_labels)

      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                # 'coarse_label': _int64_feature(coarse_labels[i]),
                'label': _int64_feature(fine_labels[i])
            }))
        record_writer.write(example.SerializeToString())


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  # download_and_extract(data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-100 to.')

  args = parser.parse_args()
  main(args.data_dir)
