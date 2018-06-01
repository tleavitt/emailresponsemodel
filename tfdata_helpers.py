# adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# ==============================================================================
"""Converts data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import pdb
import numpy as np
import tensorflow as tf

from util import check_dirs
from defs import MAX_LENGTH, N_CLASSES


DIRECTORY = os.path.abspath('../processed-data/tfrecords-2')


tf_filename_func = lambda dataset, it: os.path.abspath(
                "{}/{}{}.tfrecords".format(DIRECTORY, dataset, it)
                # "{}/buy_{}{}.tfrecords".format(DIRECTORY, dataset, it)
            )


def _int64_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecords(email_examples, filepath):
  """Converts a dataset to tfrecords."""
  if filepath is None: raise "No filename given."
  print('Writing', filepath)
  check_dirs(os.path.dirname(filepath))

  with tf.python_io.TFRecordWriter(filepath) as writer:
    for email in email_examples:
      body, length, label = email['Body'], email['Length'], email['Label']
      body = np.asarray(body, dtype=np.int64)
      body_raw = body.tostring()
      id_ = bytes(email['Id'], encoding="utf-8", errors='replace')
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'Body_raw': _bytes_feature(body_raw),
                  'Length': _int64_feature(length),
                  'Label': _int64_feature(label),
                  'Id': _bytes_feature(id_),
              }))
      writer.write(example.SerializeToString())


def parse_tfrecord(example):
  features = {
              'Body_raw': tf.FixedLenFeature((), tf.string, default_value=''),
              'Length': tf.FixedLenFeature((), tf.int64, default_value=0),
              'Label': tf.FixedLenFeature((), tf.int64, default_value=0),
              'Id': tf.FixedLenFeature((), tf.string, default_value=''),
             }
  parsed_features = tf.parse_single_example(example, features)
  parsed_features['Body'] = tf.decode_raw(parsed_features['Body_raw'], out_type=tf.int64)
  parsed_features['Body'] = tf.reshape(parsed_features['Body'], shape=(MAX_LENGTH, 2))
  return parsed_features['Body'], parsed_features['Length'], parsed_features['Label'], parsed_features['Id']


def tf_record_parser(one_hot=True):

  def parse(example):
    body, length, label, id_ = parse_tfrecord(example)
    if one_hot:
      label = tf.one_hot(label, N_CLASSES)
    return body, length, label, id_

  return parse

if __name__ == '__main__':

  filenames = ['{}/dev0.tfrecords'.format(DIRECTORY)]
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parse_tfrecord)

  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  with tf.Session() as sess:
    pdb.set_trace()
    body, length, label, id_ = sess.run(next_element)
    print(length, id_)

