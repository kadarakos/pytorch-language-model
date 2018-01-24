# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


"""Utilities for parsing PTB and Text8 text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  """Read textfile into a list of words."""
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _read_chars(filename):
  """Read textfile into a list of characters. """
  with open(filename, "r") as f:
    return list(f.read())


def _build_vocab(data):

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict((v, k) for k, v in word_to_id.items())

  return word_to_id, id_to_word


def _file_to_word_ids(data, word_to_id):
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, prefix="ptb"):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, prefix + ".train.txt")
  valid_path = os.path.join(data_path, prefix + ".valid.txt")
  test_path = os.path.join(data_path, prefix + ".test.txt")
  train_w = _read_words(train_path)
  valid_w = _read_words(valid_path)
  test_w = _read_words(test_path)
  word_to_id, id_2_word = _build_vocab(train_w)
  train_data = _file_to_word_ids(train_w, word_to_id)
  valid_data = _file_to_word_ids(valid_w, word_to_id)
  test_data = _file_to_word_ids(test_w, word_to_id)
  return train_data, valid_data, test_data, word_to_id, id_2_word


def text8_raw_data(data_path=None):
  """Load text8 raw data from "data_path".
  Reads the text8 text file, converts strings to integer ids,
  and performs mini-batching of the inputs. Uses the standard
  train, val, test splits from Mikolov et al. (2012).
  http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
  The text8 dataset comes from http://mattmahoney.net/dc/text8.zip:
  Args:
    data_path: string path to the text8 file.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  text8 = _read_chars(data_path)
  train = text8[:int(9e7)]
  val = text8[int(9e7):int(95e6)]
  test = text8[int(95e6):]
  word_to_id, id_2_word = _build_vocab(train)
  train_data = _file_to_word_ids(train, word_to_id)
  valid_data = _file_to_word_ids(val, word_to_id)
  test_data = _file_to_word_ids(test, word_to_id)
  return train_data, valid_data, test_data, word_to_id, id_2_word


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]


  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

