# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tools for model introspection."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import sonnet as snt
import tensorflow as tf


def classification_probe(features, labels, n_classes, labeled=None, multi=False):
  """Classification probe with stopped gradient on features."""

  def _classification_probe(features):
    logits = snt.Linear(n_classes)(tf.stop_gradient(features))

    if multi:
      # Labels is int32 [128, 10]
      # Logits is [128, 10] float32
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels),
                                                     logits=logits)
      # CP is 128,10
      correct_predictions = tf.equal(
        tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.int32),
        tf.round(labels))
      # ALT is 128
      all_labels_true = tf.reduce_min(tf.cast(correct_predictions, tf.float32), 1)
      # ACC is ()
      acc = tf.reduce_mean(all_labels_true)
    else:
      print('Labels: ', labels, ' ... Logits: ', logits)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
      acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1),
                                                labels)))

    if labeled is not None:
      loss = loss * tf.to_float(labeled)
    loss = tf.reduce_mean(loss)

    return loss, acc

  return snt.Module(_classification_probe)(features)
