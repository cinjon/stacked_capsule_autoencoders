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

"""Image datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow import nest
import tensorflow_datasets as tfds


from stacked_capsule_autoencoders.capsules.data import tfrecords as _tfrecords


def create(which,
           batch_size,
           subset=None,
           n_replicas=1,
           transforms=None,
           **kwargs):
  """Creates data loaders according to the dataset name `which`."""

  func = globals().get('_create_{}'.format(which), None)
  if func is None:
    raise ValueError('Dataset "{}" not supported. Only {} are'
                     ' supported.'.format(which, SUPPORTED_DATSETS))

  dataset = func(subset, batch_size, **kwargs)
  print('create dataaset: ', dataset)

  if transforms is not None:
    if not isinstance(transforms, dict):
      transforms = {'image': transforms}

    for k, v in transforms.items():
      transforms[k] = snt.Sequential(nest.flatten(v))

  def map_func(data):
    """Replicates data if necessary."""
    data = dict(data)
    
    if n_replicas > 1:
      tile_by_batch = snt.TileByDim([0], [n_replicas])
      data = {k: tile_by_batch(v) for k, v in data.items()}
      print(data)

    if transforms is not None:
      img = data['image']

      for k, transform in transforms.items():
        data[k] = transform(img)

    return data

  def moving_mnist_map_func(index, image, label):
    data = {'index': index, 'image': image, 'label': label}

    if n_replicas > 1:
      print('n_replicas: ', n_replicas)
      tile_by_batch = snt.TileByDim([0], [n_replicas])
      data = {k: tile_by_batch(v) for k, v in data.items()}
      print(data)

    if transforms is not None:
      img = data['image']
      print('before transforms: ', data)

      for k, transform in transforms.items():
        data[k] = transform(img)
      print('after transforms: ', data)

    return data

  if transforms is not None or n_replicas > 1:
    if 'moving_mnist' in which:
      dataset = dataset.map(moving_mnist_map_func) \
                       .prefetch(tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.map(map_func)

  iter_data = dataset.make_one_shot_iterator()
  input_batch = iter_data.get_next()
  for _, v in input_batch.items():
    v.set_shape([batch_size * n_replicas] + v.shape[1:].as_list())

  return input_batch


def _create_mnist(subset, batch_size, **kwargs):
  return tfds.load(
      name='mnist', split=subset, **kwargs).repeat().batch(batch_size)


def _gen_moving_mnist(directory, shuffle=False, first_only=False, rand_choice=False, do_argmax=False):
  img_dir = os.path.join(directory, 'imgs')
  lbl_dir = os.path.join(directory, 'labels')
  count = len(img_dir)

  def _get_order():
    if shuffle:
      return np.random.choice(count, size=count, replace=False)
    else:
      return list(range(count))
  order = _get_order()

  i = 0
  while True:
    f = '%06d.npy' % order[i]
    img = np.load(os.path.join(img_dir, f))
    img = img.transpose(0, 2, 3, 1)

    if first_only:
      img = img[0]
    elif rand_choice:
      img = img[random.choice(range(img.shape[0]))]

    lbl = np.load(os.path.join(lbl_dir, f))
    if do_argmax:
      lbl = np.argmax(lbl)
    yield order[i], img, lbl

    i += 1
    if i == count:
      order = _get_order()
      i = 0


def _create_moving_mnist(subset, batch_size, first_only=False, rand_choice=False):
  # imgs: [20, num_imgs, 1, 64, 64], lbsl: [num_imgs, 10] because multilabel.
  g = lambda: _gen_moving_mnist(
    '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/moving_mnist/%s' % subset,
    shuffle=subset == 'train',
    first_only=first_only,
    rand_choice=rand_choice
  )
  if first_only or rand_choice:
    output_shapes = ((), (64, 64, 1), (10))
  else:
    output_shapes = ((), (20, 64, 64, 1), (10))

  dataset = tf.data.Dataset.from_generator(
    g,
    output_types=(tf.int32, tf.int32, tf.int32),
    output_shapes=output_shapes
  )
  return dataset.repeat().batch(batch_size)


def _create_moving_mnist_first(subset, batch_size, **kwargs):
  """This dataset differs from moving mnist in that we only take the first image."""
  return _create_moving_mnist(subset, batch_size, first_only=True)


def _create_moving_mnist_rand(subset, batch_size, **kwargs):
  """This dataset differs from moving mnist in that choose a random image."""
  return _create_moving_mnist(subset, batch_size, rand_choice=True)


def _create_moving_mnist_single(subset, batch_size, first_only=False, rand_choice=False):
  # imgs: [20, num_imgs, 1, 64, 64], lbsl: [num_imgs, 10] because multilabel.
  g = lambda: _gen_moving_mnist(
    '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion/moving_mnist_single/%s' % subset,
    shuffle=subset == 'train',
    first_only=first_only,
    rand_choice=rand_choice,
    do_argmax=True
  )
  if first_only or rand_choice:
    output_shapes = ((), (64, 64, 1), ())
  else:
    output_shapes = ((), (20, 64, 64, 1), ())

  dataset = tf.data.Dataset.from_generator(
    g,
    output_types=(tf.int32, tf.int32, tf.int64),
    output_shapes=output_shapes
  )
  return dataset.repeat().batch(batch_size)


def _create_moving_mnist_single_rand(subset, batch_size, **kwargs):
  """This dataset differs from moving mnist in that choose a random image."""
  return _create_moving_mnist_single(subset, batch_size, rand_choice=True)


SUPPORTED_DATSETS = set(
    k.split('_', 2)[-1] for k in globals().keys() if k.startswith('_create'))

