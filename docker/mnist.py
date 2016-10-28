# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import print_function

import math
import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

flags = tf.app.flags

# Flags for configuring the task
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", 0,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task the performs the variable "
                     "initialization")
flags.DEFINE_string("ps_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "",
                    "Comma-separated list of hostname:port pairs")

# Training related flags
flags.DEFINE_string("data_dir", None,
                    "Directory where the mnist data is stored")
flags.DEFINE_string("train_dir", None,
                    "Directory for storing the checkpoints")
flags.DEFINE_integer("hidden1", 128,
                     "Number of units in the 1st hidden layer of the NN")
flags.DEFINE_integer("hidden2", 128,
                     "Number of units in the 2nd hidden layer of the NN")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")

FLAGS = flags.FLAGS
TRAIN_FILE = "train.tfrecords"


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([mnist.IMAGE_PIXELS])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(batch_size):
  """Reads input data.

  Args:
    batch_size: Number of examples per returned batch.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
  """
  filename = os.path.join(FLAGS.data_dir, TRAIN_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels


def device_and_target():
  # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
  # Don't set a device.
  if FLAGS.job_name is None:
    print("Running single-machine training")
    return (None, "")

  # Otherwise we're running distributed TensorFlow.
  print("Running distributed training")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
    raise ValueError("Must specify an explicit `ps_hosts`")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
    raise ValueError("Must specify an explicit `worker_hosts`")

  cluster_spec = tf.train.ClusterSpec({
      "ps": FLAGS.ps_hosts.split(","),
      "worker": FLAGS.worker_hosts.split(","),
  })
  server = tf.train.Server(
      cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()

  worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  return (
      tf.train.replica_device_setter(
          worker_device=worker_device,
          cluster=cluster_spec),
      server.target,
  )


def main(unused_argv):
  if FLAGS.data_dir is None or FLAGS.data_dir == "":
    raise ValueError("Must specify an explicit `data_dir`")
  if FLAGS.train_dir is None or FLAGS.train_dir == "":
    raise ValueError("Must specify an explicit `train_dir`")

  device, target = device_and_target()
  with tf.device(device):
    images, labels = inputs(FLAGS.batch_size)
    logits = mnist.inference(images, FLAGS.hidden1, FLAGS.hidden2)
    loss = mnist.loss(logits, labels)
    train_op = mnist.training(loss, FLAGS.learning_rate)

  with tf.train.MonitoredTrainingSession(
      master=target,
      is_chief=(FLAGS.task_index == 0),
      checkpoint_dir=FLAGS.train_dir) as sess:
    while not sess.should_stop():
      sess.run(train_op)


if __name__ == "__main__":
  tf.app.run()
