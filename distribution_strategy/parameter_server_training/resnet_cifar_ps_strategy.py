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
# =============================================================================

# Lint as: python3
"""ResNet Cifar + ParameterServerStrategy example.
"""
import os
from datetime import datetime
import multiprocessing
from absl import app
from absl import flags
from absl import logging
import portpicker
import tensorflow as tf
from tensorflow_models.official.benchmark.models import cifar_preprocessing
from tensorflow_models.official.benchmark.models import resnet_cifar_model
from tensorflow_models.official.vision.image_classification.resnet import common as img_class_common

flags.DEFINE_string("checkpoint_dir", "gs://cifar10_ckpt/",
                    "Directory for writing model checkpoints.")
flags.DEFINE_string("data_dir", "gs://cifar10_data/",
                    "Directory for Resnet Cifar model input. Follow the "
                    "instruction here to get Cifar10 data: "
                    "https://github.com/tensorflow/models/tree/r1.13.0/official/resnet#cifar-10")
flags.DEFINE_string("train_log_dir", "gs://cifar10_train_log/",
                    "Directory for Resnet Cifar training logs")
flags.DEFINE_boolean(
    "use_in_process_cluster", False,
    "Whether to use in-process cluster for testing.")
flags.DEFINE_boolean(
    "run_in_process_training", True,
    "Whether to use in-process cluster to run training or evaluation.")

FLAGS = flags.FLAGS

TRAIN_EPOCHS = 182
STEPS_PER_EPOCH = 781
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 8
EVAL_STEPS_PER_EPOCH = 88

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_spec dict."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        protocol="grpc",
        config=worker_config,
        task_index=i,
        start=True)

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        protocol="grpc",
        task_index=i,
        start=True)

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

def run_tf_server_and_wait(cluster_resolver):
  assert cluster_resolver.task_type in ("worker", "ps")
  server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol=cluster_resolver.rpc_layer or "grpc",
    start=True)
  server.join()


def train_resnet_cifar(cluster_resolver):
  """Trains the resnet56 model using parameter server distribution strategy.

  Args:
      cluster_resolver: cluster resolver to give neccessary information to
        set up distributed training
  """

  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver)
  coordinator = (
        tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))
  with strategy.scope():
    model = resnet_cifar_model.resnet56()

    initial_learning_rate = (
        img_class_common.BASE_LEARNING_RATE * BATCH_SIZE / 128)
    # Using the learning rate schedule from the model garden:
    # tensorflow_models/official/benchmark/models/resnet_cifar_main.py
    lr_segments = [  # (multiplier, epoch to start) tuples
        (0.1, 91), (0.01, 136), (0.001, 182)
    ]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(p[1] * STEPS_PER_EPOCH for p in lr_segments),
        values=[initial_learning_rate] +
        list(p[0] * initial_learning_rate for p in lr_segments))
    optimizer = img_class_common.get_optimizer(lr_schedule)

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy")
    eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="eval_accuracy")

    @tf.function
    def worker_train_fn(iterator):

      def replica_fn(inputs):
        """Training loop function."""
        batch_data, labels = inputs
        with tf.GradientTape() as tape:
          predictions = model(batch_data, training=True)
          xent_loss = tf.keras.losses.SparseCategoricalCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE)(labels, predictions)
          loss = (
              tf.nn.compute_average_loss(xent_loss) +
              tf.nn.scale_regularization_loss(model.losses))
        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss

      inputs = next(iterator)
      losses = strategy.run(replica_fn, args=(inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


    @tf.function
    def worker_eval_fn(iterator):

      def eval_fn(inputs):
        """Evaluation function"""
        batch_data, labels = inputs
        predictions = model(batch_data, training=False)
        eval_accuracy.update_state(labels, predictions)

      inputs = next(iterator)
      strategy.run(eval_fn, args=(inputs,))

    checkpoint_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(model=model, optimizer=optimizer),
        FLAGS.checkpoint_dir,
        max_to_keep=2)
    if checkpoint_manager.latest_checkpoint:
      checkpoint = checkpoint_manager.checkpoint
      checkpoint.restore(
          checkpoint_manager.latest_checkpoint
          ).assert_existing_objects_matched()

    train_dataset_fn = lambda _: cifar_preprocessing.input_fn(
        is_training=True,
        data_dir=FLAGS.data_dir,
        batch_size=BATCH_SIZE,
        parse_record_fn=cifar_preprocessing.parse_record,
        dtype=tf.float32,
        drop_remainder=True)
    eval_dataset_fn = lambda _: cifar_preprocessing.input_fn(
        is_training=False,
        data_dir=FLAGS.data_dir,
        batch_size=EVAL_BATCH_SIZE,
        parse_record_fn=cifar_preprocessing.parse_record,
        dtype=tf.float32)

    # The following wrappers will allow efficient prefetching to GPUs
    # when GPUs are supported by ParameterServerStrategy
    @tf.function
    def per_worker_train_dataset_fn():
      return strategy.distribute_datasets_from_function(train_dataset_fn)

    @tf.function
    def per_worker_eval_dataset_fn():
      return strategy.distribute_datasets_from_function(eval_dataset_fn)

    per_worker_train_dataset = coordinator.create_per_worker_dataset(
      per_worker_train_dataset_fn)
    per_worker_eval_dataset = coordinator.create_per_worker_dataset(
      per_worker_eval_dataset_fn)

  global_steps = int(optimizer.iterations.numpy())
  logging.info("Training starts with global_steps = %d", global_steps)
  current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = FLAGS.train_log_dir + current_time
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)

  for epoch in range(global_steps // STEPS_PER_EPOCH,
                     TRAIN_EPOCHS):
    per_worker_train_iterator = iter(per_worker_train_dataset)
    per_worker_eval_iterator = iter(per_worker_eval_dataset)
    for _ in range(STEPS_PER_EPOCH):
      coordinator.schedule(worker_train_fn, args=(per_worker_train_iterator,))
    coordinator.join()
    logging.info("Finished joining at epoch %d. Training accuracy: %f.",
                  epoch, train_accuracy.result())

    # Since we are running inline evaluation below, a side-car evaluator job is not necessary.
    for _ in range(EVAL_STEPS_PER_EPOCH):
      coordinator.schedule(worker_eval_fn, args=(per_worker_eval_iterator,))
    coordinator.join()
    logging.info("Finished joining at epoch %d. Evaluation accuracy: %f.",
                  epoch, eval_accuracy.result())

    with train_summary_writer.as_default():
      tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
      tf.summary.scalar('eval_accuracy', eval_accuracy.result(), step=epoch)
    train_accuracy.reset_states()
    eval_accuracy.reset_states()
    checkpoint_manager.save()


def evaluate_resnet_cifar():
  """Evaluates the resnet56 model

  This method provides side-car evaluation using the checkpoints

  """
  eval_model = resnet_cifar_model.resnet56()
  eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="eval_accuracy")
  eval_model.compile(metrics=eval_accuracy)
  eval_dataset = cifar_preprocessing.input_fn(
      is_training=False,
      data_dir=FLAGS.data_dir,
      batch_size=BATCH_SIZE,
      parse_record_fn=cifar_preprocessing.parse_record)


  checkpoint = tf.train.Checkpoint(model=eval_model)

  for latest_checkpoint in tf.train.checkpoints_iterator(
      FLAGS.checkpoint_dir):
    try:
      checkpoint.restore(latest_checkpoint).expect_partial()
    except tf.errors.OpError:
      # checkpoint may be deleted by training when it is about to read it.
      continue

    # Optionally add callbacks to write summaries.
    eval_model.evaluate(eval_dataset)

    # Evaluation finishes when it has evaluated the last epoch.
    if latest_checkpoint.endswith("-{}".format(TRAIN_EPOCHS)):
      break


def main(_):
  if FLAGS.use_in_process_cluster:
    if FLAGS.run_in_process_training:
      cluster_resolver = create_in_process_cluster(3, 1)
      train_resnet_cifar(cluster_resolver)
    else:
      evaluate_resnet_cifar()
  else:
    os.environ["grpc_fail_fast"] = "use_caller"
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
      run_tf_server_and_wait(cluster_resolver)
    elif cluster_resolver.task_type == "evaluator":
      evaluate_resnet_cifar()
    else:
      train_resnet_cifar(cluster_resolver)


if __name__ == "__main__":
  app.run(main)
