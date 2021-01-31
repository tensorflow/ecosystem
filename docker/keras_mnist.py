from __future__ import print_function

import math
import os
import tensorflow as tf
import numpy as np
import json

"""
This code serves as an example of using Tensorflow 2.0 Keras API to build and train a CNN model on the 
MNIST dataset using the tf.distribute.MultiWorkerMirroredStrategy described here 
https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy.
This code is very similar to the example provided here
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
Assumptions: 
  1) The code assumes that the cluster configurations needed for the TF distribute strategy is available through the 
  TF_CONFIG environment variable. See the link provided above for details
  2) The model is checkpointed and saved in /pvcmnt by the chief worker process. All other worker processes checkpoint 
  their code in the /tmp directory
"""


# Used to run example using CPU only. Untested on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model save directory
MAIN_MODEL_PATH = '/pvcmnt'

def _is_chief(task_type, task_id):
  # If `task_type` is None, this may be operating as single worker, which works
  # effectively as chief.
  return task_type is None or task_type == 'chief' or (
            task_type == 'worker' and task_id == 0)

def _get_temp_dir(task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join("/tmp", base_dirpath)
  os.makedirs(temp_dir)
  return temp_dir

def write_filepath(strategy):
  task_type, task_id = strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id
  if not _is_chief(task_type, task_id):
    checkpoint_dir = _get_temp_dir(task_id)
  else:
    base_dirpath = 'workertemp_' + str(task_id)
    checkpoint_dir = os.path.join(MAIN_MODEL_PATH, base_dirpath)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
  return checkpoint_dir

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

def main():
  per_worker_batch_size = 64
  tf_config = json.loads(os.environ['TF_CONFIG'])
  num_workers = len(tf_config['cluster']['worker'])
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  
  global_batch_size = per_worker_batch_size * num_workers
  multi_worker_dataset = mnist_dataset(global_batch_size)
  
  # missing needs to be fixed
  # multi_worker_dataset = strategy.distribute_datasets_from_function(mnist_dataset(global_batch_size))  
  
  callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=write_filepath(strategy))]
  with strategy.scope():
      multi_worker_model = build_and_compile_cnn_model()
  multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70,
                          callbacks=callbacks)
  multi_worker_model.save(filepath=write_filepath(strategy))

if __name__=="__main__":
  main()