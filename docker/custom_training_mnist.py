# ==============================================================================
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

# This code serves as an example of using Tensorflow 2.0 to build and train a CNN model on the 
# Fashion MNIST dataset using the tf.distribute.MultiWorkerMirroredStrategy described here 
# https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy 
# using a custom training loop. This code is very similar to the example provided here
# https://www.tensorflow.org/tutorials/distribute/custom_training
# Assumptions: 
#   1) The code assumes that the cluster configuration needed for the TF distribute strategy is available through the 
#   TF_CONFIG environment variable. See the link provided above for details
#   2) The model is checkpointed and saved in /pvcmnt by the chief worker process.

import tensorflow as tf
import numpy as np
import os

# Used to run example using CPU only. Untested on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MAIN_MODEL_PATH = '/pvcmnt'

EPOCHS = 10
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA

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

def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
  return model

def get_dist_data_set(strategy, batch_size):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Adding a dimension to the array -> new shape == (28, 28, 1)
    # We are doing this because the first layer in our model is a convolutional
    # layer and it requires a 4D input (batch_size, height, width, channels).
    # batch_size dimension will be added later on.
    train_images = train_images[..., None]
    test_images = test_images[..., None]
    # Getting the images in [0, 1] range.
    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(batch_size) 
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
    return train_dist_dataset, test_dist_dataset

def main():
    global GLOBAL_BATCH_SIZE
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dist_dataset, test_dist_dataset = get_dist_data_set(strategy, GLOBAL_BATCH_SIZE)
    checkpoint_pfx = write_filepath(strategy)
    with strategy.scope():
        model = create_model()
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    
    def test_step(inputs):
        images, labels = inputs
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)
    
    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss

    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs,))

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)
        if epoch % 2 == 0:
            checkpoint.save(checkpoint_pfx)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print (template.format(epoch+1, train_loss,
                                train_accuracy.result()*100, test_loss.result(),
                                test_accuracy.result()*100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

if __name__=="__main__":
    main()