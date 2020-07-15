# Spark TensorFlow Distributor

This package helps users do distributed training with TensorFlow on their Spark clusters.

## Installation

This package requires Python 3.6+, `tensorflow>=2.1.0` and `pyspark>=3.0.0` to run.
To install `spark-tensorflow-distributor`, run:

```bash
pip install spark-tensorflow-distributor
```

The installation does not install PySpark because for most users, PySpark is already installed.
If you do not have PySpark installed, you can install it directly:

```bash
pip install pyspark>=3.0.*
```

Note also that in order to use many features of this package, you must set up Spark custom
resource scheduling for GPUs on your cluster. See the Spark docs for this.

## Running Tests

For integration tests, first build the master and worker images and then run the test script.

```bash
docker-compose build --build-arg PYTHON_INSTALL_VERSION=3.7
./tests/integration/run.sh
```

For linting, run the following.

```bash
./tests/lint.sh
```

To use the autoformatter, run the following.

```bash
yapf --recursive --in-place spark_tensorflow_distributor
```

## Examples

Run following example code in `pyspark` shell:

```python
from spark_tensorflow_distributor import MirroredStrategyRunner

# Adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
def train():
    import tensorflow as tf
    import uuid

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def make_datasets():
        (mnist_images, mnist_labels), _ = \
            tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')

        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64))
        )
        dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return dataset

    def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'],
        )
        return model

    train_datasets = make_datasets()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)

MirroredStrategyRunner(num_slots=8).run(train)
```

