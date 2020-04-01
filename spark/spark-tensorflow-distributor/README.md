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
pip install pyspark>=3.0.0
```

## Running Tests

For integration tests, first build the master and worker images and then run the test script.

```bash
docker-compose build --build-arg PYTHON_INSTALL_VERSION=3.7 --build-arg UBUNTU_VERSION=18.04
./tests/integration/run.sh
```

## Examples

Run following example code in `pyspark` shell:

```python
from spark_tensorflow_distributor import MirroredStrategyRunner


# Taken from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
def train():
    import tensorflow_datasets as tfds
    import tensorflow as tf
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def make_datasets_unbatched():
        # Scaling MNIST data from (0, 255] to (0., 1.]
        def scale(image, label):
            image = tf.cast(image, tf.float32)
            image /= 255
            return image, label
        datasets, info = tfds.load(
            name='mnist',
            with_info=True,
            as_supervised=True,
        )
        return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

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

    GLOBAL_BATCH_SIZE = 64 * 8
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE).repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)
    return tf.config.experimental.list_physical_devices('GPU')

MirroredStrategyRunner(num_slots=4).run(train)
```

