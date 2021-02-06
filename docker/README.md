# TensorFlow Docker Images

This directory contains example Dockerfiles to run TensorFlow on cluster
managers.

- [Dockerfile](Dockerfile) is the most basic example, which just adds a Python
  training program on top of the tensorflow/tensorflow Docker image. All training programs
  in this directory will be copied into docker image
- [Dockerfile.hdfs](Dockerfile.hdfs) installs Hadoop libraries and sets the
  appropriate environment variables to enable reading from HDFS.
- [mnist.py](mnist.py) demonstrates the programmatic setup for distributed
  TensorFlow training using the tensorflow 1.x API.
- [keras_mnist.py](mnist.py) demonstrates how to train an MNIST classifier using
  [tf.distribute.MultiWorkerMirroredStrategy and Keras Tensorflow 2.0 API](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).
- [custom_training_mnist.py](mnist.py) demonstrates how to train a fashion MNIST classifier using
  [tf.distribute.MultiWorkerMirroredStrategy and Tensorflow 2.0 Custom Training Loop APIs](https://www.tensorflow.org/tutorials/distribute/custom_training).

## Best Practices

- Always pin the TensorFlow version with the Docker image tag. This ensures that
  TensorFlow updates don't adversely impact your training program for future
  runs.
- When creating an image, specify version tags (see below). If you make code
  changes, increment the version. Cluster managers will not pull an updated
  Docker image if they have them cached. Also, versions ensure that you have
  a single copy of the code running for each job.

## Building the Docker Files

First, pick an image name for the job. When running on a cluster manager, you
will want to push your images to a container registry. Note that both the
[Google Container Registry](https://cloud.google.com/container-registry/)
and the [Amazon EC2 Container Registry](https://aws.amazon.com/ecr/) require
special paths. We append `:v1` to version our images. Versioning images is
strongly recommended for reasons described in the best practices section.

```sh
docker build -t <image_name>:v1 -f Dockerfile .
# Use gcloud docker push instead if on Google Container Registry.
docker push <image_name>:v1
```

If you make any updates to the code, increment the version and rerun the above
commands with the new version.

## Running the mnist Example

The [mnist.py](mnist.py) example reads the mnist data in the TFRecords format.
You can run the [convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
program to convert mnist data to TFRecords.

When running distributed TensorFlow, you should upload the converted data to
a common location on distributed storage, such as GCS or HDFS.

## Running the keras_mnist.py example

The [keras_mnist.py](keras_mnist.py) example demonstrates how to train an MNIST classifier using
[tf.distribute.MultiWorkerMirroredStrategy and Keras Tensorflow 2.0 API](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).
The final model is saved to disk by the chief worker process. The disk is assumed to be mounted onto the running container by the cluster manager.
It assumes that the cluster configuration is passed in through the `TF_CONFIG` environment variable when deployed in the cluster

## Running the custom_training_mnist.py example

The [custom_training_mnist.py](mnist.py) example demonstrates how to train a fashion MNIST classifier using
[tf.distribute.MultiWorkerMirroredStrategy and Tensorflow 2.0 Custom Training Loop APIs](https://www.tensorflow.org/tutorials/distribute/custom_training).
The final model is saved to disk by the chief worker process. The disk is assumed to be mounted onto the running container by the cluster manager.
It assumes that the cluster configuration is passed in through the `TF_CONFIG` environment variable when deployed in the cluster.
