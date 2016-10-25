# TensorFlow Docker images

This directory contains example Dockerfiles to run TensorFlow on cluster
managers.

- [Dockerfile](Dockerfile) is the most basic example, which just adds a Python
  training program on top of the tensorflow/tensorflow Docker image.
- [Dockerfile.hdfs](Dockerfile.hdfs) installs Hadoop libraries and sets the
  appropriate environment variables to enable reading from HDFS.
- [mnist_replica.py](mnist_replica.py) demonstrates the programmatic setup
  required for distributed TensorFlow training.

## Best practices

- Always pin the TensorFlow version with the Docker image tag. This ensures that
  TensorFlow updates don't adversely impact your training program for future
  runs.
- When creating an image, specify version tags (see below). If you make code
  changes, increment the version. Cluster managers will not pull an updated
  Docker image if they have them cached. Also, versions ensure that you have
  a single copy of the code running for each job.

## Building the Docker files

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
