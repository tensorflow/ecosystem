# TensorFlow Ecosystem

This repository contains examples for integrating TensorFlow with other
open-source frameworks. The examples are minimal and intended for use as
templates. Users can tailor the templates for their own use-cases.

If you have any additions or improvements, please file an issue or pull request.

## Contents

- [docker](docker) - Docker configuration for running TensorFlow on
  cluster managers.
- [kubernetes](kubernetes) - Templates for running distributed TensorFlow on
  Kubernetes.
- [marathon](marathon) - Templates for running distributed TensorFlow using
  Marathon, deployed on top of Mesos.

## Distributed TensorFlow

See the [Distributed TensorFlow](https://www.tensorflow.org/versions/master/how_tos/distributed/index.html)
documentation for a description of how it works. The examples in this
repository focus on the most common form of distributed training: between-graph
replication with asynchronous training.

### Between-graph Replication

In this mode, each worker constructs their own copy of the TensorFlow graph.
The parameter servers (or ps jobs) share the graph variables, and the workers
asynchronously send gradients to update the weights. See
[mnist_replica.py](docker/mnist_replica.py) for example code to setup this mode
of distributed training.
