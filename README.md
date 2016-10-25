# TensorFlow Ecosystem

This repository contains examples for integrating TensorFlow with other
open-source frameworks. The examples are minimal and intended for use as
templates. Users can tailor the templates for their own use-cases.

If you have any additions or improvements, please create an issue or pull
request.

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
replication with asynchronous updates.

### Common setup for for distributed training

Every distributed training program has some common setup. First, define flags so
that the worker knows about other workers and knows what role it plays in
distributed training:

```python
# Flags for configuring the task
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization.")
flags.DEFINE_string("ps_hosts", None,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", None,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
```

Then, start your server. Parameter servers (ps jobs) should stop at this point
because they only store variables, so they are joined with the server.

```python
# Construct the cluster and start the server
ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")

cluster = tf.train.ClusterSpec({
    "ps": ps_spec,
    "worker": worker_spec})

server = tf.train.Server(
    cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
  server.join()
```

Afterwards, your code varies depending on the form of distributed training you
intend on doing. The most common form is between-graph replication.

### Between-graph Replication

In this mode, each worker separately constructs the exact same graph. Each
worker then runs the graph in isolation, only sharing gradients with the
parameter servers.

You must explicitly set the device before graph construction for this mode of
training. The following code snippet from the
[Distributed TensorFlow tutorial](https://www.tensorflow.org/versions/master/how_tos/distributed/index.html)
demonstrates the setup:

```python
with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):
  # Construct the TensorFlow graph.

# Run the TensorFlow graph.
```
