## Running Distributed TensorFlow on Mesos/Marathon

### Prerequisite
Before you start, you need to set up a Mesos cluster with Marathon installed and Docker Containerizer and Mesos-DNS enabled. It is also preferable to set up some shared storage such as HDFS in the cluster. All of these could be easily installed and configured with the help of [DC/OS](https://dcos.io/docs/1.7/administration/installing/custom/gui/). You need to remember the master target, DNS domain and HDFS namenode which are needed to bring up the TensorFlow cluster.

### Write your training program
This section covers instructions on how to write your trainer program, and build your docker image.

 1. Write your own training program. This program must accept `worker_hosts`, `ps_hosts`, `job_name`, `task_index` as command line flags:

    ```python
    # Flags for configuring the task
    flags.DEFINE_integer("task_index", None,
                                        "Worker task index, should be >= 0. task_index=0 is "
                                        "the master worker task the performs the variable "
                                        "initialization ")
    flags.DEFINE_string("ps_hosts", None,
                                     "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", None,
                                     "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("job_name", None,"job name: worker or ps")
    ```

    and parse them into `ClusterSpec` at the beginning and starts the tensorflow server before your training begins:

    ```python
    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")


    cluster_spec = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})


    server = tf.train.Server(
        cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


    if FLAGS.job_name == "ps":
      server.join()
    ```

  This code is included in the example located in `docker/mnist_replica.py`.
  The worker task and parameter server task usually share a common program. Therefore in the training program, if it is a parameter server task then it will just join the server; otherwise it builds the graph and start its session. This is the typical setup for between-graph replication training which is illustrated in the following diagram. Note, each dashed box indicate a task.
  ![Diagram for Between-graph replication]
  (images/between-graph_replication.png "Between-graph Replication")

 2. Write your own Docker file which simply copies your training program into the image and optionally specify an entrypoint. An example is located in `docker/Dockerfile` or `docker/Dockerfile.hdfs` if you need the HDFS support. TensorBoard can also use the same image, but with a different entry point.

 3. Build your docker image, push it to a docker repository:

  ```bash
  cd docker
  docker build -t <image_name> -f Dockerfile.hdfs .
  # Use gcloud docker push instead if on Google Container Registry.
  docker push <image_name>
  ```

### Generate Marathon Config
The Marathon config is generated from a Jinja template where you need to customize your own cluster configuration in the file header.

 1. Copy over the template file:

  ```
  cp marathon/template.json.jinja mycluster.json.jinja
  ```

 2. Edit the `mycluster.json.jinja` file. You need to specify the `name`, `image_name`, `data_dir`, `train_dir` and optionally change number of worker and ps replicas. `data_dir` points to your training data, and `train_dir` points to the directory on shared storage if you would like to use TensorBoard or sharded checkpoint.
 3. Generate the Marathon json config:

  ```bash
  python render_template.py mycluster.json.jinja > mycluster.json
  ```

### Start the Tensorflow cluster
To start the cluster, simply post the Marathon json config file to the Marathon master target which is `marathon.mesos:8080` by default:

  ```bash
  curl -i -H 'Content-Type: application/json' -d @mycluster.json http://marathon.mesos:8080/v2/groups
  ```

You may want to make sure your cluster is running the training program correctly. Navigate to the DC/OS web console and look for stdout or stderr of the chief worker. The `mnist_replica.py` example would print losses for each step and final loss when training is done.

![Screenshot of the chief worker]
(images/chief_worker_stdout.png "Screenshot of the chief worker")

If TensorBoard is enabled, navigate to `tensorboard.marathon.mesos:6006` with your browser or find out its IP address from the DC/OS web console.
