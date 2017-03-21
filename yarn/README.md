# TensorFlow launcher for Apache Hadoop YARN

This project implements a [TensorFlow](http://www.tensorflow.org/) session
launcher for [Apache Hadoop YARN](http://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html),
such that users can utilize resources in a YARN cluster.  It can support both
local and distributed TensorFlow application.

## Prerequisites

1. Apache Hadoop YARN
2. Zookeeper
2. Python 2.6+
3. TensorFlow + related packages
4. Docker [optional]

In particular, TernsorFlow and its necessary packages must be either
pre-installed on nodes in the YARN cluster or be available as a Docker image
accessible from those nodes.

## Build

```sh
mvn clean package
```

Configure
---------

Configure Apache Hadoop YARN cluster with Registry/Zookeeper enabled.

## Examples
Tasks are submitted using `ytf-submit` script.

```
ytf-submit [OPTIONS] -r <cluster_requirement>  <task_command>
```

`task_command` is the command to be execute for each of the task of the session.
The two environment variables, `DTF_TASK_JOB_NAME` and `DTF_TASK_INDEX`, will be
set before the task is executed. `cluster_requirement` is a comma separated list
of job names and the number of the instances for that job, with this format:
`<job_name1>:<num_tasks1>,<job_name2>:<num_task2>, ...`.

### Simple task submission

Let's execute a session with 2 x *Parameter Servers (ps)* and 4 x *Workers*.
Assume task program, input data, and output train, all reside in
`/home/user1/mnist` and is accessible to every node.

```sh
$ ytf-submit -r "ps:1,worker:4" \
'python /home/user1/mnist/mnist.py \
--job_name ${DTF_TASK_JOB_NAME} --task_index ${DTF_TASK_INDEX} \
--ps_hosts ${DTF_PS_HOSTS} --worker_hosts ${DTF_WORKER_HOSTS} \
--data_dir /home/user1/mnist/data --train_dir /home/user1/mnist/train'
```

### Enabling TensorBoard
TensorBoard is enabled by `--tensorboard` or `-t`. The address of TensorBoard is
available at **Tracking URL** section of the submitted aplication in Apache YARN
Resource Manager web interface. For using TensorBoard, output path must be
specified by `--output` or `-o`. `DTF_OUTPUT_PATH` environment variable wil be
set and can be used in `task_command`. Similarly, input path can be passed to
`ytf-submit` and will be available as `DTF_INPUT_PATH`.

```sh
$ ytf-submit --tensorboard \
-i /home/user1/mnist/data -o /home/user1/mnist/train10 -r "ps:1,worker:2" \
'python /home/user1/mnist/mnist.py \
--job_name ${DTF_TASK_JOB_NAME} --task_index ${DTF_TASK_INDEX} \
--ps_hosts ${DTF_PS_HOSTS} --worker_hosts ${DTF_WORKER_HOSTS} \
--data_dir ${DTF_INPUT_PATH} --train_dir ${DTF_OUTPUT_PATH}'
```

### Passing the script file

The training code itself can be passed to `ytf-submit`. The code will be copied
to HDFS and will be available at execution time. The path to the training code
will be available as `DTF_TASK_SCRIPT` environment variable.

```sh
$ ytf-submit --tensorboard \
-i /home/user1/mnist/data -o /home/user1/mnist/train10 -r "ps:1,worker:2" \
-s /home/user1/mnist/mnist.py \
'python ${DTF_TASK_SCRIPT} \
--job_name ${DTF_TASK_JOB_NAME} --task_index ${DTF_TASK_INDEX} \
--ps_hosts ${DTF_PS_HOSTS} --worker_hosts ${DTF_WORKER_HOSTS} \
--data_dir ${DTF_INPUT_PATH} --train_dir ${DTF_OUTPUT_PATH}'
```

### Using HDFS paths
Input and output paths can be HDFS paths.

```sh
$ ytf-submit --tensorboard \
-i hdfs://users/user1/mnist/data -o hdfs://users/user1/mnist/train10
-r "ps:1,worker:2" -s /home/user1/mnist/mnist.py \
'python ${DTF_TASK_SCRIPT} \
--job_name ${DTF_TASK_JOB_NAME} --task_index ${DTF_TASK_INDEX} \
--ps_hosts ${DTF_PS_HOSTS} --worker_hosts ${DTF_WORKER_HOSTS} \
--data_dir ${DTF_INPUT_PATH} --train_dir ${DTF_OUTPUT_PATH}'
```

### Using Docker
To execute the tasks as a Docker container, pass the Docker image name using
`--docker_image <image_name>`. The docker image is required to be accesible on
the execution host. In addition to variables in **TASK EXECUTION ENVIRONMENT**,
the following paths are mounted in the container.

- `HADOOP_HOME`, `HADOOP_CONF_DIR`, `JAVA_HOME`
- `DTF_INPUT_PATH` and `DTF_OUT_PATH` if they are not hdfs path.

## TASK EXECUTION ENVIRONMENT

   The user specified `task_command` will be executed as a YARN container
   allocated to the session. The following environment variables will be
   set for the `task_command` to consume.
- `DTF_TASK_SCRIPT`:

    Name of file which contains the content of the `script_file` specified
    during submission.

- `DTF_INPUT_PATH`:

    Input path specified during submission.

- `DTF_OUTPUT_PATH`:

    Output path specified during submission.

- `DTF_{JOBNAME}_HOSTS`:

    Variable with a list of host (and port) allocated to the job with name
    `{JOBNAME}`.

    - Format: "host1:port1,host2:port2,..."

    The number of host:port in the list should match one specified in
    `cluster-requirement`.  For example, `DTF_PS_HOSTS` and `DTF_WORKER_HOSTS`
    would be commonly used for PS and WORKER jobs.

- `DTF_TASK_JOB_NAME`:

    Name of job this task is assigned to.  See also `DTF_TASK_INDEX`.

- `DTF_TASK_INDEX`:

    Index of the job this task is assigned to. The tuple of `DTF_TASK_JOB_NAME`,
    and `DTF_TASK_INDEX` can also be used to cross reference with
    `DTF_{JOBNAME}_HOSTS`.  For example, to get the dynamic port allocated to
    this task.
