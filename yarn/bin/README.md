Submit Command Line
-------------------

```sh
% ytf-submit -h
NAME
    ytf-submit - Submit a TensorFlow session to Apache Hadoop YARN

    This tool submits a YARN application master, resposible to allocate
    required resources, and execute corresponding tasks.

SYNOPSIS
    Usage: ./ytf-submit [OPTIONS] -r <cluster_requirement>  <task_command>

DESCRIPTION
    task_command
        The command to be execute for each of the task of the session. The two
        environment variables DTF_TASK_JOB_NAME and DTF_TASK_INDEX will be set
        before the task is executed.
        See aslo TASK EXECUTION ENVIRONMENT

    -r, --cluster_requirement <requirement>
        Specify cluster requiement for the session.
            Format: <job_name1>:<num_tasks1>,<job_name2>:<num_task2>,...
            Example: "ps:2,worker:4"
        See also TASK EXECUTION ENVIRONMENT

    Additional options:

    -c, --task_vcores <vcores>
        General form to specify number of vcores required by each of the task.
        DEFAULT=1

    -c, --task_vcores <job_name>:<vcores>
        **NOT IMPLEMENTED YET**
        Job-level form to specify number of vcores required by tasks in specific
        job. Overrides "general" form.

    -c, --task_vcores <job_name>[<task_index>]:<vcores>
        **NOT IMPLEMENTED YET**
        Task-level form to specify number of vcores required by a specific task.
        Overrides both "job-level" and "general" form.

    -m, --task_memory <memory>
        General form to specify amount of memory required by each of task; with
        unit in MB. DEFAULT=8192

    -m, --task_memory <job_name>:<memory>
        **NOT IMPLEMENTED YET**
        Job-level form to specify amount of memory required by tasks in specific
        job. Overrides "general" form.

    -m, --task_memory <job_name>[<task_index]:<memory>
        **NOT IMPLEMENTED YET**
        Task-level form to specify amount of memory required by a specific task.
        Overrides both "job-level" and "general" form.

    -i, --input input_path
        Input path, this variable is not interpreted by YARN-DTF at the
        momement, it serve as a convenience. Its value will be set as
        environment variable {DTF_INPUT_PATH} in tasks execution environment.
        DEFAULT=

    -o, --output <output_path>
        Output path, this variable is not interpreted by YARN-DTF at the
        momement, it serve as a convenience. Its value will be set as
        environment variable {DTF_OUTPUT_PATH} in tasks  execution environment.

        However, when TensorBoard integration is enabled, this option becomes
        mandatory. See also --tensorborad option.

        Its value will be set as environment variable {DTF_OUTPUT_PATH} in tasks
        execution environment.

    -s, --script <script_file>
        A local script file to be transfer to tasks execution environment, where
        a file named by variable {DTF_TASK_SCRIPT} will contain the content of
        the script file. For example, if the script is a Python script,
        the execution command can be written as "python ${DTF_TASK_SCRIPT} ..."

    -t, --tensorboard
        Enable TensorBoard integration. When enabled, YARN-DTF will start an
        additional YARN container as tensorboard with output path specified in
        --output option.  DEFAULT=disabled

    --docker_image <image_name>
        Enable tasks to be executed as a docker container. The docker image is
        required to be accesible on the execution host. In addition to variables
        in TASK EXECUTION ENVIRONMENT, the following paths are mounted in
        container to the execution host.

          HADOOP_HOME, HADOOP_CONF_DIR, JAVA_HOME.
          DTF_INPUT_PATH and DTF_OUT_PATH if they are not hdfs path.

    -q, --queue
        Specify which YARN queue to submit this session to.
        DEFAULT=default

    -n, --name
        Name of this session, will be used as name of YARN application.
        DEFAULT=TensorFlow

    --client
        **NOT IMPLEMENTED YET**
        Specify if an additional task should be started on locally. This
        would be useful if user interaction is required.

        This task will same execution environment as the rest of the tasks,
        and will be assigned with DTF_TASK_JOB_NAME=client and DTF_TASK_INDEX=0;
        however, will not be part of the TensorFlow cluster and dynamic port
        allocation would not apply.

TASK EXECUTION ENVIRONMENT

   The user specified 'task_command' will be executed as a YARN container
   allocated to the session. The following environment variables will be
   set for the 'task_command' to consume.

   DTF_TASK_SCRIPT:
       Name of file which contains the content of the 'script_file' specified
       during submission.

   DTF_INPUT_PATH:
       Input path specified during submission.

   DTF_OUTPUT_PATH:
       Output path specified during submission.

   DTF_{JOBNAME}_HOSTS:
       Variable with a list of host (and port) allocated to the job with name
       {JOBNAME}.
           Format: "host1:port1,host2:port2,..."
       The number of host:port in the list should match one specified in
       "cluster-requirement".  For example, DTF_PS_HOSTS and DTF_WORKER_HOSTS
       would be commonly used for PS and WORKER jobs.

   DTF_TASK_JOB_NAME:
       Name of job this task is assigned to.  See also DTF_TASK_INDEX.

   DTF_TASK_INDEX
       Index of the job this task is assigned to.
       The tuple of DTF_TASK_JOB_NAME, and DTF_TASK_INDEX can also be used
       to cross reference with DTF_{JOBNAME}_HOSTS.  For example, to get the
       dynamic port allocated to this task.
```
