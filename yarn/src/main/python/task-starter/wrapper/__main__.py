#!/usr/bin/env python
"""
A wrapper that launches a TensorFlow program. This is launched by the
Application master. Input (Using ENV vars):

DTF_APPLICATION_ID
DTF_TASK_PROGRAM
DTF_TASK_JOB_NAME
DTF_TASK_INDEX
DTF_INPUT_PATH
DTF_OUTPUT_PATH
DTF_ZK_HOSTS
DTF_SERVICE_CLASS

JAVA_HOME
HADOOP_HOME
CONTAINER_ID
"""
from __future__ import print_function

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
from threading import Event

from kazoo.client import KazooClient

DTF_APPLICATION_ID = "DTF_APPLICATION_ID"
DTF_SERVICE_CLASS = "DTF_SERVICE_CLASS"
DTF_ZK_HOSTS = "DTF_ZK_HOSTS"

DTF_TASK_PROGRAM = "DTF_TASK_PROGRAM"
DTF_INPUT_PATH = "DTF_INPUT_PATH"
DTF_OUTPUT_PATH = "DTF_OUTPUT_PATH"
DTF_TASK_JOB_NAME = "DTF_TASK_JOB_NAME"
DTF_TASK_INDEX = "DTF_TASK_INDEX"
DTF_DOCKER_IMAGE = "DTF_DOCKER_IMAGE"

JAVA_HOME = "JAVA_HOME"
HADOOP_HOME = "HADOOP_HOME"
USER = "USER"
CONTAINER_ID = "CONTAINER_ID"

AM_PATH = "/registry/users/{user}/{service_class}/{app_id}"
CN_PATH = "/components/{container_id}"

LAUNCH_CMD = """
source $HADOOP_HOME/libexec/hadoop-config.sh ;
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server ;
export CLASSPATH=$($HADOOP_HDFS_HOME/bin/hdfs classpath --glob) ; {task_program}
"""

DOCKER_ENGINE_PREFIX = """
MAPPING="" ;
if [[ "${{DTF_INPUT_PATH}}" != "" && "${{DTF_INPUT_PATH}}" != hdfs://* ]]; then
   MAPPING=${{MAPPING}}" -v ${{DTF_INPUT_PATH}}:${{DTF_INPUT_PATH}}"
fi ;
if [[ "${{DTF_OUTPUT_PATH}}" != "" && "${{DTF_OUTPUT_PATH}}" != hdfs://* ]]; then
   MAPPING=${{MAPPING}}" -v ${{DTF_OUTPUT_PATH}}:${{DTF_OUTPUT_PATH}}"
fi ;
/usr/bin/docker run --rm -u $(id -u $USER):$(id -g $USER) --net=host --name=${{CONTAINER_ID}} \
-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v ${{LOCAL_DIRS}}:${{LOCAL_DIRS}} -v ${{LOG_DIRS}}:${{LOG_DIRS}} \
-v ${{HADOOP_HOME}}:${{HADOOP_HOME}} -v ${{HADOOP_CONF_DIR}}:${{HADOOP_CONF_DIR}} \
-v ${{JAVA_HOME}}:${{JAVA_HOME}} ${{MAPPING}} -e JAVA_HOME -e HADOOP_HOME -e LD_LIBRARY_PATH \
-e CLASSPATH -e HADOOP_CONF_DIR -e HADOOP_HDFS_HOME -e LOCAL_DIRS -e LOG_DIRS -e CONTAINER_ID \
{dtf_vars} {docker_image} bash -c 'cd ${{LOCAL_DIRS}}/${{CONTAINER_ID}} ; {task_program}'
"""

SLEEP_TIME = 1
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG = logging.getLogger('wrapper')

def generate_docker_cmd(docker_image, task_program):
    "Generate docker command"
    dtf_vars = ""
    for key in os.environ:
        if key.startswith('DTF_'):
            dtf_vars += " -e %s" % key

    return DOCKER_ENGINE_PREFIX.format(
        dtf_vars=dtf_vars,
        docker_image=docker_image,
        task_program=task_program)

def launch_prog(opts):
    """
    Launches the main program
    """
    task_program = 'source %s' % opts[DTF_TASK_PROGRAM]
    docker_image = os.getenv(DTF_DOCKER_IMAGE)
    if docker_image is None:
        cmd = LAUNCH_CMD.format(task_program=task_program)
    else:
        cmd = LAUNCH_CMD.format(task_program=generate_docker_cmd(docker_image, task_program))

    LOG.info("Running %s", cmd)
    proc = subprocess.Popen(
        ['bash', '-c', cmd], stdout=sys.stdout.fileno(),
        stderr=sys.stderr.fileno())

    proc.communicate()

    LOG.info('Task program return code=' + str(proc.returncode))

    return proc.returncode

def get_socket(ipaddr='', port=0):
    "Reserve a port and returns port and the socket"
    try:
        sock = socket.socket()
        sock.bind((ipaddr, port))
        port = sock.getsockname()[1]
        return port, sock
    except socket.error as err:
        LOG.error("Encountered error while trying to open socket - " + str(err))
        raise err

def get_envs():
    """
    Retrives input data frome env vars
    """
    opts = {}
    keys = [
        DTF_APPLICATION_ID, DTF_ZK_HOSTS, DTF_SERVICE_CLASS,
        DTF_TASK_PROGRAM, DTF_TASK_JOB_NAME, DTF_TASK_INDEX,
        HADOOP_HOME, JAVA_HOME, USER, CONTAINER_ID]
    for key in keys:
        env = os.getenv(key)
        if env is None:
            LOG.error("%s env var not found", key)
            raise KeyError(key)
        opts[key] = env
    return opts

def zookeeper_get_spec(opts, port_data):
    """
    Uses zookeeper to register the port, and get cluster_spec
    """
    service_class = opts[DTF_SERVICE_CLASS]
    app_id = opts[DTF_APPLICATION_ID]
    container_id = opts[CONTAINER_ID]
    user = opts[USER]
    zk_host = opts[DTF_ZK_HOSTS]
    zk_client = KazooClient(hosts=zk_host)
    zk_client.start()

    am_path = AM_PATH.format(service_class=service_class, app_id=app_id, user=user)
    cn_path = am_path + CN_PATH.format(container_id=container_id)
    zk_client.ensure_path(cn_path)

    port_data = port_data.encode('ascii')

    zk_client.set(cn_path, port_data)

    LOG.info("Wating for cluster spec")

    event = Event()
    data, _ = zk_client.get(am_path, watch=lambda _: event.set())
    if data == b'':
        event.wait()

    data, _ = zk_client.get(am_path)

    data = data.decode('ascii')
    spec = json.loads(data)
    return spec

def main():
    """
    Main function
    """

    if sys.argv[-1] == "--debug":
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    opts = get_envs()
    LOG.debug(os.environ)

    port, sock = get_socket()

    LOG.debug("Reserverd port number %d", port)

    port_data = json.dumps({
        "type": "JSONServiceRecord",
        "description": "YARN Distributed TensorFlow Container",
        "external": [],
        "internal": [],
        "yarn:persistence": "container",
        "yarn:id": opts[CONTAINER_ID],
        "task_job_name": opts[DTF_TASK_JOB_NAME],
        "task_job_index": opts[DTF_TASK_INDEX],
        "task_port": str(port),
    })

    cluster_spec = zookeeper_get_spec(opts, port_data)

    LOG.debug("Spec %s", cluster_spec)
    for key in cluster_spec.keys():
        if key.startswith('DTF_'):
            os.environ[key] = cluster_spec[key]
    sock.close()
    return launch_prog(opts)

def exit_handler(signum, _):
    """
    Capture exit signal
    """
    LOG.info('Killed by signal %d', signum)
    sys.exit(0)

def kill_docker():
    "Cleanup docker container"
    docker_image = os.getenv(DTF_DOCKER_IMAGE)
    container_id = os.getenv(CONTAINER_ID)
    if container_id and docker_image:
        LOG.info('Killing docker image %s with id %s', docker_image, container_id)
        kill_cmd = "docker ps | awk '/{container_id}/ {{print $1}}' | xargs -r docker kill"
        proc = subprocess.Popen(["bash", "-c", kill_cmd.format(container_id=container_id)])
        proc.communicate()

if __name__ == '__main__':
    atexit.register(kill_docker)
    signal.signal(signal.SIGTERM, exit_handler)
    sys.exit(main())
