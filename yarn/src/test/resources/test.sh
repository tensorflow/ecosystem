#!/usr/bin/bash
set -u -e
export USER=test_user
export HADOOP_HOME=~/hadoop
export CONTAINER_ID=234
export DTF_ZK_HOSTS=localhost:2181
export DTF_SERVICE_CLASS=yarn-dtf
export DTF_APPLICATION_ID=123
export DTF_TASK_PROGRAM="test_task.sh"
export DTF_TASK_JOB_NAME=ps
export DTF_TASK_INDEX=0
export DTF_INPUT_PATH=.
export DTF_OUTPUT_PATH=.

(
python - <<EOF
from kazoo.client import KazooClient
import json
import time
zk_client = KazooClient(hosts='$ZK_HOST')
zk_client.start()
path = '/registry/users/$USER/$DTF_SERVICE_CLASS/$DTF_APPLICATION_ID'
cn_path = path + '/components/$CONTAINER_ID'
while not zk_client.exists(cn_path):
    time.sleep(0.1)
data, _ = zk_client.get(cn_path)
data = json.loads(data)
am_data = {
    "DTF_PS_HOSTS": "localhost:" + data["task_port"],
    "DTF_WORKER_HOSTS": "localhost:" + data["task_port"]
}
zk_client.set(path, json.dumps(am_data))
time.sleep(1)
zk_client.delete(path, recursive=True)
EOF
) &

python wrapper.py --debug
