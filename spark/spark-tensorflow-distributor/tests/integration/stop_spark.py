import subprocess
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_workers',
    help='Number of docker workers',
    required=True,
)
args = parser.parse_args()
num_workers = int(args.num_workers)

# Stop spark daemon on worker nodes
for worker_index in range(1, num_workers + 1):
    subprocess.run(
        [
            'docker-compose exec -T --index={} worker '
            '/mnt/spark-tensorflow-distributor/tests/integration/stop_worker.sh'.format(worker_index)
        ],
        shell=True,
    )

# Stop spark daemon on master node
subprocess.run(
    [
        '/mnt/spark-tensorflow-distributor/tests/integration/stop_master.sh'
    ],
    shell=True,
)
