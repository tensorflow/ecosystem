import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_workers',
    help='Number of docker workers',
    required=True,
)
args = parser.parse_args()
num_workers = int(args.num_workers)

# Start spark daemon from master node
subprocess.run(
    [
        '/mnt/spark-tensorflow-distributor/tests/integration/start_master.sh'
    ],
    shell=True,
)

# Start spark daemon on worker nodes
for worker_index in range(1, num_workers + 1):
    print(f'Starting worker {worker_index}')
    subprocess.run(
        [
            'docker-compose exec -T --index={} worker '
            'tests/integration/start_worker.sh'.format(worker_index)
        ],
        shell=True,
    )
