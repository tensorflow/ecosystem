"""
This script sets the contents of spark-defaults.conf and
gpuDiscoveryScriptStub.sh for the purpose of integration
testing. It does so based on the arg inputs, spark-base.conf
which is static, and spark-custom.conf which is dynamically
changed by tests.
"""


import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_workers',
    help='Number of workers to be set in the spark conf'
)
parser.add_argument(
    '--num_gpus_per_worker',
    help='Number of gpus on each worker to be set in the spark conf'
)
args = parser.parse_args()
num_workers = int(args.num_workers)
num_gpus_per_worker = str(args.num_gpus_per_worker)

conf = {}

with open('tests/integration/spark_conf/spark-base.conf', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        l = lines[i].strip()
        if l:
            k, v = l.split(None, 1)
            conf[k] = v

with open('tests/integration/spark_conf/spark-custom.conf', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        l = lines[i].strip()
        if l:
            k, v = l.split(None, 1)
            conf[k] = v

with open('tests/integration/spark_conf/spark-defaults.conf', 'w') as f:
    f.writelines(
        ['{} {}\n'.format(k, v) for k, v in conf.items()]
    )

with open('tests/integration/spark_conf/gpuDiscoveryScriptStub.sh', 'w+') as f:
        original_file_content = f.read()
        gpus = '","'.join(str(e) for e in range(int(num_gpus_per_worker)))
        cmd = "echo '{\"name\": \"gpu\", \"addresses\":[\"" + gpus + "\"]}'"
        f.writelines([
            '#!/usr/bin/env bash\n',
            '\n',
            cmd,
        ])
