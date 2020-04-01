#!/usr/bin/env bash

set -e

if [ "$1" != "--num_workers" ]; then
    echo 'Usage: restart_spark.sh --num_workers <int> --num_gpus_per_worker <int> --max_num_workers <int>';
    exit 1;
fi

if [ "$3" != "--num_gpus_per_worker" ]; then
    echo 'Usage: restart_spark.sh --num_workers <int> --num_gpus_per_worker <int> --max_num_workers <int>';
    exit 1;
fi

if [ "$5" != "--max_num_workers" ]; then
    echo 'Usage: restart_spark.sh --num_workers <int> --num_gpus_per_worker <int> --max_num_workers <int>';
    exit 1;
fi

python3 tests/integration/stop_spark.py --num_workers $6 || true && \
python3 tests/integration/set_spark_conf.py --num_workers $2 --num_gpus_per_worker $4 && \
python3 tests/integration/start_spark.py --num_workers $2
