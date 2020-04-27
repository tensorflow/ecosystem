#!/usr/bin/env bash

set -e

${SPARK_HOME}/sbin/spark-daemon.sh \
    start org.apache.spark.deploy.worker.Worker 1 \
    --properties-file /mnt/spark-tensorflow-distributor/tests/integration/spark_conf/spark-defaults.conf \
    spark://master:7077
