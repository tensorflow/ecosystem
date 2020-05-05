#!/usr/bin/env bash

set -e

${SPARK_HOME}/sbin/spark-daemon.sh \
    start org.apache.spark.deploy.master.Master 1 \
    --properties-file /mnt/spark-tensorflow-distributor/tests/integration/spark_conf/spark-defaults.conf \
    -h master
