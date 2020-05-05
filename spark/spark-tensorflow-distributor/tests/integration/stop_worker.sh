#!/usr/bin/env bash

set -e

${SPARK_HOME}/sbin/spark-daemon.sh \
    stop org.apache.spark.deploy.worker.Worker 1

exit 0
