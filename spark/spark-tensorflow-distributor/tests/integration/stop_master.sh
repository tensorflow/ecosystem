#!/usr/bin/env bash

set -e

${SPARK_HOME}/sbin/spark-daemon.sh \
    stop org.apache.spark.deploy.master.Master 1

exit 0
