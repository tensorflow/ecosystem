#!/usr/bin/env bash

set -e

yapf --recursive --diff spark_tensorflow_distributor
pylint spark_tensorflow_distributor
