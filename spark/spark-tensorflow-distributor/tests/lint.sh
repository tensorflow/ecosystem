#!/usr/bin/env bash

set -e

flake8 spark_tensorflow_distributor --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 spark_tensorflow_distributor --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
