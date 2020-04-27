#!/usr/bin/env bash

set -e

chmod +x tests/integration/spark_conf/gpuDiscoveryScriptStub.sh
chmod +x tests/integration/start_spark.py
chmod +x tests/integration/stop_spark.py
chmod +x tests/integration/start_master.sh
chmod +x tests/integration/start_worker.sh
chmod +x tests/integration/stop_worker.sh
chmod +x tests/integration/stop_master.sh
chmod +x tests/integration/restart_spark.sh
python3 tests/integration/run_tests.py
