import pytest
import logging
import subprocess


from pyspark.sql import SparkSession


def pytest_addoption(parser):
    parser.addoption('--max_num_workers', action='store')

def restart_spark(num_workers, num_gpus_per_worker, max_num_workers):
    subprocess.run(
        [
            'cat /dev/null | tests/integration/restart_spark.sh --num_workers {} '
            '--num_gpus_per_worker {} '
            '--max_num_workers {}'.format(num_workers, num_gpus_per_worker, max_num_workers)
        ],
        shell=True,
        check=True,
    )

@pytest.fixture(scope='session')
def max_num_workers(request):
    return int(request.config.getoption("--max_num_workers"))

@pytest.fixture(scope='session')
def extra_spark_configs(request, autouse=True):
    if hasattr(request, 'param'):
        conf = request.param
    else:
        conf = {}
    with open('tests/integration/spark_conf/spark-custom.conf', 'w') as f:
        f.writelines(
            ['{} {}\n'.format(k, v) for k, v in conf.items()]
        )
    return conf

@pytest.fixture(scope='session', autouse=True)
def num_workers(request, max_num_workers):
    if not hasattr(request, 'param'):
        raise Exception(
            'num_workers is a required fixture for Spark '
            'TensorFlow Distributor tests, but test `{}` does not '
            'use it.'.format(request.node.name)
        )
    num_workers_value = request.param
    if num_workers_value > max_num_workers:
        raise Exception(
            'num_workers cannot be greater than {max_num_workers} but '
            'test `{test_name}` requested num_workers = `{num_workers}`.'
            'use it.'.format(
                max_num_workers=max_num_workers,
                test_name=request.node.name,
                num_workers=num_workers_value,
            )
        )
    return num_workers_value

@pytest.fixture(scope='session', autouse=True)
def num_gpus_per_worker(request):
    if not hasattr(request, 'param'):
        raise Exception(
            'num_gpus_per_worker is a required fixture for Spark '
            'TensorFlow Distributor tests, but test `{}` does not '
            'use it.'.format(request.node.name)
        )
    num_gpus_per_worker_value = request.param
    return num_gpus_per_worker_value

@pytest.fixture(scope='session', autouse=True)
def spark_session(num_workers, num_gpus_per_worker, max_num_workers, extra_spark_configs):
    restart_spark(num_workers, num_gpus_per_worker, max_num_workers)
    builder = SparkSession.builder.appName('Spark TensorFlow Distributor Tests')
    logging.getLogger().info('Creating spark session with the following confs.')
    with open('tests/integration/spark_conf/spark-defaults.conf') as f:
        for line in f:
            l = line.strip()
            if l:
                k, v = l.split(None, 1)
                builder.config(k, v)
    session = builder.getOrCreate()
    yield session
    session.stop()


