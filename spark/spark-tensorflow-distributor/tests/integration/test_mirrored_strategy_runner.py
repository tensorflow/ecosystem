import math
import os
import pytest
from pyspark.sql import SparkSession

from spark_tensorflow_distributor import MirroredStrategyRunner
from unittest import mock


@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize(
    'extra_spark_configs',
    [{'spark.task.resource.gpu.amount': '1', 'spark.cores.max': 8, 'spark.executor.cores': 4},
     {'spark.task.resource.gpu.amount': '2', 'spark.cores.max': 4, 'spark.executor.cores': 2},
     {'spark.task.resource.gpu.amount': '4', 'spark.cores.max': 2, 'spark.executor.cores': 1}],
    indirect=True,
)
def test_equal_gpu_allocation(num_workers, num_gpus_per_worker):
    def train_fn():
        import os
        from pyspark import BarrierTaskContext
        context = BarrierTaskContext.get()
        cuda_state = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_state:
            num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            num_gpus = 0
        return [int(e) for e in context.allGather(str(num_gpus))]

    for num_slots in [2, 4, 6, 8]:
        runner = MirroredStrategyRunner(num_slots=num_slots)
        task_gpu_amount = int(runner.sc.getConf().get('spark.task.resource.gpu.amount'))
        expected_num_task = math.ceil(num_slots / task_gpu_amount)
        assert runner.get_num_tasks() == expected_num_task
        gpus_used_by_each_task = runner.run(train_fn)
        assert gpus_used_by_each_task == [
            (num_slots // expected_num_task) + (i < (num_slots % expected_num_task))
            for i in range(expected_num_task)
        ]

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
def test_zero_num_slots(num_workers, num_gpus_per_worker):
    with pytest.raises(ValueError):
        result = MirroredStrategyRunner(num_slots=0).run(lambda: None)

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize('num_slots', [1, 2, 3])
@pytest.mark.parametrize('old_cuda_state', [None, '10,11,12,13'])
def test_local_run(num_workers, num_gpus_per_worker, num_slots, old_cuda_state):
    def train_fn():
        import os
        return os.environ['CUDA_VISIBLE_DEVICES']

    if old_cuda_state is not None:
        mock_env = {'CUDA_VISIBLE_DEVICES': old_cuda_state}
    else:
        mock_env = {}

    with mock.patch.dict(os.environ, mock_env, clear=True):
        task_cuda_env = MirroredStrategyRunner(num_slots=num_slots, local_mode=True, gpu_resource_name='gpu').run(train_fn)
        gpu_set = {int(i) for i in task_cuda_env.split(',')}
        assert len(gpu_set) == num_slots
        for gpu_id in gpu_set:
            if old_cuda_state is not None:
                assert gpu_id in [10, 11, 12, 13]
            else:
                assert gpu_id in [0, 1, 2, 3]
        new_cuda_state = os.environ.get('CUDA_VISIBLE_DEVICES')
        assert old_cuda_state == new_cuda_state

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize(
    'extra_spark_configs',
    [{'spark.ssl.enabled': 'true'}],
    indirect=True,
)
def test_run_on_ssl_cluster(num_workers, num_gpus_per_worker, extra_spark_configs):
    with pytest.raises(Exception):
        MirroredStrategyRunner(num_slots=2, gpu_resource_name='gpu').run(lambda: None)

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize(
    'extra_spark_configs',
    [{'spark.ssl.enabled': 'true', 'tensorflow.spark.distributor.ignoreSsl': 'true'}],
    indirect=True,
)
def test_run_on_ssl_cluster_override(num_workers, num_gpus_per_worker, extra_spark_configs):
    MirroredStrategyRunner(num_slots=2, gpu_resource_name='gpu').run(lambda: None)

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
def test_cpu_training_with_gpus(num_workers, num_gpus_per_worker):
    def train_fn():
        from pyspark import BarrierTaskContext
        context = BarrierTaskContext.get()
        cuda_state = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_state:
            num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            num_gpus = 0
        return [int(e) for e in context.allGather(str(num_gpus))]

    runner = MirroredStrategyRunner(num_slots=2, use_gpu=False)
    assert runner.get_num_tasks() == 2
    gpus_used_by_each_task = runner.run(train_fn)
    assert gpus_used_by_each_task == [0, 0]

@pytest.mark.parametrize('num_workers', [1], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize(
    'extra_spark_configs',
    [{'spark.executorEnv.CUDA_VISIBLE_DEVICES': '10,11,12,13'}],
    indirect=True,
)
def test_spark_task_cuda_devices_env_support(num_workers, num_gpus_per_worker):
    def train_fn():
        import os
        return os.environ['CUDA_VISIBLE_DEVICES']

    for num_slots in [2, 3, 4]:
        runner = MirroredStrategyRunner(num_slots=num_slots)
        task_cuda_env = runner.run(train_fn)
        gpu_set = {int(i) for i in task_cuda_env.split(',')}
        assert len(gpu_set) == num_slots
        for gpu_id in gpu_set:
            assert gpu_id in [10, 11, 12, 13]
