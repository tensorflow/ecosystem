import os
import pytest


from spark_tensorflow_distributor import MirroredStrategyRunner


@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
def test_equal_gpu_allocation(num_workers, num_gpus_per_worker):
    def train_fn():
        import os
        from pyspark import BarrierTaskContext
        context = BarrierTaskContext.get()
        num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        return [int(e) for e in context.allGather(str(num_gpus))]

    runner = MirroredStrategyRunner(num_slots=2, gpu_resource_name='gpu')
    assert runner.get_num_tasks() == 1
    gpus_used_by_each_task = runner.run(train_fn)
    assert gpus_used_by_each_task == [2]

    runner = MirroredStrategyRunner(num_slots=4, gpu_resource_name='gpu')
    assert runner.get_num_tasks() == 1
    gpus_used_by_each_task = runner.run(train_fn)
    assert gpus_used_by_each_task == [4]

    runner = MirroredStrategyRunner(num_slots=6, gpu_resource_name='gpu')
    assert runner.get_num_tasks() == 2
    gpus_used_by_each_task = runner.run(train_fn)
    assert gpus_used_by_each_task == [3, 3]

    runner = MirroredStrategyRunner(num_slots=8, gpu_resource_name='gpu')
    assert runner.get_num_tasks() == 2
    gpus_used_by_each_task = runner.run(train_fn)
    assert gpus_used_by_each_task == [4, 4]

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
def test_zero_num_slots(num_workers, num_gpus_per_worker):
    def train_fn():
        import os
        return os.environ['CUDA_VISIBLE_DEVICES']

    with pytest.raises(ValueError):
        result = MirroredStrategyRunner(num_slots=0, gpu_resource_name='gpu').run(train_fn)

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize('num_slots', [1, 2, 3])
def test_local_run(num_workers, num_gpus_per_worker, num_slots):
    def train_fn():
        import os
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    old_cuda_state = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    result = MirroredStrategyRunner(num_slots=-num_slots, gpu_resource_name='gpu').run(train_fn)
    gpus_on_the_driver = [str(e) for e in range(num_slots)]
    assert result == num_slots
    new_cuda_state = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert old_cuda_state == new_cuda_state

@pytest.mark.parametrize('num_workers', [2], indirect=True)
@pytest.mark.parametrize('num_gpus_per_worker', [4], indirect=True)
@pytest.mark.parametrize('extra_spark_configs', [{'spark.ssl.enabled': 'true'}], indirect=True)
def test_run_on_ssl_cluster(num_workers, num_gpus_per_worker, extra_spark_configs):
    def train_fn():
        import os
        return os.environ['CUDA_VISIBLE_DEVICES']

    with pytest.raises(Exception):
        MirroredStrategyRunner(num_slots=2, gpu_resource_name='gpu').run(train_fn)
