"""
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import math
import os
import random
import re
import sys

from pyspark.sql import SparkSession


def _get_logger(name):
    """
    Gets a logger by name, or creates and configures it for the first time.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    return logger


class MirroredStrategyRunner:
    """
    MirroredStrategyRunner runs TensorFlow deep learning training jobs
    on Spark clusters. It trains synchronously by mirroring the model's
    variables among all the workers and shares the gradient updates
    in a decentralized manner. MirroredStrategyRunner can take a regular
    TensorFlow program with no special distributed training code and
    launch it as a distributed training program.

    See more at https://www.tensorflow.org/guide/distributed_training
    """

    def __init__(self,
                 *,
                 num_slots,
                 local_mode=False,
                 use_gpu=True,
                 gpu_resource_name='gpu',
                 use_custom_strategy=False):
        """
        Args:
            num_slots: Total number of GPUs or CPU only Spark tasks that
                participate in distributed training. For example, if
                num_slots = 16 we train on the Spark cluster with 16 GPUs
                if doing GPU training, or with 16 Spark tasks if doing
                CPU training. num_slots cannot be less than or equal to 0.
                Note that when doing CPU training, Spark will still be
                subject to any GPU-aware scheduling confs set
                in the Spark configuration. Note also that for GPU training,
                num_slots will limit the number of GPUs used for training
                even if more are available, so that exactly num_slots GPUs
                are used in total. Spark does not restrict CPU cores for
                tasks and so for CPU training, num_slots rarely needs to
                be greater than the number of workers and in local mode
                set num_slots=1.
            local_mode: If True, the training function will be run locally
                on the driver. If False training is distributed among the
                workers.
            use_gpu: If True, training is done with GPUs using Spark
                resource scheduling with the gpu_resource_name parameter
                as the resource name. If False, do CPU only training.
            gpu_resource_name: The name of the Spark resource scheduling
                GPU resource. It may be set under
                `spark.executor.resource.{gpu_resource_name}`,
                `spark.task.resource.{gpu_resource_name}`,
                `spark.driver.resource.{gpu_resource_name}`, and
                `spark.worker.resource.{gpu_resource_name}` in the Spark
                conf. Contact the cluster administrator to set these
                configurations. The resource should be configured with
                a discovery script that is formatted according to the
                Spark configuration docs. Make sure
                `spark.driver.resource.{gpu_resource_name}.discoveryScript` and
                `spark.driver.resource.{gpu_resource_name}.discoveryScript` are
                also set in the Spark conf. In particular, the GPU addresses
                should be zero indexed. For example, the output of the
                discovery script for 3 GPUs with gpu_resource_name='gpu'
                would be `{"name": "gpu", "addresses":["0","1","2"]}`.
                See an example discovery script: `github.com/apache/spark/blob/
                master/examples/src/main/scripts/getGpusResources.sh`.
            use_custom_strategy: When true, the training function passed to the
                MirroredStrategyRunner.run method must construct and use its
                own tensorflow.distribute.Strategy() object. When false,
                MirroredStrategyRunner constructs one for the user and wraps
                the training function in the strategy context, allowing
                the user to provide non-distributed TensorFlow code that is
                executed as distributed code.

                Example with use_custom_strategy=True:

                    def train_fn():
                        import tensorflow as tf
                        strategy = tf.distribute.experimental \
                            .MultiWorkerMirroredStrategy()
                        with strategy.scope():
                            # training code

                Example with use_custom_strategy=False:

                    def train_fn():
                        import tensorflow as tf
                        # training code
        """
        self._logger = _get_logger(self.__class__.__name__)
        self._num_slots = num_slots
        if num_slots <= 0:
            raise ValueError(f'num_slots is set to {num_slots} but '
                             'cannot be less than or equal to 0.')
        self._local_mode = local_mode
        self._use_gpu = use_gpu
        self._gpu_resource_name = gpu_resource_name
        self._use_custom_strategy = use_custom_strategy
        if self._use_gpu:
            self._logger.info('Doing GPU training...')
        else:
            self._logger.info('Doing CPU training...')
        spark = SparkSession.builder.getOrCreate()
        self.sc = spark.sparkContext
        if self._local_mode is True:
            self._logger.warning('MirroredStrategyRunner will run in '
                                 'local mode on the driver node. '
                                 'There would be resource contention if '
                                 'the driver also runs other workloads.')
            self._num_tasks = None
        else:
            self._num_tasks = self.get_num_tasks()
            self._logger.info(f'Will run with {self._num_tasks} Spark tasks.')

    def get_num_tasks(self):
        """
        Returns:
            The number of Spark tasks to use for distributed training
        """
        if self._use_gpu:
            key = f'spark.task.resource.{self._gpu_resource_name}.amount'
            if not self.sc.getConf().contains(key):
                raise Exception('Your cluster might not have '
                                'Spark GPU-aware scheduling enabled, '
                                'please contact your cluster administrator.'
                                f'The conf `{key}` was not found '
                                'in the Spark configuration.')
            task_gpu_amount = int(self.sc.getConf().get(key))
            if task_gpu_amount < 1:
                raise ValueError(f'The Spark conf `{key}` has a value '
                                 f'of {task_gpu_amount} but it '
                                 'should not have a value less than 1.')
            return math.ceil(self._num_slots / task_gpu_amount)
        return self._num_slots

    def run(self, train_fn, **kwargs):
        """
        Args:
            train_fn: Function that contains TensorFlow training code.
                If it constructs its own tensorflow.distribute.Strategy
                object, then construct MirroredStrategyRunner with
                use_custom_strategy set to `True`.
            kwargs: keyword arguments passed to the training function
                at invocation time. When train_fn is called, it will
                be called with train_fn(**kwargs).
        Returns:
            Return value of the training function
            from the chief training worker (partition ID 0) in
            distributed mode, or the direct return value of train_fn in
            local mode.
        """
        spark_task_program = self._get_spark_task_program(train_fn, **kwargs)

        # Run in local mode
        if self._local_mode:
            old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES',
                                                      '')
            cuda_state_was_set = 'CUDA_VISIBLE_DEVICES' in os.environ
            try:
                if self._use_gpu:
                    # TODO: handle the case that driver gpu resources
                    #       is not properly configured
                    gpus_owned = MirroredStrategyRunner._get_gpus_owned(
                        self.sc.resources, self._gpu_resource_name)
                    num_gpus_owned = len(gpus_owned)
                    if self._num_slots > num_gpus_owned:
                        raise ValueError(
                            f'{self._num_slots} slots were requested '
                            'for local training with '
                            f'GPU training but only {num_gpus_owned} GPUs '
                            'were available.')
                    # TODO: Check GPU utilization to avoid resource contention
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                        str(e)
                        for e in random.sample(gpus_owned, self._num_slots))
                else:
                    if self._num_slots > 1:
                        raise ValueError(f'Cannot run with more than 1 CPU '
                                         'machine in local mode. '
                                         'Try setting num_slots to -1.')
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                result = MirroredStrategyRunner._run_tensorflow_program(
                    train_fn, self._use_custom_strategy, **kwargs)
            finally:
                if cuda_state_was_set:
                    os.environ[
                        'CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
                else:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            return result

        # Run in distributed mode
        self._check_encryption()
        self._logger.info('Distributed training in progress...')
        self._logger.info(
            'View Spark executor stderr logs to inspect training...')
        result = self.sc.parallelize(range(self._num_tasks), self._num_tasks) \
            .barrier() \
            .mapPartitions(spark_task_program) \
            .collect()[0]
        self._logger.info(f'Training with {self._num_slots} slots is complete!')
        return result

    @staticmethod
    def _get_gpus_owned(resources, gpu_resource_name):
        """
        Gets the number of GPUs that Spark scheduled to the calling task.

        Returns:
            The number of GPUs that Spark scheduled to the calling task.
        """
        if gpu_resource_name in resources:
            addresses = resources[gpu_resource_name].addresses
            pattern = re.compile('^[1-9][0-9]*|0$')
            if any(not pattern.match(address) for address in addresses):
                raise ValueError(f'Found GPU addresses {addresses} which '
                                 'are not all in the correct format '
                                 'for CUDA_VISIBLE_DEVICES, which requires '
                                 'integers with no zero padding.')
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpu_indices = list(map(int, addresses))
                gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                gpu_owned = [gpu_list[i] for i in gpu_indices]
                return gpu_owned
            return addresses
        raise ValueError(
            f'The provided GPU resource name `{gpu_resource_name}` '
            'was not found in the '
            f'context resources. Contact your cluster administrator '
            'to make sure that the '
            f'spark.task.resource.{gpu_resource_name}, '
            f'spark.worker.resource.{gpu_resource_name}, '
            f'spark.executor.resource.{gpu_resource_name}, and '
            f'spark.driver.resource.{gpu_resource_name} confs are '
            'set and that the '
            f'GPU resource name `{gpu_resource_name}` matches '
            'those confs correctly.')

    # Runs the training function
    @staticmethod
    def _run_tensorflow_program(train_fn, use_custom_strategy, **kwargs):
        if not use_custom_strategy:
            import tensorflow as tf
            from tensorflow.python.eager import context
            # Reset the tenosrflow eager context to clear
            # leftover state from previous runs
            # Disable protected member access rule
            # pylint: disable=protected-access
            context._reset_context()
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            with strategy.scope():
                return train_fn(**kwargs)
        return train_fn(**kwargs)

    def _get_spark_task_program(self, train_fn, **kwargs):
        num_slots = self._num_slots
        use_custom_strategy = self._use_custom_strategy
        gpu_resource_name = self._gpu_resource_name
        num_tasks = self._num_tasks
        use_gpu = self._use_gpu
        run_tensorflow_program = MirroredStrategyRunner._run_tensorflow_program

        # Spark task program
        def wrapped_train_fn(_):
            import json
            import logging
            import os
            import socket
            from contextlib import closing
            from pyspark import BarrierTaskContext

            # Sets the TF_CONFIG env var so TF servers
            # can communicate with each other
            def set_tf_config(context):
                addrs = [
                    e.address.split(':')[0] for e in context.getTaskInfos()
                ]
                my_addr = addrs[context.partitionId()]
                with closing(socket.socket(socket.AF_INET,
                                           socket.SOCK_STREAM)) as my_sock:
                    my_sock.bind(('', 0))
                    _, my_port = my_sock.getsockname()
                    my_endpoint = "{}:{}".format(my_addr, my_port)
                    worker_endpoints = context.allGather(my_endpoint)
                cluster = {'worker': worker_endpoints}
                tf_config = {
                    'cluster': cluster,
                    'task': {
                        'type': 'worker',
                        'index': context.partitionId()
                    }
                }
                os.environ['TF_CONFIG'] = json.dumps(tf_config)

            # Sets the CUDA_VISIBLE_DEVICES env var so only
            # the appropriate GPUS are used
            def set_gpus(context):
                gpus_owned = MirroredStrategyRunner._get_gpus_owned(
                    context.resources(), gpu_resource_name)

                my_num_gpus = (num_slots //
                               num_tasks) + (context.partitionId() <
                                             (num_slots % num_tasks))
                gpu_addresses = [
                    str(e) for e in random.sample(gpus_owned, my_num_gpus)
                ]
                logging.info(f'Using GPU addresses: {gpu_addresses}')
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_addresses)

            context = BarrierTaskContext.get()
            if use_gpu:
                set_gpus(context)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            set_tf_config(context)
            result = run_tensorflow_program(train_fn, use_custom_strategy,
                                            **kwargs)
            if context.partitionId() == 0:
                return [result]
            return [None]

        return wrapped_train_fn

    @staticmethod
    def _get_conf_boolean(sc, key, default_value):
        """
        Get the conf "key" from the given spark context,
        or return the default value if the conf is not set.
        This expects the conf value to be a boolean or string;
        if the value is a string, this checks for all capitalization
        patterns of "true" and "false" to match Scala.

        Args:
            key: string for conf name
            default_value: default value for the conf value for the given key
        """
        val = sc.getConf().get(key, default_value)
        lowercase_val = val.lower()
        if lowercase_val == 'true':
            return True
        if lowercase_val == 'false':
            return False
        raise Exception("_getConfBoolean expected a boolean conf "
                        "value but found value of type {} "
                        "with value: {}".format(type(val), val))

    # Protects users that want to use encryption
    # against passing around unencrypted data
    def _check_encryption(self):
        is_ssl_enabled = MirroredStrategyRunner._get_conf_boolean(
            self.sc, 'spark.ssl.enabled', 'false')
        ignore_ssl = MirroredStrategyRunner._get_conf_boolean(
            self.sc, 'tensorflow.spark.distributor.ignoreSsl', 'false')
        if is_ssl_enabled:
            if ignore_ssl:
                self._logger.warning('''
                    This cluster has TLS encryption enabled;
                    however, {name} does not
                    support data encryption in transit. 
                    The Spark configuration 
                    'tensorflow.ignoreSsl' has been set to 
                    'true' to override this 
                    configuration and use {name} anyway. Please 
                    note this will cause model 
                    parameters and possibly training data to 
                    be sent between nodes unencrypted.
                    '''.format(name=self.__class__.__name__))
                return
            raise Exception('''
                This cluster has TLS encryption enabled; 
                however, {name} does not support 
                data encryption in transit. To override 
                this configuration and use {name} 
                anyway, you may set 'tensorflow.spark.distributor.ignoreSsl' 
                to 'true' in the Spark configuration. Please note this 
                will cause model parameters and possibly training 
                data to be sent between nodes unencrypted.
                '''.format(name=self.__class__.__name__))
