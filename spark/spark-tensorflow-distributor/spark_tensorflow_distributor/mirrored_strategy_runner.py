import logging
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext


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
    MirroredStrategyRunner runs TensorFlow deep learning training jobs on Spark clusters.
    It trains synchronously by mirroring the model's variables among all the workers and
    shares the gradient updates in a decentralized manner.

    .. note:: See more at https://www.tensorflow.org/guide/distributed_training
    """
    def __init__(self, num_gpus):
        """
        :param num_gpus: Number of GPUs that participate in distributed training. When
        num_gpus < 0, training is done in local mode on the Spark driver, and otherwise
        training is distributed among the workers on the Spark cluster. For example,
        num_gpus = -4 means we train locally on 4 GPUs. If num_gpus = 16 we train
        on the Spark cluster with 16 GPUs.
        """
        self.logger = _get_logger(self.__class__.__name__)
        self.num_gpus = num_gpus
        if self.num_gpus < 0:
            self.logger.warning(
                'MirroredStrategyRunner will run on the driver node. '
                'There would be resource contention if you share the cluster with others.'
            )
            self.sc = None
            self.num_workers = None
        else:
            spark = SparkSession.builder.getOrCreate()
            self.sc = spark.sparkContext
            self.num_workers = self.sc._jsc.sc().maxNumConcurrentTasks()
        self.logger.info(f'There are {self.num_workers} workers available.')

    def run(self, main, use_custom_strategy=False, **kwargs):
        """
        :param main: Function that contains TensorFlow training code
        :param use_custom_strategy: Main function constructs its own TensorFlow strategy
        :param kwargs: keyword arguments passed to the main function at invocation time
        :return: return value of the main function from the chief training worker (partition ID 0)
        """

        num_gpus = self.num_gpus
        num_workers = self.num_workers

        # Runs the main function
        def run_program():
            if use_custom_strategy:
                return main(**kwargs)
            else:
                import tensorflow as tf
                from tensorflow.python.eager import context
                context._reset_context()
                strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
                with strategy.scope():
                    return main(**kwargs)

        # Gets the number of GPUs on the caller's machine
        def get_num_gpus():
            from tensorflow.python.client import device_lib
            return sum(1 for d in device_lib.list_local_devices() if d.device_type == 'GPU')

        # Run in local mode
        if num_gpus < 0:
            num_gpus = -num_gpus
            if num_gpus < get_num_gpus():
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in range(num_gpus))
            return run_program()

        # Spark task program
        def wrapped_main(_):
            import os, json, socket
            from pyspark import BarrierTaskContext

            # Find and hold a free port
            def find_free_port(sock):
                sock.bind(('', 0))
                return sock.getsockname()[1]

            # Sets the TF_CONFIG env var so TF servers can communicate with each other
            def set_tf_config(my_addr, context):
                my_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                my_port = find_free_port(my_sock)
                my_endpoint = "{}:{}".format(my_addr, my_port)
                worker_endpoints = context.allGather(my_endpoint)
                my_sock.close()
                cluster = {
                    'worker': worker_endpoints
                }
                tf_config = {
                    'cluster': cluster,
                    'task': {
                        'type': 'worker',
                        'index': worker_endpoints.index(my_endpoint)
                    }
                }
                os.environ['TF_CONFIG'] = json.dumps(tf_config)

            # Allocate GPUs to each task with round robin based on num_gpus
            def get_gpu_allocation(gpus_per_task):
                num_gpus_to_assign = num_gpus
                if num_gpus_to_assign > sum(gpus_per_task):
                    raise ValueError('Not enough gpus available. Try decreasing `num_gpus`.')
                gpu_allocation_per_task = [0 for e in gpus_per_task]
                while num_gpus_to_assign > 0:
                    for pid in range(len(gpus_per_task)):
                        if gpus_per_task[pid] > 0:
                            gpu_allocation_per_task[pid] += 1
                            gpus_per_task[pid] -= 1
                            num_gpus_to_assign -= 1
                        if num_gpus_to_assign == 0:
                            break
                return gpu_allocation_per_task

            # Get the number of GPUs available to each task
            def get_gpus_per_task(my_num_gpus, friend_group, my_friend_group_id):
                gpu_available = my_num_gpus // len(friend_group)
                gpu_excess = my_num_gpus - gpu_available * len(friend_group)
                owned_gpus = gpu_available
                if (my_friend_group_id < gpu_excess):
                    owned_gpus += 1
                return [int(e) for e in context.allGather(str(owned_gpus))]

            # Sets the CUDA_VISIBLE_DEVICES env var so only the appropriate GPUS are used
            def set_gpus(addrs, my_addr, context):
                my_num_gpus = get_num_gpus()
                friend_group = [partition_id for partition_id, addr in enumerate(addrs) if addr == my_addr]
                my_friend_group_id = friend_group.index(context.partitionId())
                gpus_per_task = get_gpus_per_task(my_num_gpus, friend_group, my_friend_group_id)
                if my_num_gpus == 0:
                    return
                gpu_allocation_per_task = get_gpu_allocation(gpus_per_task)
                friend_group_gpu_allocations = [gpu_allocation_per_task[pid] for pid in friend_group]
                num_gpus_allocated_before_me = sum(friend_group[:my_friend_group_id])
                gpus = [str(e) for e in range(my_num_gpus)]
                my_gpus = gpus[
                    num_gpus_allocated_before_me :
                    num_gpus_allocated_before_me + friend_group_gpu_allocations[my_friend_group_id]
                ]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(my_gpus)

            context = BarrierTaskContext.get()
            addrs = [e.address.split(':')[0] for e in context.getTaskInfos()]
            my_addr = addrs[context.partitionId()]
            set_tf_config(my_addr, context)
            set_gpus(addrs, my_addr, context)
            result = run_program()
            return [result]

        self._check_encryption()
        self.logger.info('Begin distributed training...')
        return self.sc.parallelize(range(self.num_workers), self.num_workers) \
            .barrier() \
            .mapPartitions(wrapped_main) \
            .collect()
    
    def _getConfBoolean(self, sqlContext, key, defaultValue):
        """
        Get the conf "key" from the given sqlContext,
        or return the default value if the conf is not set.
        This expects the conf value to be a boolean or string; if the value is a string,
        this checks for all capitalization patterns of "true" and "false" to match Scala.
        :param key: string for conf name
        """
        # Convert default value to str to avoid a Spark 2.3.1 + Python 3 bug: SPARK-25397
        val = sqlContext.getConf(key, str(defaultValue))
        # Convert val to str to handle unicode issues across Python 2 and 3.
        lowercase_val = str(val.lower())
        if lowercase_val == 'true':
            return True
        elif lowercase_val == 'false':
            return False
        else:
            raise Exception("_getConfBoolean expected a boolean conf value but found value of type {} "
                            "with value: {}".format(type(val), val))

    # Protects users that want to use encryption against passing around unencrypted data
    def _check_encryption(self):
        if self.num_gpus >= 0:
            sql_context = SQLContext(self.sc)
            is_ssl_enabled = self._getConfBoolean(sql_context, 'spark.ssl.enabled', 'false')
            ignore_ssl = self._getConfBoolean(sql_context, 'tensorflow.ignoreSsl', 'false')
            if is_ssl_enabled and ignore_ssl:
                self.logger.warning(
                    """
                    This cluster has TLS encryption enabled; however, {name} does not
                    support data encryption in transit. The Spark configuration 
                    'tensorflow.ignoreSsl' has been set to true to override this 
                    configuration and use {name} anyway. Please note this will cause model 
                    parameters and possibly training data to be sent between nodes unencrypted.
                    """.format(name=self.__class__.__name__)
                )
            elif is_ssl_enabled:
                raise Exception(
                    """
                    This cluster has TLS encryption enabled; however, {name} does not support 
                    data encryption in transit. To override this configuration and use {name} 
                    anyway, you may set 'tensorflow.ignoreSsl' to true in the Spark 
                    configuration. Please note this will cause model parameters and possibly training 
                    data to be sent between nodes unencrypted.
                    """.format(name=self.__class__.__name__)
                )
        return
