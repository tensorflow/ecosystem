# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run a tf.data service server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.compat.v1.app.flags

flags.DEFINE_integer("port", 0, "Port to listen on")
flags.DEFINE_bool("is_dispatcher", False, "Whether to start a dispatcher (as opposed to a worker server")
flags.DEFINE_string("dispatcher_address", "", "The address of the dispatcher. This is only needed when starting a worker server.")
flags.DEFINE_string("worker_address", "", "The address of the worker server. This is only needed when starting a worker server.")

FLAGS = flags.FLAGS


def main(unused_argv):
  if FLAGS.is_dispatcher:
    print("Starting tf.data service dispatcher")
    server = tf.data.experimental.service.DispatchServer(
        tf.data.experimental.service.DispatcherConfig(
            port=FLAGS.port,
            protocol="grpc"))
  else:
    print("Starting tf.data service worker")
    server = tf.data.experimental.service.WorkerServer(
        tf.data.experimental.service.WorkerConfig(
            port=FLAGS.port,
            protocol="grpc",
            dispatcher_address=FLAGS.dispatcher_address,
            worker_address=FLAGS.worker_address))
  server.join()


if __name__ == "__main__":
  tf.compat.v1.app.run()
