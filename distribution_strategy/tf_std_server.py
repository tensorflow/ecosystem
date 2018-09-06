# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Run a standard tensorflow server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def main(unused_argv):
  # Contrib ops are lazily loaded. So we touch one contrib module to load them
  # immediately.
  to_import_contrib_ops = tf.contrib.resampler

  # Load you custom ops here before starting the standard TensorFlow server.

  # Start and join the standard TensorFlow server.
  tf.contrib.distribute.run_standard_tensorflow_server().join()


if __name__ == "__main__":
  tf.app.run()
