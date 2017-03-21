/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.hadoop.yarn;

/**
 * TensorFlow launcher for YARN
 *
 * Modified from YARN sample application: DistributedShell.
 */

public class TFContainerRequest {
    private int virtualCores;
    private int memory;
    private int priority;

    private TFContainerRequest() {}

    public TFContainerRequest(int virtualCores, int memory, int priority)
    {
        this.virtualCores = virtualCores;
        this.memory = memory;
        this.priority = priority;
    }

    public TFContainerRequest(TFContainerRequest that) {
        this.virtualCores = that.virtualCores;
        this.memory = that.memory;
        this.priority = that.priority;
    }

    public int getVirtualCores() {
        return virtualCores;
    }

    public int getMemory() {
        return memory;
    }

    public int getPriority() {
        return priority;
    }
}
