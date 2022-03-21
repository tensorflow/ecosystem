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
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;

/**
 * Constants used in both Client and ApplicationMaster
 */
@InterfaceAudience.Public
@InterfaceStability.Unstable
public class TFConstants {

  /**
   * Environment key name pointing to the shell script's location
   */
  public static final String SCRIPTLOCATION = "SCRIPTLOCATION";

  /**
   * Environment key name denoting the file timestamp for the shell script. 
   * Used to validate the local resource. 
   */
  public static final String SCRIPTTIMESTAMP = "SCRIPTTIMESTAMP";

  /**
   * Environment key name denoting the file content length for the shell script. 
   * Used to validate the local resource. 
   */
  public static final String SHELLSCRIPTLEN = "SHELLSCRIPTLEN";

  /**
   * Environment key name denoting the timeline domain ID.
   */
  public static final String TIMELINEDOMAIN = "TIMELINEDOMAIN";

  /**
   * Default application name
   */
  public static final String DEFAULT_APPNAME = "TensorFlow";
}
