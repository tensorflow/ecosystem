/**
 *  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.tensorflow.spark.datasources.tfrecords

import java.io._
import java.nio.file.Paths

import org.apache.spark.sql.{DataFrame, Row}
import org.tensorflow.hadoop.util._
import org.tensorflow.spark.datasources.tfrecords.serde.DefaultTfRecordRowEncoder

/**
 * Utilities to distribute dataframe-based content on a cluster.
 */
object Propagate {

  /**
   * Writes the content of a dataframe on the Spark workers, in a partitioned manner, using the
   * TFRecord format.
   *
   * After calling this function, each of the workers stores on the local disk a subset of the data.
   * Which subset that it stored on each worker is determined by the partitioning of the Dataframe.
   * Each of the partitions is coalesced into a single TFRecord file and written on the node where
   * the partition lives.
   *
   * This is useful in the context of distributed training, in which each of the workers gets a
   * subset of the data to work on.
   *
   * @param df a dataframe. All the content of that dataframe must be writable into TFRecords.
   * @param localPath a base path that is created on each of the worker nodes, and that will be
   *                  populated with data from the dataframe. For example, if path is /path, then
   *                  each of the worker nodes will have the following:
   *                   - worker1: /path/part-0001.tfrecord, /path/part-0002.tfrecord, ...
   *                   - worker2: /path/part-0042.tfrecord, /path/part-0043.tfrecord, ...
   *
   * If localPath exists already, an error is returned.
   */
  def writePartitionsLocal(df: DataFrame, localPath: String): Unit = {

    val cleanedPath = Paths.get(localPath).toAbsolutePath.toString

    // Writes the partition to a file and returns the number of records output.
    def mapFun(index: Int, part: Iterator[Row]): Iterator[Int] = {
      val dir = new File(cleanedPath)
      // Make the directory if it does not exist
      dir.mkdirs()
      // The path to the partition file.
      val filePath = cleanedPath + s"/part-" + String.format("%05d", new java.lang.Integer(index))
      val fos = new DataOutputStream(new FileOutputStream(filePath))
      var count = 0
      try {
        val tfw = new TFRecordWriter(fos)
        for(row <- part) {
          val example = DefaultTfRecordRowEncoder.encodeExample(row)
          val b = example.toByteArray
          tfw.write(b)
          count += 1
        }
      } finally {
        fos.close()
      }
      Iterator(count)
    }

    // Perform the action.
    df.rdd.mapPartitionsWithIndex(mapFun).collect()
  }
}
