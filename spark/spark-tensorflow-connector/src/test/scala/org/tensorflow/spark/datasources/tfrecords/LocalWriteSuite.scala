/**
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.spark.datasources.tfrecords

import java.nio.file.Files
import java.nio.file.Paths

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._

import org.apache.commons.io.FileUtils

class LocalWriteSuite extends SharedSparkSessionSuite {

  val testRows: Array[Row] = Array(
    new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 3.0), "r1")),
    new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 3.0), "r2")),
    new GenericRow(Array[Any](31, 3, 25L, 14.0F, 16.0, List(3.0, 3.0), "r3")))
  val schema = StructType(List(StructField("id", IntegerType),
    StructField("IntegerTypeLabel", IntegerType),
    StructField("LongTypeLabel", LongType),
    StructField("FloatTypeLabel", FloatType),
    StructField("DoubleTypeLabel", DoubleType),
    StructField("VectorLabel", ArrayType(DoubleType, true)),
    StructField("name", StringType)))


  "Propagate" should {
    "write data locally" in {
      // Create a dataframe with 2 partitions
      val rdd = spark.sparkContext.parallelize(testRows, numSlices = 2)
      val df = spark.createDataFrame(rdd, schema)

      // Write the partitions onto the local hard drive. Since it is going to be the
      // local file system, the partitions will be written in the same directory of the
      // same machine.
      // In a distributed setting though, two different machines would each hold a single
      // partition.
      val localPath = Files.createTempDirectory("spark-connector-propagate").toAbsolutePath.toString
      val savePath = localPath + "/testResult"
      df.write.format("tfrecords")
        .option("recordType", "Example")
        .option("writeLocality", "local")
        .save(savePath)

      // Read again this directory, this time using the Hadoop file readers, it should
      // return the same data.
      // This only works in this test and does not hold in general, because the partitions
      // will be written on the workers. Everything runs locally for tests.
      val df2 = spark.read.format("tfrecords").option("recordType", "Example")
        .load(savePath).sort("id").select("id", "IntegerTypeLabel", "LongTypeLabel",
        "FloatTypeLabel", "DoubleTypeLabel", "VectorLabel", "name") // Correct column order.

      assert(df2.collect().toSeq === testRows.toSeq)
    }
  }
}