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

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

class TensorFlowSuite extends SharedSparkSessionSuite {

  "Spark TensorFlow module" should {

    "Test Import/Export of Example records" in {

      val path = s"$TF_SANDBOX_DIR/example.tfrecord"
      val testRows: Array[Row] = Array(
        new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1")),
        new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2")))

      val schema = StructType(List(
        StructField("id", IntegerType),
        StructField("IntegerLabel", IntegerType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("DoubleArrayLabel", ArrayType(DoubleType, true)),
        StructField("StrLabel", StringType)))

      val rdd = spark.sparkContext.parallelize(testRows)

      val df: DataFrame = spark.createDataFrame(rdd, schema)
      df.write.format("tfrecords").option("recordType", "Example").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").schema(schema).load(path)
      val actualDf = importedDf.select("id", "IntegerLabel", "LongLabel", "FloatLabel", "DoubleLabel", "DoubleArrayLabel", "StrLabel").sort("StrLabel")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      assert(expectedRows === actualRows)
    }

    "Test Import/Export of SequenceExample records" in {

      val path = s"$TF_SANDBOX_DIR/sequenceExample.tfrecord"
      val testRows: Array[Row] = Array(
        new GenericRow(Array[Any](23L, Seq(Seq(2.0F, 4.5F)), Seq(Seq("r1", "r2")))),
        new GenericRow(Array[Any](24L, Seq(Seq(-1.0F, 0F)), Seq(Seq("r3")))))

      val schema = StructType(List(
        StructField("id",LongType),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
        StructField("StrArrayOfArrayLabel", ArrayType(ArrayType(StringType)))
      ))

      val rdd = spark.sparkContext.parallelize(testRows)

      val df: DataFrame = spark.createDataFrame(rdd, schema)
      df.write.format("tfrecords").option("recordType", "SequenceExample").save(path)

      val importedDf: DataFrame = spark.read.format("tfrecords").option("recordType", "SequenceExample").schema(schema).load(path)
      val actualDf = importedDf.select("id", "FloatArrayOfArrayLabel", "StrArrayOfArrayLabel").sort("id")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      assert(expectedRows === actualRows)
    }
  }
}
