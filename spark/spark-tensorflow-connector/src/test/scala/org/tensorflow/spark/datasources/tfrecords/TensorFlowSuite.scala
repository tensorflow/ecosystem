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

import org.apache.hadoop.fs.Path
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SaveMode}

import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

class TensorFlowSuite extends SharedSparkSessionSuite {

  val exampleSchema = StructType(List(
    StructField("id", IntegerType),
    StructField("IntegerLabel", IntegerType),
    StructField("LongLabel", LongType),
    StructField("FloatLabel", FloatType),
    StructField("DoubleLabel", DoubleType),
    StructField("DoubleArrayLabel", ArrayType(DoubleType, true)),
    StructField("StrLabel", StringType),
    StructField("BinaryLabel", BinaryType)))

  val exampleTestRows: Array[Row] = Array(
    new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1",
      Array[Byte](0xff.toByte, 0xf0.toByte))),
    new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2",
      Array[Byte](0xff.toByte, 0xf1.toByte))))


  val sequenceExampleTestRows: Array[Row] = Array(
    new GenericRow(Array[Any](23L, Seq(Seq(2.0F, 4.5F)), Seq(Seq("r1", "r2")))),
    new GenericRow(Array[Any](24L, Seq(Seq(-1.0F, 0F)), Seq(Seq("r3")))))

  val sequenceExampleSchema = StructType(List(
    StructField("id",LongType),
    StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
    StructField("StrArrayOfArrayLabel", ArrayType(ArrayType(StringType)))
  ))

  private def createDataFrameForExampleTFRecord() : DataFrame = {
    val rdd = spark.sparkContext.parallelize(exampleTestRows)
    spark.createDataFrame(rdd, exampleSchema)
  }

  private def createDataFrameForSequenceExampleTFRecords() : DataFrame = {
    val rdd = spark.sparkContext.parallelize(sequenceExampleTestRows)
    spark.createDataFrame(rdd, sequenceExampleSchema)
  }


  "Spark TensorFlow module" should {

    "Test Import/Export of Example records" in {
      val path = s"$TF_SANDBOX_DIR/example.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecords").option("recordType", "Example").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").schema(exampleSchema).load(path)
      val actualDf = importedDf.select("id", "IntegerLabel", "LongLabel", "FloatLabel",
        "DoubleLabel", "DoubleArrayLabel", "StrLabel", "BinaryLabel").sort("StrLabel")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      expectedRows.zip(actualRows).foreach { case (expected: Row, actual: Row) =>
        assert(expected ~== actual, exampleSchema)
      }
    }

    "Test Import/Export of SequenceExample records" in {

      val path = s"$TF_SANDBOX_DIR/sequenceExample.tfrecord"

      val df: DataFrame = createDataFrameForSequenceExampleTFRecords()
      df.write.format("tfrecords").option("recordType", "SequenceExample").save(path)

      val importedDf: DataFrame = spark.read.format("tfrecords").option("recordType", "SequenceExample").schema(sequenceExampleSchema).load(path)
      val actualDf = importedDf.select("id", "FloatArrayOfArrayLabel", "StrArrayOfArrayLabel").sort("id")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      assert(expectedRows === actualRows)
    }

    "Test overwrite mode during export of Example records" in {

      val path = s"$TF_SANDBOX_DIR/example_overwrite.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecords").option("recordType", "Example").save(path)

      df.write.format("tfrecords").mode(SaveMode.Overwrite).option("recordType", "Example").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").schema(exampleSchema).load(path)
      val actualDf = importedDf.select("id", "IntegerLabel", "LongLabel", "FloatLabel",
        "DoubleLabel", "DoubleArrayLabel", "StrLabel", "BinaryLabel").sort("StrLabel")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      expectedRows.zip(actualRows).foreach { case (expected: Row, actual: Row) =>
        assert(expected ~== actual, exampleSchema)
      }

    }

    "Test append mode during export of Example records" in {

      val path = s"$TF_SANDBOX_DIR/example_append.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecords").option("recordType", "Example").save(path)

      intercept [IllegalArgumentException] {
        df.write.format("tfrecords").mode(SaveMode.Append).option("recordType", "Example").save(path)
      }
    }

    "Test errorIfExists mode during export of Example records" in {

      val path = s"$TF_SANDBOX_DIR/example_errorIfExists.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecords").mode(SaveMode.ErrorIfExists).option("recordType", "Example").save(path)

      intercept [IllegalStateException] {
        df.write.format("tfrecords").mode(SaveMode.ErrorIfExists).option("recordType", "Example").save(path)
      }
    }

    "Test ignore mode during export of Example records" in {

      val path = s"$TF_SANDBOX_DIR/example_ignore.tfrecord"

      val hadoopConf = spark.sparkContext.hadoopConfiguration
      val outputPath = new Path(path)
      val fs = outputPath.getFileSystem(hadoopConf)
      val qualifiedOutputPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecords").mode(SaveMode.Ignore).option("recordType", "Example").save(path)

      assert(fs.exists(qualifiedOutputPath))
      val timestamp1 = fs.getFileStatus(qualifiedOutputPath).getModificationTime

      df.write.format("tfrecords").mode(SaveMode.Ignore).option("recordType", "Example").save(path)

      val timestamp2 = fs.getFileStatus(qualifiedOutputPath).getModificationTime

      assert(timestamp1 == timestamp2, "SaveMode.Ignore Error: File was overwritten. Timestamps do not match")
    }

  }
}
