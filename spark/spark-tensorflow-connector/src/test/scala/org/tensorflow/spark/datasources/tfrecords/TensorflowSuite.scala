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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{GenericRow, GenericRowWithSchema}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.tensorflow.example._
import org.tensorflow.hadoop.shaded.protobuf.ByteString
import org.tensorflow.spark.datasources.tfrecords.serde.{DefaultTfRecordRowDecoder, DefaultTfRecordRowEncoder}

import scala.collection.JavaConverters._

class TensorflowSuite extends SharedSparkSessionSuite {

  "Spark TensorFlow module" should {

    "Test Import/Export" in {

      val path = s"$TF_SANDBOX_DIR/output25.tfr"
      val testRows: Array[Row] = Array(
        new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1")),
        new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2")))

      val schema = StructType(List(
        StructField("id", IntegerType),
        StructField("IntegerTypelabel", IntegerType),
        StructField("LongTypelabel", LongType),
        StructField("FloatTypelabel", FloatType),
        StructField("DoubleTypelabel", DoubleType),
        StructField("vectorlabel", ArrayType(DoubleType, true)),
        StructField("name", StringType)))

      val rdd = spark.sparkContext.parallelize(testRows)

      val df: DataFrame = spark.createDataFrame(rdd, schema)
      df.write.format("tfrecords").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tfrecords").schema(schema).load(path)
      val actualDf = importedDf.select("id", "IntegerTypelabel", "LongTypelabel", "FloatTypelabel", "DoubleTypelabel", "vectorlabel", "name").sort("name")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      expectedRows should equal(actualRows)
    }

    "Check infer schema" in {

      //Build example1
      val intFeature1 = Int64List.newBuilder().addValue(1)
      val longFeature1 = Int64List.newBuilder().addValue(Int.MaxValue + 10L)
      val floatFeature1 = FloatList.newBuilder().addValue(10.0F)
      val doubleFeature1 = FloatList.newBuilder().addValue(14.0F)
      val vectorFeature1 = FloatList.newBuilder().addValue(1F).build()
      val strFeature1 = BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes)).build()
      val features1 = Features.newBuilder()
        .putFeature("IntegerTypelabel", Feature.newBuilder().setInt64List(intFeature1).build())
        .putFeature("LongTypelabel", Feature.newBuilder().setInt64List(longFeature1).build())
        .putFeature("FloatTypelabel", Feature.newBuilder().setFloatList(floatFeature1).build())
        .putFeature("DoubleTypelabel", Feature.newBuilder().setFloatList(doubleFeature1).build())
        .putFeature("vectorlabel", Feature.newBuilder().setFloatList(vectorFeature1).build())
        .putFeature("strlabel", Feature.newBuilder().setBytesList(strFeature1).build())
        .build()
      val example1 = Example.newBuilder()
        .setFeatures(features1)
        .build()

      //Build example2
      val intFeature2 = Int64List.newBuilder().addValue(2)
      val longFeature2 = Int64List.newBuilder().addValue(24)
      val floatFeature2 = FloatList.newBuilder().addValue(12.0F)
      val doubleFeature2 = FloatList.newBuilder().addValue(Float.MaxValue + 15)
      val vectorFeature2 = FloatList.newBuilder().addValue(2F).addValue(2F).build()
      val strFeature2 = BytesList.newBuilder().addValue(ByteString.copyFrom("r2".getBytes)).build()
      val features2 = Features.newBuilder()
        .putFeature("IntegerTypelabel", Feature.newBuilder().setInt64List(intFeature2).build())
        .putFeature("LongTypelabel", Feature.newBuilder().setInt64List(longFeature2).build())
        .putFeature("FloatTypelabel", Feature.newBuilder().setFloatList(floatFeature2).build())
        .putFeature("DoubleTypelabel", Feature.newBuilder().setFloatList(doubleFeature2).build())
        .putFeature("vectorlabel", Feature.newBuilder().setFloatList(vectorFeature2).build())
        .putFeature("strlabel", Feature.newBuilder().setBytesList(strFeature2).build())
        .build()
      val example2 = Example.newBuilder()
        .setFeatures(features2)
        .build()

      val exampleRDD: RDD[Example] = spark.sparkContext.parallelize(List(example1, example2))

      val actualSchema = TensorflowInferSchema(exampleRDD)

      //Verify each TensorFlow Datatype is inferred as one of our Datatype
      actualSchema.fields.map { colum =>
        colum.name match {
          case "IntegerTypelabel" => colum.dataType.equals(IntegerType)
          case "LongTypelabel" => colum.dataType.equals(LongType)
          case "FloatTypelabel" | "DoubleTypelabel" | "vectorlabel" => colum.dataType.equals(FloatType)
          case "strlabel" => colum.dataType.equals(StringType)
        }
      }
    }
  }
}

