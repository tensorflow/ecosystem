/**
 * Copyright (c) 2016 Intel Corporation 
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
      df.write.format("tensorflow").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tensorflow").schema(schema).load(path)
      val actualDf = importedDf.select("id", "IntegerTypelabel", "LongTypelabel", "FloatTypelabel", "DoubleTypelabel", "vectorlabel", "name").sort("name")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      expectedRows should equal(actualRows)
    }

    "Encode given Row as TensorFlow example" in {
      val schemaStructType = StructType(Array(
        StructField("IntegerTypelabel", IntegerType),
        StructField("LongTypelabel", LongType),
        StructField("FloatTypelabel", FloatType),
        StructField("DoubleTypelabel", DoubleType),
        StructField("vectorlabel", ArrayType(DoubleType, true)),
        StructField("strlabel", StringType)
      ))
      val doubleArray = Array(1.1, 111.1, 11111.1)
      val expectedFloatArray = Array(1.1F, 111.1F, 11111.1F)

      val rowWithSchema = new GenericRowWithSchema(Array[Any](1, 23L, 10.0F, 14.0, doubleArray, "r1"), schemaStructType)

      //Encode Sql Row to TensorFlow example
      val example = DefaultTfRecordRowEncoder.encodeTfRecord(rowWithSchema)
      import org.tensorflow.example.Feature

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = example.getFeatures.getFeatureMap.asScala
      assert(featureMap("IntegerTypelabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerTypelabel").getInt64List.getValue(0).toInt == 1)

      assert(featureMap("LongTypelabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("LongTypelabel").getInt64List.getValue(0).toInt == 23)

      assert(featureMap("FloatTypelabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatTypelabel").getFloatList.getValue(0) == 10.0F)

      assert(featureMap("DoubleTypelabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DoubleTypelabel").getFloatList.getValue(0) == 14.0F)

      assert(featureMap("vectorlabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("vectorlabel").getFloatList.getValueList.toArray === expectedFloatArray)

      assert(featureMap("strlabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("strlabel").getBytesList.toByteString.toStringUtf8.trim == "r1")

    }

    "Throw an exception for a vector with null values during Encode" in {
      intercept[Exception] {
        val schemaStructType = StructType(Array(
          StructField("vectorlabel", ArrayType(DoubleType, true))
        ))
        val doubleArray = Array(1.1, null, 111.1, null, 11111.1)

        val rowWithSchema = new GenericRowWithSchema(Array[Any](doubleArray), schemaStructType)

        //Throws NullPointerException
        DefaultTfRecordRowEncoder.encodeTfRecord(rowWithSchema)
      }
    }

    "Decode given TensorFlow Example as Row" in {

      //Here Vector with null's are not supported
      val expectedRow = new GenericRow(Array[Any](1, 23L, 10.0F, 14.0, Seq(1.0, 2.0), "r1"))

      val schema = StructType(List(
        StructField("IntegerTypelabel", IntegerType),
        StructField("LongTypelabel", LongType),
        StructField("FloatTypelabel", FloatType),
        StructField("DoubleTypelabel", DoubleType),
        StructField("vectorlabel", ArrayType(DoubleType)),
        StructField("strlabel", StringType)))

      //Build example
      val intFeature = Int64List.newBuilder().addValue(1)
      val longFeature = Int64List.newBuilder().addValue(23L)
      val floatFeature = FloatList.newBuilder().addValue(10.0F)
      val doubleFeature = FloatList.newBuilder().addValue(14.0F)
      val vectorFeature = FloatList.newBuilder().addValue(1F).addValue(2F).build()
      val strFeature = BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes)).build()
      val features = Features.newBuilder()
        .putFeature("IntegerTypelabel", Feature.newBuilder().setInt64List(intFeature).build())
        .putFeature("LongTypelabel", Feature.newBuilder().setInt64List(longFeature).build())
        .putFeature("FloatTypelabel", Feature.newBuilder().setFloatList(floatFeature).build())
        .putFeature("DoubleTypelabel", Feature.newBuilder().setFloatList(doubleFeature).build())
        .putFeature("vectorlabel", Feature.newBuilder().setFloatList(vectorFeature).build())
        .putFeature("strlabel", Feature.newBuilder().setBytesList(strFeature).build())
        .build()
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()

      //Decode TensorFlow example to Sql Row
      val actualRow = DefaultTfRecordRowDecoder.decodeTfRecord(example, schema)
      actualRow should equal(expectedRow)
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

