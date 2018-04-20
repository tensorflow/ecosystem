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
package org.tensorflow.spark.datasources.tfrecords.serde

import com.google.protobuf.ByteString
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.scalatest.{Matchers, WordSpec}
import org.tensorflow.example._
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

class TfRecordRowDecoderTest extends WordSpec with Matchers {
  val intFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(1)).build()
  val longFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(23L)).build()
  val floatFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(10.0F)).build()
  val doubleFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(14.0F)).build()
  val decimalFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(2.5F)).build()
  val longArrFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(-2L).addValue(7L).build()).build()
  val doubleArrFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(1F).addValue(2F).build()).build()
  val decimalArrFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(3F).addValue(5F).build()).build()
  val strFeature = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes)).build()).build()
  val strListFeature =Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r2".getBytes))
    .addValue(ByteString.copyFrom("r3".getBytes)).build()).build()
  val binaryFeature = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r4".getBytes))).build()
  val binaryListFeature = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r5".getBytes))
    .addValue(ByteString.copyFrom("r6".getBytes)).build()).build()
  val vectorFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(1F).addValue(2F).build()).build()

  "TensorFlow row decoder" should {

    "Decode given TensorFlow Example as Row" in {
      val schema = StructType(List(
        StructField("IntegerLabel", IntegerType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("DecimalLabel", DataTypes.createDecimalType()),
        StructField("LongArrayLabel", ArrayType(LongType)),
        StructField("DoubleArrayLabel", ArrayType(DoubleType)),
        StructField("DecimalArrayLabel", ArrayType(DataTypes.createDecimalType())),
        StructField("StrLabel", StringType),
        StructField("StrArrayLabel", ArrayType(StringType)),
        StructField("VectorLabel", VectorType),
        StructField("BinaryTypeLabel", BinaryType),
        StructField("BinaryTypeArrayLabel", ArrayType(BinaryType))
      ))

      val expectedRow = new GenericRow(
        Array[Any](1, 23L, 10.0F, 14.0, Decimal(2.5d), Seq(-2L,7L), Seq(1.0, 2.0),
          Seq(Decimal(3.0), Decimal(5.0)), "r1", Seq("r2", "r3"), Vectors.dense(Array(1.0, 2.0)),
          "r4".getBytes, Seq("r5", "r6").map(_.getBytes)
        )
      )

      //Build example
      val features = Features.newBuilder()
        .putFeature("IntegerLabel", intFeature)
        .putFeature("LongLabel", longFeature)
        .putFeature("FloatLabel", floatFeature)
        .putFeature("DoubleLabel", doubleFeature)
        .putFeature("DecimalLabel", decimalFeature)
        .putFeature("LongArrayLabel", longArrFeature)
        .putFeature("DoubleArrayLabel", doubleArrFeature)
        .putFeature("DecimalArrayLabel", decimalArrFeature)
        .putFeature("StrLabel", strFeature)
        .putFeature("StrArrayLabel", strListFeature)
        .putFeature("VectorLabel", vectorFeature)
        .putFeature("BinaryTypeLabel", binaryFeature)
        .putFeature("BinaryTypeArrayLabel", binaryListFeature)
        .build()
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()

      //Decode TensorFlow example to Sql Row
      val actualRow = DefaultTfRecordRowDecoder.decodeExample(example, schema)
      assert(actualRow ~== (expectedRow,schema))
    }

    "Decode given TensorFlow SequenceExample as Row" in {

      val schema = StructType(List(
        StructField("FloatLabel", FloatType),
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
        StructField("DecimalArrayOfArrayLabel", ArrayType(ArrayType(DataTypes.createDecimalType()))),
        StructField("StrArrayOfArrayLabel", ArrayType(ArrayType(StringType))),
        StructField("ByteArrayOfArrayLabel", ArrayType(ArrayType(BinaryType)))
      ))

      val expectedRow = new GenericRow(Array[Any](
        10.0F, Seq(Seq(-2L, 7L)), Seq(Seq(10.0F), Seq(1.0F, 2.0F)), Seq(Seq(Decimal(3.0), Decimal(5.0))), Seq(Seq("r2", "r3"), Seq("r1")),
        Seq(Seq("r5", "r6"), Seq("r4")).map(stringSeq => stringSeq.map(_.getBytes)))
      )

      //Build sequence example
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val floatFeatureList = FeatureList.newBuilder().addFeature(floatFeature).addFeature(doubleArrFeature).build()
      val decimalFeatureList = FeatureList.newBuilder().addFeature(decimalArrFeature).build()
      val stringFeatureList = FeatureList.newBuilder().addFeature(strListFeature).addFeature(strFeature).build()
      val binaryFeatureList = FeatureList.newBuilder().addFeature(binaryListFeature).addFeature(binaryFeature).build()


      val features = Features.newBuilder()
        .putFeature("FloatLabel", floatFeature)

      val featureLists = FeatureLists.newBuilder()
        .putFeatureList("LongArrayOfArrayLabel", int64FeatureList)
        .putFeatureList("FloatArrayOfArrayLabel", floatFeatureList)
        .putFeatureList("DecimalArrayOfArrayLabel", decimalFeatureList)
        .putFeatureList("StrArrayOfArrayLabel", stringFeatureList)
        .putFeatureList("ByteArrayOfArrayLabel", binaryFeatureList)
        .build()

      val seqExample = SequenceExample.newBuilder()
        .setContext(features)
        .setFeatureLists(featureLists)
        .build()

      //Decode TensorFlow example to Sql Row
      val actualRow = DefaultTfRecordRowDecoder.decodeSequenceExample(seqExample, schema)
      assert(actualRow ~== (expectedRow, schema))
    }

    "Throw an exception for unsupported data types" in {

      val features = Features.newBuilder().putFeature("MapLabel1", intFeature)
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val featureLists = FeatureLists.newBuilder().putFeatureList("MapLabel2", int64FeatureList)

      intercept[RuntimeException] {
        val example = Example.newBuilder()
          .setFeatures(features)
          .build()
        val schema = StructType(List(StructField("MapLabel1", TimestampType)))
        DefaultTfRecordRowDecoder.decodeExample(example, schema)
      }

      intercept[RuntimeException] {
        val seqExample = SequenceExample.newBuilder()
          .setContext(features)
          .setFeatureLists(featureLists)
          .build()
        val schema = StructType(List(StructField("MapLabel2", TimestampType)))
        DefaultTfRecordRowDecoder.decodeSequenceExample(seqExample, schema)
      }
    }

    "Throw an exception for non-nullable data types" in {
      val features = Features.newBuilder().putFeature("FloatLabel", floatFeature)
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val featureLists = FeatureLists.newBuilder().putFeatureList("LongArrayOfArrayLabel", int64FeatureList)

      intercept[NullPointerException] {
        val example = Example.newBuilder()
          .setFeatures(features)
          .build()
        val schema = StructType(List(StructField("MissingLabel", FloatType, nullable = false)))
        DefaultTfRecordRowDecoder.decodeExample(example, schema)
      }

      intercept[NullPointerException] {
        val seqExample = SequenceExample.newBuilder()
          .setContext(features)
          .setFeatureLists(featureLists)
          .build()
        val schema = StructType(List(StructField("MissingLabel", ArrayType(ArrayType(LongType)), nullable = false)))
        DefaultTfRecordRowDecoder.decodeSequenceExample(seqExample, schema)
      }
    }

    "Return null fields for nullable data types" in {
      val features = Features.newBuilder().putFeature("FloatLabel", floatFeature)
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val featureLists = FeatureLists.newBuilder().putFeatureList("LongArrayOfArrayLabel", int64FeatureList)

      // Decode Example
      val schema1 = StructType(List(
        StructField("FloatLabel", FloatType),
        StructField("MissingLabel", FloatType, nullable = true))
      )
      val expectedRow1 = new GenericRow(
        Array[Any](10.0F, null)
      )
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()

      assert(DefaultTfRecordRowDecoder.decodeExample(example, schema1) ~== (expectedRow1, schema1))

      // Decode SequenceExample
      val schema2 = StructType(List(
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("MissingLabel", ArrayType(ArrayType(DoubleType)), nullable = true))
      )
      val expectedRow2 = new GenericRow(
        Array[Any](Seq(Seq(-2L, 7L)), null)
      )
      val seqExample = SequenceExample.newBuilder()
        .setContext(features)
        .setFeatureLists(featureLists)
        .build()

      assert(DefaultTfRecordRowDecoder.decodeSequenceExample(seqExample, schema2) ~== (expectedRow2, schema2))
    }
  }
}
