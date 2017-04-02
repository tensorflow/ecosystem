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

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._
import org.scalatest.{Matchers, WordSpec}
import org.tensorflow.example._
import org.tensorflow.hadoop.shaded.protobuf.ByteString
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

class TfRecordRowDecoderTest extends WordSpec with Matchers {

  "TensorFlow row decoder" should {

    "Decode given TensorFlow Example as Row" in {

      val schema = StructType(List(
        StructField("IntegerLabel", IntegerType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("LongArrayLabel", ArrayType(LongType)),
        StructField("DoubleArrayLabel", ArrayType(DoubleType)),
        StructField("StrLabel", StringType),
        StructField("StrArrayLabel", ArrayType(StringType))
      ))

      val expectedRow = new GenericRow(
        Array[Any](1, 23L, 10.0F, 14.0, Seq(-2L,7L), Seq(1.0, 2.0), "r1", Seq("r2", "r3"))
      )

      //Build example
      val intFeature = Int64List.newBuilder().addValue(1)
      val longFeature = Int64List.newBuilder().addValue(23L)
      val floatFeature = FloatList.newBuilder().addValue(10.0F)
      val doubleFeature = FloatList.newBuilder().addValue(14.0F)
      val longArrFeature = Int64List.newBuilder().addValue(-2L).addValue(7L).build()
      val doubleArrFeature = FloatList.newBuilder().addValue(1F).addValue(2F).build()
      val strFeature = BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes)).build()
      val strListFeature = BytesList.newBuilder().addValue(ByteString.copyFrom("r2".getBytes))
        .addValue(ByteString.copyFrom("r3".getBytes)).build()
      val features = Features.newBuilder()
        .putFeature("IntegerLabel", Feature.newBuilder().setInt64List(intFeature).build())
        .putFeature("LongLabel", Feature.newBuilder().setInt64List(longFeature).build())
        .putFeature("FloatLabel", Feature.newBuilder().setFloatList(floatFeature).build())
        .putFeature("DoubleLabel", Feature.newBuilder().setFloatList(doubleFeature).build())
        .putFeature("LongArrayLabel", Feature.newBuilder().setInt64List(longArrFeature).build())
        .putFeature("DoubleArrayLabel", Feature.newBuilder().setFloatList(doubleArrFeature).build())
        .putFeature("StrLabel", Feature.newBuilder().setBytesList(strFeature).build())
        .putFeature("StrArrayLabel", Feature.newBuilder().setBytesList(strListFeature).build())
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
        StructField("LongArrayLabel", ArrayType(LongType)),
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
        StructField("StrArrayOfArrayLabel", ArrayType(ArrayType(StringType)))
      ))

      val expectedRow = new GenericRow(Array[Any](
        Seq(-2L,7L), Seq(Seq(4L, 10L)), Seq(Seq(2.25F), Seq(-1.9F,3.5F)), Seq(Seq("r1", "r2"), Seq("r3")))
      )

      //Build sequence example
      val longArrFeature = Int64List.newBuilder().addValue(-2L).addValue(7L).build()

      val int64List1 = Int64List.newBuilder().addValue(4L).addValue(10L).build()
      val intFeature1 = Feature.newBuilder().setInt64List(int64List1).build()
      val int64FeatureList = FeatureList.newBuilder().addFeature(intFeature1).build()

      val floatList1 = FloatList.newBuilder().addValue(2.25F).build()
      val floatList2 = FloatList.newBuilder().addValue(-1.9F).addValue(3.5F).build()
      val floatFeature1 = Feature.newBuilder().setFloatList(floatList1).build()
      val floatFeature2 = Feature.newBuilder().setFloatList(floatList2).build()
      val floatFeatureList = FeatureList.newBuilder().addFeature(floatFeature1).addFeature(floatFeature2).build()

      val bytesList1 = BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes))
        .addValue(ByteString.copyFrom("r2".getBytes)).build()
      val bytesList2 = BytesList.newBuilder().addValue(ByteString.copyFrom("r3".getBytes)).build()
      val bytesFeature1 = Feature.newBuilder().setBytesList(bytesList1).build()
      val bytesFeature2 = Feature.newBuilder().setBytesList(bytesList2).build()
      val bytesFeatureList = FeatureList.newBuilder().addFeature(bytesFeature1).addFeature(bytesFeature2).build()

      val features = Features.newBuilder()
        .putFeature("LongArrayLabel", Feature.newBuilder().setInt64List(longArrFeature).build())

      val featureLists = FeatureLists.newBuilder()
        .putFeatureList("LongArrayOfArrayLabel", int64FeatureList)
        .putFeatureList("FloatArrayOfArrayLabel", floatFeatureList)
        .putFeatureList("StrArrayOfArrayLabel", bytesFeatureList)
        .build()

      val seqExample = SequenceExample.newBuilder()
        .setContext(features)
        .setFeatureLists(featureLists)
        .build()

      //Decode TensorFlow example to Sql Row
      val actualRow = DefaultTfRecordRowDecoder.decodeSequenceExample(seqExample, schema)
      assert(actualRow ~== (expectedRow, schema))
    }
  }
}
