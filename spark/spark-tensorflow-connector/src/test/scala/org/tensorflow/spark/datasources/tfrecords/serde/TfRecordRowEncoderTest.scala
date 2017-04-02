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

import org.tensorflow.example.Feature
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.scalatest.{Matchers, WordSpec}
import scala.collection.JavaConverters._
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

class TfRecordRowEncoderTest extends WordSpec with Matchers {

  "TensorFlow row encoder" should {

    "Encode given Row as TensorFlow Example" in {
      val schemaStructType = StructType(Array(
        StructField("IntegerTypeLabel", IntegerType),
        StructField("LongTypeLabel", LongType),
        StructField("FloatTypeLabel", FloatType),
        StructField("DoubleTypeLabel", DoubleType),
        StructField("vectorLabel", ArrayType(DoubleType)),
        StructField("strLabel", StringType),
        StructField("strListLabel", ArrayType(StringType))
      ))
      val doubleArray = Array(1.1, 111.1, 11111.1)
      val expectedFloatArray = Array(1.1F, 111.1F, 11111.1F)

      val rowWithSchema = new GenericRowWithSchema(Array[Any](1, 23L, 10.0F, 14.0, doubleArray, "r1", Seq("r2", "r3")), schemaStructType)

      //Encode Sql Row to TensorFlow example
      val example = DefaultTfRecordRowEncoder.encodeExample(rowWithSchema)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = example.getFeatures.getFeatureMap.asScala
      assert(featureMap.size == 7)

      assert(featureMap("IntegerTypeLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerTypeLabel").getInt64List.getValue(0).toInt == 1)

      assert(featureMap("LongTypeLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("LongTypeLabel").getInt64List.getValue(0).toInt == 23)

      assert(featureMap("FloatTypeLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatTypeLabel").getFloatList.getValue(0) == 10.0F)

      assert(featureMap("DoubleTypeLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DoubleTypeLabel").getFloatList.getValue(0) == 14.0F)

      assert(featureMap("vectorLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("vectorLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== expectedFloatArray)

      assert(featureMap("strLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("strLabel").getBytesList.getValue(0).toStringUtf8 == "r1")

      assert(featureMap("strListLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("strListLabel").getBytesList.getValueList.asScala.map(_.toStringUtf8) === Seq("r2", "r3"))
    }

    "Encode given Row as TensorFlow SequenceExample" in {

      val schemaStructType = StructType(Array(
        StructField("IntegerTypeLabel", IntegerType),
        StructField("LongListOfListsLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatListOfListsLabel", ArrayType(ArrayType(FloatType))),
        StructField("StringListOfListsLabel", ArrayType(ArrayType(StringType)))
      ))

      val longListOfLists = Seq(Seq(3L, 5L), Seq(-8L, 0L))
      val floatListOfLists = Seq(Seq(1.5F, -6.5F), Seq(-8.2F, 0F))
      val stringListOfLists = Seq(Seq("r1"), Seq("r2", "r3"), Seq("r4"))

      val rowWithSchema = new GenericRowWithSchema(Array[Any](10, longListOfLists, floatListOfLists, stringListOfLists), schemaStructType)

      //Encode Sql Row to TensorFlow example
      val seqExample = DefaultTfRecordRowEncoder.encodeSequenceExample(rowWithSchema)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = seqExample.getContext.getFeatureMap.asScala
      val featureListMap = seqExample.getFeatureLists.getFeatureListMap.asScala

      assert(featureMap.size == 1)
      assert(featureMap("IntegerTypeLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerTypeLabel").getInt64List.getValue(0).toInt == 10)

      assert(featureListMap.size == 3)
      assert(featureListMap("LongListOfListsLabel").getFeatureList.asScala.map(_.getInt64List.getValueList.asScala.toSeq) === longListOfLists)
      assert(featureListMap("FloatListOfListsLabel").getFeatureList.asScala.map(_.getFloatList.getValueList.asScala.map(_.toFloat).toSeq) ~== floatListOfLists)
      assert(featureListMap("StringListOfListsLabel").getFeatureList.asScala.map(_.getBytesList.getValueList.asScala.map(_.toStringUtf8).toSeq) === stringListOfLists)
    }
  }
}
