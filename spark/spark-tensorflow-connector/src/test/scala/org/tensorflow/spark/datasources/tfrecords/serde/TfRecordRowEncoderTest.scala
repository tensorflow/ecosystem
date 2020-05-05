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

import org.apache.spark.ml.linalg.Vectors
import org.tensorflow.example._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.scalatest.{Matchers, WordSpec}
import scala.collection.JavaConverters._
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

class TfRecordRowEncoderTest extends WordSpec with Matchers {

  "TensorFlow row encoder" should {

    "Encode given Row as TensorFlow Example" in {
      val schemaStructType = StructType(Array(
        StructField("IntegerLabel", IntegerType),
        StructField("BooleanLabel", BooleanType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("DecimalLabel", DataTypes.createDecimalType()),
        StructField("DoubleArrayLabel", ArrayType(DoubleType)),
        StructField("DecimalArrayLabel", ArrayType(DataTypes.createDecimalType())),
        StructField("StrLabel", StringType),
        StructField("StrArrayLabel", ArrayType(StringType)),
        StructField("DenseVectorLabel", VectorType),
        StructField("SparseVectorLabel", VectorType),
        StructField("BinaryLabel", BinaryType),
        StructField("BinaryArrayLabel", ArrayType(BinaryType)),
        StructField("BooleanArrayLabel", ArrayType(BooleanType))
      ))
      val doubleArray = Array(1.1, 111.1, 11111.1)
      val decimalArray = Array(Decimal(4.0), Decimal(8.0))
      val sparseVector = Vectors.sparse(3, Seq((1, 2.0), (2, 1.5)))
      val denseVector = Vectors.dense(Array(5.6, 7.0))
      val byteArray = Array[Byte](0xde.toByte, 0xad.toByte, 0xbe.toByte, 0xef.toByte)
      val byteArray1 = Array[Byte](-128, 23, 127)
      val booleanArray = Array(false, true)

      val row = Array[Any](1, true, 23L, 10.0F, 14.0, Decimal(6.5), doubleArray, decimalArray,
        "r1", Seq("r2", "r3"), denseVector, sparseVector, byteArray, Seq(byteArray, byteArray1), booleanArray)
      val rowWithSchema = new GenericRowWithSchema(row, schemaStructType)

      //Encode Sql Row to TensorFlow example
      val example = DefaultTfRecordRowEncoder.encodeExample(rowWithSchema)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = example.getFeatures.getFeatureMap.asScala
      assert(featureMap.size == row.length)

      assert(featureMap("IntegerLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerLabel").getInt64List.getValue(0).toInt == 1)

      assert(featureMap("BooleanLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("BooleanLabel").getInt64List.getValue(0).toInt == 1)

      assert(featureMap("LongLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("LongLabel").getInt64List.getValue(0).toInt == 23)

      assert(featureMap("FloatLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatLabel").getFloatList.getValue(0) == 10.0F)

      assert(featureMap("DoubleLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DoubleLabel").getFloatList.getValue(0) == 14.0F)

      assert(featureMap("DecimalLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DecimalLabel").getFloatList.getValue(0) == 6.5F)

      assert(featureMap("DoubleArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DoubleArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== doubleArray.map(_.toFloat))

      assert(featureMap("DecimalArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DecimalArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== decimalArray.map(_.toFloat))

      assert(featureMap("StrLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("StrLabel").getBytesList.getValue(0).toStringUtf8 == "r1")

      assert(featureMap("StrArrayLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("StrArrayLabel").getBytesList.getValueList.asScala.map(_.toStringUtf8) === Seq("r2", "r3"))

      assert(featureMap("DenseVectorLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DenseVectorLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== denseVector.toArray.map(_.toFloat))

      assert(featureMap("SparseVectorLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("SparseVectorLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== sparseVector.toDense.toArray.map(_.toFloat))

      assert(featureMap("BinaryLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("BinaryLabel").getBytesList.getValue(0).toByteArray.deep == byteArray.deep)

      assert(featureMap("BinaryArrayLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      val binaryArrayValue = featureMap("BinaryArrayLabel").getBytesList.getValueList.asScala.map((byteArray) => byteArray.asScala.toArray.map(_.toByte))
      assert(binaryArrayValue.toArray.deep == Array(byteArray, byteArray1).deep)

      assert(featureMap("BooleanArrayLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("BooleanArrayLabel").getInt64List.getValueList.asScala.toSeq.map(_.toLong) === booleanArray.map(if (_) 1 else 0))
    }

    "Encode given Row as TensorFlow SequenceExample" in {

      val schemaStructType = StructType(Array(
        StructField("IntegerLabel", IntegerType),
        StructField("StringArrayLabel", ArrayType(StringType)),
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
        StructField("DoubleArrayOfArrayLabel", ArrayType(ArrayType(DoubleType))),
        StructField("DecimalArrayOfArrayLabel", ArrayType(ArrayType(DataTypes.createDecimalType()))),
        StructField("StringArrayOfArrayLabel", ArrayType(ArrayType(StringType))),
        StructField("BinaryArrayOfArrayLabel", ArrayType(ArrayType(BinaryType)))
      ))

      val stringList = Seq("r1", "r2", "r3")
      val longListOfLists = Seq(Seq(3L, 5L), Seq(-8L, 0L))
      val floatListOfLists = Seq(Seq(1.5F, -6.5F), Seq(-8.2F, 0F))
      val doubleListOfLists = Seq(Seq(3.0), Seq(6.0, 9.0))
      val decimalListOfLists = Seq(Seq(Decimal(2.0), Decimal(4.0)), Seq(Decimal(6.0)))
      val stringListOfLists = Seq(Seq("r1"), Seq("r2", "r3"), Seq("r4"))
      val binaryListOfLists = stringListOfLists.map(stringList => stringList.map(_.getBytes))

      val rowWithSchema = new GenericRowWithSchema(Array[Any](10, stringList, longListOfLists, floatListOfLists,
        doubleListOfLists, decimalListOfLists, stringListOfLists, binaryListOfLists), schemaStructType)

      //Encode Sql Row to TensorFlow example
      val seqExample = DefaultTfRecordRowEncoder.encodeSequenceExample(rowWithSchema)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = seqExample.getContext.getFeatureMap.asScala
      val featureListMap = seqExample.getFeatureLists.getFeatureListMap.asScala

      assert(featureMap.size == 2)
      assert(featureMap("IntegerLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerLabel").getInt64List.getValue(0).toInt == 10)
      assert(featureMap("StringArrayLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("StringArrayLabel").getBytesList.getValueList.asScala.map(_.toStringUtf8) === stringList)

      assert(featureListMap.size == 6)
      assert(featureListMap("LongArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getInt64List.getValueList.asScala.toSeq) === longListOfLists)
      assert(featureListMap("FloatArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(_.toFloat).toSeq) ~== floatListOfLists)
      assert(featureListMap("DoubleArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(_.toDouble).toSeq) ~== doubleListOfLists)
      assert(featureListMap("DecimalArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(x => Decimal(x.toDouble)).toSeq) ~== decimalListOfLists)
      assert(featureListMap("StringArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getBytesList.getValueList.asScala.map(_.toStringUtf8).toSeq) === stringListOfLists)
      assert(featureListMap("BinaryArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getBytesList.getValueList.asScala.map(byteList => byteList.asScala.toSeq)) === binaryListOfLists.map(_.map(_.toSeq)))
    }

    "Throw an exception for non-nullable data types" in {
      val schemaStructType = StructType(Array(
        StructField("NonNullLabel", ArrayType(FloatType), nullable = false)
      ))

      val rowWithSchema = new GenericRowWithSchema(Array[Any](null), schemaStructType)

      intercept[NullPointerException]{
        DefaultTfRecordRowEncoder.encodeExample(rowWithSchema)
      }

      intercept[NullPointerException]{
        DefaultTfRecordRowEncoder.encodeSequenceExample(rowWithSchema)
      }
    }

    "Omit null fields from Example for nullable data types" in {
      val schemaStructType = StructType(Array(
        StructField("NullLabel", ArrayType(FloatType), nullable = true),
        StructField("FloatArrayLabel", ArrayType(FloatType))
      ))

      val floatArray = Array(2.5F, 5.0F)
      val rowWithSchema = new GenericRowWithSchema(Array[Any](null, floatArray), schemaStructType)

      val example = DefaultTfRecordRowEncoder.encodeExample(rowWithSchema)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = example.getFeatures.getFeatureMap.asScala
      assert(featureMap.size == 1)
      assert(featureMap("FloatArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== floatArray.toSeq)
    }

    "Omit null fields from SequenceExample for nullable data types" in {
      val schemaStructType = StructType(Array(
        StructField("NullLabel", ArrayType(FloatType), nullable = true),
        StructField("FloatArrayLabel", ArrayType(FloatType))
      ))

      val floatArray = Array(2.5F, 5.0F)
      val rowWithSchema = new GenericRowWithSchema(Array[Any](null, floatArray), schemaStructType)

      val seqExample = DefaultTfRecordRowEncoder.encodeSequenceExample(rowWithSchema)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = seqExample.getContext.getFeatureMap.asScala
      val featureListMap = seqExample.getFeatureLists.getFeatureListMap.asScala
      assert(featureMap.size == 1)
      assert(featureListMap.isEmpty)
      assert(featureMap("FloatArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== floatArray.toSeq)
    }

    "Throw an exception for unsupported data types" in {

      val schemaStructType = StructType(Array(
        StructField("TimestampLabel", TimestampType)
      ))

      val rowWithSchema = new GenericRowWithSchema(Array[Any]("2017/07/01 18:00"), schemaStructType)

      intercept[RuntimeException]{
        DefaultTfRecordRowEncoder.encodeExample(rowWithSchema)
      }

      intercept[RuntimeException]{
        DefaultTfRecordRowEncoder.encodeSequenceExample(rowWithSchema)
      }
    }
  }
}
