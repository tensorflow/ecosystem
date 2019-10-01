package org.tensorflow.spark.datasources.tfrecords.udf

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, struct}
import org.apache.spark.sql.types._
import org.tensorflow.example.{Example, Feature, SequenceExample}
import org.tensorflow.spark.datasources.tfrecords.SharedSparkSessionSuite
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

import scala.collection.JavaConverters._

class DataFrameTfrConverterTest extends SharedSparkSessionSuite {

  "DataFrame to tfr" should {
    "Encode given Row as TensorFlow Example" in {

      val schema = StructType(Array(
        StructField("IntegerLabel", IntegerType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("DoubleArrayLabel", ArrayType(DoubleType)),
        StructField("StrLabel", StringType),
        StructField("StrArrayLabel", ArrayType(StringType)),
        StructField("DenseVectorLabel", VectorType),
        StructField("SparseVectorLabel", VectorType),
        StructField("BinaryLabel", BinaryType),
        StructField("BinaryArrayLabel", ArrayType(BinaryType))
      ))
      val doubleArray = Array(1.1, 111.1, 11111.1)
      val sparseVector = Vectors.sparse(3, Seq((1, 2.0), (2, 1.5)))
      val denseVector = Vectors.dense(Array(5.6, 7.0))
      val byteArray = Array[Byte](0xde.toByte, 0xad.toByte, 0xbe.toByte, 0xef.toByte)
      val byteArray1 = Array[Byte](-128, 23, 127)

      val data =
          Row(1, 23L, 10.0F, 14.0, doubleArray,
              "r1", Seq("r2", "r3"), denseVector, sparseVector,
              byteArray, Seq(byteArray, byteArray1)) ::
          Nil

      val input = spark
        .createDataFrame(spark.sparkContext.makeRDD(data), schema)

      val examples = input
        .select(DataFrameTfrConverter.getRowToTFRecordExampleUdf(struct(input.columns.map(col): _*)).as("tfr"))
        .collect()
        .map {case Row(tfr: Array[Byte]) =>
          Example.parseFrom(tfr)
        }
        .toList

      val featureMap = examples.head.getFeatures.getFeatureMap.asScala

      assert(featureMap("IntegerLabel").getInt64List.getValue(0).toInt == 1)
      assert(featureMap("LongLabel").getInt64List.getValue(0).toInt == 23)
      assert(featureMap("FloatLabel").getFloatList.getValue(0) == 10.0F)
      assert(featureMap("DoubleLabel").getFloatList.getValue(0) == 14.0F)
      assert(featureMap("DoubleArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== doubleArray.map(_.toFloat))
      assert(featureMap("StrLabel").getBytesList.getValue(0).toStringUtf8 == "r1")
      assert(featureMap("StrArrayLabel").getBytesList.getValueList.asScala.map(_.toStringUtf8) === Seq("r2", "r3"))
      assert(featureMap("DenseVectorLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== denseVector.toArray.map(_.toFloat))
      assert(featureMap("SparseVectorLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== sparseVector.toDense.toArray.map(_.toFloat))
      assert(featureMap("BinaryLabel").getBytesList.getValue(0).toByteArray.deep == byteArray.deep)
      val binaryArrayValue = featureMap("BinaryArrayLabel").getBytesList.getValueList.asScala.map(byteArray => byteArray.asScala.toArray.map(_.toByte))
      assert(binaryArrayValue.toArray.deep == Array(byteArray, byteArray1).deep)
    }

    "Encode given Row as TensorFlow SequenceExample" in {

      val schemaStructType = StructType(Array(
        StructField("IntegerLabel", IntegerType),
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
        StructField("DoubleArrayOfArrayLabel", ArrayType(ArrayType(DoubleType))),
        StructField("StringArrayOfArrayLabel", ArrayType(ArrayType(StringType))),
        StructField("BinaryArrayOfArrayLabel", ArrayType(ArrayType(BinaryType)))
      ))

      val longListOfLists = Seq(Seq(3L, 5L), Seq(-8L, 0L))
      val floatListOfLists = Seq(Seq(1.5F, -6.5F), Seq(-8.2F, 0F))
      val doubleListOfLists = Seq(Seq(3.0), Seq(6.0, 9.0))
      val stringListOfLists = Seq(Seq("r1"), Seq("r2", "r3"), Seq("r4"))
      val binaryListOfLists = stringListOfLists.map(stringList => stringList.map(_.getBytes))

      val data =
        Row(10, longListOfLists,
          floatListOfLists, doubleListOfLists,
          stringListOfLists, binaryListOfLists) ::
          Nil

      val input = spark
        .createDataFrame(spark.sparkContext.makeRDD(data), schemaStructType)

      val examples = input
        .select(DataFrameTfrConverter.getRowToTFRecordSequenceExampleUdf(struct(input.columns.map(col): _*)).as("tfr"))
        .collect()
        .map {case Row(tfr: Array[Byte]) =>
          SequenceExample.parseFrom(tfr)
        }
        .toList

      val featureMap = examples.head.getContext.getFeatureMap.asScala
      val featureListMap = examples.head.getFeatureLists.getFeatureListMap.asScala

      assert(featureMap.size == 1)
      assert(featureMap("IntegerLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerLabel").getInt64List.getValue(0).toInt == 10)

      assert(featureListMap.size == 5)
      assert(featureListMap("LongArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getInt64List.getValueList.asScala.toSeq) === longListOfLists)
      assert(featureListMap("FloatArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(_.toFloat).toSeq) ~== floatListOfLists)
      assert(featureListMap("DoubleArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(_.toDouble).toSeq) ~== doubleListOfLists)
      assert(featureListMap("StringArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getBytesList.getValueList.asScala.map(_.toStringUtf8).toSeq) === stringListOfLists)
      assert(featureListMap("BinaryArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getBytesList.getValueList.asScala.map(byteList => byteList.asScala.toSeq)) === binaryListOfLists.map(_.map(_.toSeq)))
    }
  }
}