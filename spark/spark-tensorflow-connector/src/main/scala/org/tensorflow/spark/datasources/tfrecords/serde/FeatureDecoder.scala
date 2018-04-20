/**
 *  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.tensorflow.spark.datasources.tfrecords.serde

import org.apache.spark.sql.types.Decimal
import org.tensorflow.example.Feature
import scala.collection.JavaConverters._

trait FeatureDecoder[T] {
  /**
   * Decodes each TensorFlow "Feature" to desired Scala type
   * 
   * @param feature TensorFlow Feature
   * @return Decoded feature
   */
  def decode(feature: Feature): T
}

/**
 * Decode TensorFlow "Feature" to Integer
 */
object IntFeatureDecoder extends FeatureDecoder[Int] {
  override def decode(feature: Feature): Int = {
    require(feature != null && feature.getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER, "Feature must be of type Int64List")
    try {
      val int64List = feature.getInt64List.getValueList
      require(int64List.size() == 1, "Length of Int64List must equal 1")
      int64List.get(0).intValue()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to integer.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Integer array
 */
object IntListFeatureDecoder extends FeatureDecoder[Seq[Int]] {
  override def decode(feature: Feature): Seq[Int] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER, "Feature must be of type Int64List")
    try {
      val array = feature.getInt64List.getValueList.asScala.toSeq
      array.map(_.toInt)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Integer array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Long
 */
object LongFeatureDecoder extends FeatureDecoder[Long] {
  override def decode(feature: Feature): Long = {
    require(feature != null && feature.getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER, "Feature must be of type Int64List")
    try {
      val int64List = feature.getInt64List.getValueList
      require(int64List.size() == 1, "Length of Int64List must equal 1")
      int64List.get(0).longValue()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Long.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Long array
 */
object LongListFeatureDecoder extends FeatureDecoder[Seq[Long]] {
  override def decode(feature: Feature): Seq[Long] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER, "Feature must be of type Int64List")
    try {
      val array = feature.getInt64List.getValueList.asScala.toSeq
      array.map(_.toLong)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Long array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Float
 */
object FloatFeatureDecoder extends FeatureDecoder[Float] {
  override def decode(feature: Feature): Float = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val floatList = feature.getFloatList.getValueList
      require(floatList.size() == 1, "Length of FloatList must equal 1")
      floatList.get(0).floatValue()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Float.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Float array
 */
object FloatListFeatureDecoder extends FeatureDecoder[Seq[Float]] {
  override def decode(feature: Feature): Seq[Float] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val array = feature.getFloatList.getValueList.asScala.toSeq
      array.map(_.toFloat)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Float array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Double
 */
object DoubleFeatureDecoder extends FeatureDecoder[Double] {
  override def decode(feature: Feature): Double = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val floatList = feature.getFloatList.getValueList
      require(floatList.size() == 1, "Length of FloatList must equal 1")
      floatList.get(0).doubleValue()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Double.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Decimal
 */
object DecimalFeatureDecoder extends FeatureDecoder[Decimal] {
  override def decode(feature: Feature): Decimal = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val floatList = feature.getFloatList.getValueList
      require(floatList.size() == 1, "Length of FloatList must equal 1")
      Decimal(floatList.get(0).doubleValue())
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Decimal.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Double array
 */
object DoubleListFeatureDecoder extends FeatureDecoder[Seq[Double]] {
  override def decode(feature: Feature): Seq[Double] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val array = feature.getFloatList.getValueList.asScala.toSeq
      array.map(_.toDouble)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Double array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to Decimal array
 */
object DecimalListFeatureDecoder extends FeatureDecoder[Seq[Decimal]] {
  override def decode(feature: Feature): Seq[Decimal] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val array = feature.getFloatList.getValueList.asScala.toSeq
      array.map(x => Decimal(x.toDouble))
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Decimal array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to byte array
 */
object BinaryFeatureDecoder extends FeatureDecoder[Array[Byte]] {
  override def decode(feature: Feature): Array[Byte] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER, "Feature must be of type ByteList")
    try {
      val bytesList = feature.getBytesList.getValueList
      require(bytesList.size() == 1, "Length of BytesList must equal 1")
      bytesList.get(0).asScala.toArray.map(_.toByte)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to byte array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to array of byte arrays
 */
object BinaryListFeatureDecoder extends FeatureDecoder[Seq[Array[Byte]]] {
  override def decode(feature: Feature): Seq[Array[Byte]] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER, "Feature must be of type ByteList")
    try {
      feature.getBytesList.getValueList.asScala.map((byteArray) => byteArray.asScala.toArray.map(_.toByte))
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to byte array.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to String
 */
object StringFeatureDecoder extends FeatureDecoder[String] {
  override def decode(feature: Feature): String = {
    require(feature != null && feature.getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER, "Feature must be of type ByteList")
    try {
      val bytesList = feature.getBytesList.getValueList
      require(bytesList.size() == 1, "Length of BytesList must equal 1")
      bytesList.get(0).toStringUtf8
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to String.", ex)
    }
  }
}

/**
 * Decode TensorFlow "Feature" to String array
 */
object StringListFeatureDecoder extends FeatureDecoder[Seq[String]] {
  override def decode(feature: Feature): Seq[String] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER, "Feature must be of type ByteList")
    try {
      val array = feature.getBytesList.getValueList.asScala.toSeq
      array.map(_.toStringUtf8)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to String array.", ex)
    }
  }
}

