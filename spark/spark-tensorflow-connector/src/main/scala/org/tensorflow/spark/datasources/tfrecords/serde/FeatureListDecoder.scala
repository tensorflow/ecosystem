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
import org.tensorflow.example.FeatureList
import scala.collection.JavaConverters._

trait FeatureListDecoder[T] extends Serializable{
  /**
   * Decodes each TensorFlow "FeatureList" to desired Scala type
   *
   * @param featureList TensorFlow FeatureList
   * @return Decoded featureList
   */
  def decode(featureList: FeatureList): T
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional Integer array
 */
object IntFeatureListDecoder extends FeatureListDecoder[Seq[Seq[Int]]] {
  override def decode(featureList: FeatureList): Seq[Seq[Int]] = {
    featureList.getFeatureList.asScala.map(x => IntListFeatureDecoder.decode(x)).toSeq
  }
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional Long array
 */
object LongFeatureListDecoder extends FeatureListDecoder[Seq[Seq[Long]]] {
  override def decode(featureList: FeatureList): Seq[Seq[Long]] = {
    featureList.getFeatureList.asScala.map(x => LongListFeatureDecoder.decode(x)).toSeq
  }
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional Float array
 */
object FloatFeatureListDecoder extends FeatureListDecoder[Seq[Seq[Float]]] {
  override def decode(featureList: FeatureList): Seq[Seq[Float]] = {
    featureList.getFeatureList.asScala.map(x => FloatListFeatureDecoder.decode(x)).toSeq
  }
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional Double array
 */
object DoubleFeatureListDecoder extends FeatureListDecoder[Seq[Seq[Double]]] {
  override def decode(featureList: FeatureList): Seq[Seq[Double]] = {
    featureList.getFeatureList.asScala.map(x => DoubleListFeatureDecoder.decode(x)).toSeq
  }
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional Decimal array
 */
object DecimalFeatureListDecoder extends FeatureListDecoder[Seq[Seq[Decimal]]] {
  override def decode(featureList: FeatureList): Seq[Seq[Decimal]] = {
    featureList.getFeatureList.asScala.map(x => DecimalListFeatureDecoder.decode(x)).toSeq
  }
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional String array
 */
object StringFeatureListDecoder extends FeatureListDecoder[Seq[Seq[String]]] {
  override def decode(featureList: FeatureList): Seq[Seq[String]] = {
    featureList.getFeatureList.asScala.map(x => StringListFeatureDecoder.decode(x)).toSeq
  }
}

/**
 * Decode TensorFlow "FeatureList" to 2-dimensional array of Array[Byte] (a 3-dimensional array)
 */
object BinaryFeatureListDecoder extends FeatureListDecoder[Seq[Seq[Array[Byte]]]] {
  override def decode(featureList: FeatureList): Seq[Seq[Array[Byte]]] = {
    featureList.getFeatureList.asScala.map(x => BinaryListFeatureDecoder.decode(x)).toSeq
  }
}
