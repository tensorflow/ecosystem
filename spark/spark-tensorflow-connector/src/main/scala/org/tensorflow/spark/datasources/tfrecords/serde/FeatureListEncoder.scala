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

import org.tensorflow.example.FeatureList

trait FeatureListEncoder[T] extends Serializable{
  /**
   * Encodes input value as TensorFlow "FeatureList"
   *
   * Maps input value to a feature list of type Int64List, FloatList, or BytesList
   *
   * @param values Input values
   * @return TensorFlow FeatureList
   */
  def encode(values: T): FeatureList
}


/**
 * Encode 2-dimensional Long array to TensorFlow "FeatureList" of type Int64List
 */
object Int64FeatureListEncoder extends FeatureListEncoder[Seq[Seq[Long]]] {
  def encode(values: Seq[Seq[Long]]) : FeatureList = {
    val builder =  FeatureList.newBuilder()
    values.foreach { x =>
      val int64list = Int64ListFeatureEncoder.encode(x)
      builder.addFeature(int64list)
    }
    builder.build()
  }
}

/**
 * Encode 2-dimensional Float array to TensorFlow "FeatureList" of type FloatList
 */
object FloatFeatureListEncoder extends FeatureListEncoder[Seq[Seq[Float]]] {
  def encode(value: Seq[Seq[Float]]) : FeatureList = {
    val builder =  FeatureList.newBuilder()
    value.foreach { x =>
      val floatList = FloatListFeatureEncoder.encode(x)
      builder.addFeature(floatList)
    }
    builder.build()
  }
}

/**
 * Encode 2-dimensional String array to TensorFlow "FeatureList" of type BytesList
 */
object BytesFeatureListEncoder extends FeatureListEncoder[Seq[Seq[Array[Byte]]]] {
  def encode(value: Seq[Seq[Array[Byte]]]) : FeatureList = {
    val builder =  FeatureList.newBuilder()
    value.foreach { x =>
      val bytesList = BytesListFeatureEncoder.encode(x)
      builder.addFeature(bytesList)
    }
    builder.build()
  }
}
