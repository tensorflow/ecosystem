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

import org.tensorflow.example._
import com.google.protobuf.ByteString

trait FeatureEncoder[T] {
  /**
   * Encodes input value as TensorFlow "Feature"
   *
   * Maps input value to one of Int64List, FloatList, BytesList
   *
   * @param value Input value
   * @return TensorFlow Feature
   */
  def encode(value: T): Feature
}

/**
 * Encode input value to Int64List
 */
object Int64ListFeatureEncoder extends FeatureEncoder[Seq[Long]] {
  override def encode(value: Seq[Long]): Feature = {
    val intListBuilder = Int64List.newBuilder()
    value.foreach {x =>
      intListBuilder.addValue(x)
    }
    val int64List = intListBuilder.build()
    Feature.newBuilder().setInt64List(int64List).build()
  }
}

/**
 * Encode input value to FloatList
 */
object FloatListFeatureEncoder extends FeatureEncoder[Seq[Float]] {
  override def encode(value: Seq[Float]): Feature = {
    val floatListBuilder = FloatList.newBuilder()
    value.foreach {x =>
      floatListBuilder.addValue(x)
    }
    val floatList = floatListBuilder.build()
    Feature.newBuilder().setFloatList(floatList).build()
  }
}

/**
 * Encode input value to ByteList
 */
object BytesListFeatureEncoder extends FeatureEncoder[Seq[Array[Byte]]] {
  override def encode(value: Seq[Array[Byte]]): Feature = {
    val bytesListBuilder = BytesList.newBuilder()
    value.foreach {x =>
      bytesListBuilder.addValue(ByteString.copyFrom(x))
    }
    val bytesList = bytesListBuilder.build()
    Feature.newBuilder().setBytesList(bytesList).build()
  }
}