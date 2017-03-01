/**
 *  Copyright (c) 2016 Intel Corporation 
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

import org.tensorflow.example.{BytesList, Feature, FloatList, Int64List}
import org.tensorflow.hadoop.shaded.protobuf.ByteString
import org.tensorflow.spark.datasources.tfrecords.DataTypesConvertor

trait FeatureEncoder {
  /**
   * Encodes input value as TensorFlow "Feature"
   *
   * Maps input value to one of Int64List, FloatList, BytesList
   *
   * @param value Input value
   * @return TensorFlow Feature
   */
  def encode(value: Any): Feature
}

/**
 * Encode input value to Int64List
 */
object Int64ListFeatureEncoder extends FeatureEncoder {
  override def encode(value: Any): Feature = {
    try {
      val int64List = value match {
        case i: Int => Int64List.newBuilder().addValue(i.toLong).build()
        case l: Long => Int64List.newBuilder().addValue(l).build()
        case arr: scala.collection.mutable.WrappedArray[_] => toInt64List(arr.toArray[Any])
        case arr: Array[_] => toInt64List(arr)
        case seq: Seq[_] => toInt64List(seq.toArray[Any])
        case _ => throw new RuntimeException(s"Cannot convert object $value to Int64List")
      }
      Feature.newBuilder().setInt64List(int64List).build()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert object $value of type ${value.getClass} to Int64List feature.", ex)
    }
  }

  private def toInt64List[T](arr: Array[T]): Int64List = {
    val intListBuilder = Int64List.newBuilder()
    arr.foreach(x => {
      require(x != null, "Int64List with null values is not supported")
      val longValue = DataTypesConvertor.toLong(x)
      intListBuilder.addValue(longValue)
    })
    intListBuilder.build()
  }
}

/**
 * Encode input value to FloatList
 */
object FloatListFeatureEncoder extends FeatureEncoder {
  override def encode(value: Any): Feature = {
    try {
      val floatList = value match {
        case i: Int => FloatList.newBuilder().addValue(i.toFloat).build()
        case l: Long => FloatList.newBuilder().addValue(l.toFloat).build()
        case f: Float => FloatList.newBuilder().addValue(f).build()
        case d: Double => FloatList.newBuilder().addValue(d.toFloat).build()
        case arr: scala.collection.mutable.WrappedArray[_] => toFloatList(arr.toArray[Any])
        case arr: Array[_] => toFloatList(arr)
        case seq: Seq[_] => toFloatList(seq.toArray[Any])
        case _ => throw new RuntimeException(s"Cannot convert object $value to FloatList")
      }
      Feature.newBuilder().setFloatList(floatList).build()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert object $value of type ${value.getClass} to FloatList feature.", ex)
    }
  }

  private def toFloatList[T](arr: Array[T]): FloatList = {
    val floatListBuilder = FloatList.newBuilder()
    arr.foreach(x => {
      require(x != null, "FloatList with null values is not supported")
      val longValue = DataTypesConvertor.toFloat(x)
      floatListBuilder.addValue(longValue)
    })
    floatListBuilder.build()
  }
}

/**
 * Encode input value to ByteList
 */
object BytesListFeatureEncoder extends FeatureEncoder {
  override def encode(value: Any): Feature = {
    try {
      val byteList = BytesList.newBuilder().addValue(ByteString.copyFrom(value.toString.getBytes)).build()
      Feature.newBuilder().setBytesList(byteList).build()
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert object $value of type ${value.getClass} to ByteList feature.", ex)
    }
  }
}


