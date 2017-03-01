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

import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.tensorflow.example._
import scala.collection.JavaConverters._

trait TfRecordRowDecoder {
  /**
   * Decodes each TensorFlow "Example" as DataFrame "Row"
   *
   * Maps each feature in Example to element in Row with DataType based on custom schema or
   * default mapping of Int64List, FloatList, BytesList to column data type
   *
   * @param example TensorFlow Example to decode
   * @param schema Decode Example using specified schema
   * @return a DataFrame row
   */
  def decodeTfRecord(example: Example, schema: StructType): Row
}

object DefaultTfRecordRowDecoder extends TfRecordRowDecoder {

  /**
   * Decodes each TensorFlow "Example" as DataFrame "Row"
   *
   * Maps each feature in Example to element in Row with DataType based on custom schema
   *
   * @param example TensorFlow Example to decode
   * @param schema Decode Example using specified schema
   * @return a DataFrame row
   */
  def decodeTfRecord(example: Example, schema: StructType): Row = {
    val row = Array.fill[Any](schema.length)(null)
    example.getFeatures.getFeatureMap.asScala.foreach {
      case (featureName, feature) =>
        val index = schema.fieldIndex(featureName)
        val colDataType = schema.fields(index).dataType
        row(index) = colDataType match {
          case IntegerType => IntFeatureDecoder.decode(feature)
          case LongType => LongFeatureDecoder.decode(feature)
          case FloatType => FloatFeatureDecoder.decode(feature)
          case DoubleType => DoubleFeatureDecoder.decode(feature)
          case ArrayType(IntegerType, true) => IntListFeatureDecoder.decode(feature)
          case ArrayType(LongType, _) => LongListFeatureDecoder.decode(feature)
          case ArrayType(FloatType, _) => FloatListFeatureDecoder.decode(feature)
          case ArrayType(DoubleType, _) => DoubleListFeatureDecoder.decode(feature)
          case StringType => StringFeatureDecoder.decode(feature)
          case _ => throw new RuntimeException(s"Cannot convert feature to unsupported data type ${colDataType}")
        }
    }
    Row.fromSeq(row)
  }
}

