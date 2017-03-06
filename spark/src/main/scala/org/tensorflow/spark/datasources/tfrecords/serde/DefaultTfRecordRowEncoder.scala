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

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.tensorflow.example._

trait TfRecordRowEncoder {
  /**
   * Encodes each Row as TensorFlow "Example"
   *
   * Maps each column in Row to one of Int64List, FloatList, BytesList based on the column data type
   *
   * @param row a DataFrame row
   * @return TensorFlow Example
   */
  def encodeTfRecord(row: Row): Example
}

object DefaultTfRecordRowEncoder extends TfRecordRowEncoder {

  /**
   * Encodes each Row as TensorFlow "Example"
   *
   * Maps each column in Row to one of Int64List, FloatList, BytesList based on the column data type
   *
   * @param row a DataFrame row
   * @return TensorFlow Example
   */
  def encodeTfRecord(row: Row): Example = {
    val features = Features.newBuilder()
    val example = Example.newBuilder()

    row.schema.zipWithIndex.map {
      case (structField, index) =>
        val value = row.get(index)
        val feature = structField.dataType match {
          case IntegerType | LongType => Int64ListFeatureEncoder.encode(value)
          case FloatType | DoubleType => FloatListFeatureEncoder.encode(value)
          case ArrayType(IntegerType, _) | ArrayType(LongType, _) => Int64ListFeatureEncoder.encode(value)
          case ArrayType(DoubleType, _) => FloatListFeatureEncoder.encode(value)
          case _ => BytesListFeatureEncoder.encode(value)
        }
        features.putFeature(structField.name, feature)
    }

    features.build()
    example.setFeatures(features)
    example.build()
  }
}

