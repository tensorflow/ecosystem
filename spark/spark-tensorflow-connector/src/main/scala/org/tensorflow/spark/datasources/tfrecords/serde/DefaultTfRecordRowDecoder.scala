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

import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.tensorflow.example._
import scala.collection.JavaConverters._

trait TfRecordRowDecoder {
  /**
   * Decodes each TensorFlow "Example" as DataFrame "Row"
   *
   * Maps each feature in Example to element in Row with DataType based on custom schema
   *
   * @param example TensorFlow Example to decode
   * @param schema Decode Example using specified schema
   * @return a DataFrame row
   */
  def decodeExample(example: Example, schema: StructType): Row

  /**
   * Decodes each TensorFlow "SequenceExample" as DataFrame "Row"
   *
   * Maps each feature in SequenceExample to element in Row with DataType based on custom schema
   *
   * @param sequenceExample TensorFlow SequenceExample to decode
   * @param schema Decode SequenceExample using specified schema
   * @return a DataFrame row
   */
  def decodeSequenceExample(sequenceExample: SequenceExample, schema: StructType): Row
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
  def decodeExample(example: Example, schema: StructType): Row = {
    val row = Array.fill[Any](schema.length)(null)
    val featureMap = example.getFeatures.getFeatureMap.asScala
    schema.fields.zipWithIndex.foreach {
      case (field, index) =>
        val feature = featureMap.get(field.name)
        feature match {
          case Some(f) => row(index) = decodeFeature(f, schema, index)
          case None => if (!field.nullable) throw new NullPointerException(s"Field ${field.name} does not allow null values")
        }
    }
    Row.fromSeq(row)
  }

  /**
   * Decodes each TensorFlow "SequenceExample" as DataFrame "Row"
   *
   * Maps each feature in SequenceExample to element in Row with DataType based on custom schema
   *
   * @param sequenceExample TensorFlow SequenceExample to decode
   * @param schema Decode Example using specified schema
   * @return a DataFrame row
   */
  def decodeSequenceExample(sequenceExample: SequenceExample, schema: StructType): Row = {
    val row = Array.fill[Any](schema.length)(null)

    //Decode features
    val featureMap = sequenceExample.getContext.getFeatureMap.asScala
    val featureListMap = sequenceExample.getFeatureLists.getFeatureListMap.asScala

    schema.fields.zipWithIndex.foreach {
      case (field, index) =>
        val feature = featureMap.get(field.name)

        feature match {
          case Some(f) => row(index) = decodeFeature(f, schema, index)
          case None => {
            featureListMap.get(field.name) match {
              case Some(list) => row(index) = decodeFeatureList(list, schema, index)
              case None => if (!field.nullable) throw new NullPointerException(s"Field ${field.name}  does not allow null values")
            }
          }
        }
    }

    Row.fromSeq(row)
  }

  // Decode Feature to Scala Type based on field in schema
  private def decodeFeature(feature: Feature, schema: StructType, fieldIndex: Int): Any = {
    val colDataType = schema.fields(fieldIndex).dataType

    colDataType match {
      case IntegerType => IntFeatureDecoder.decode(feature)
      case LongType => LongFeatureDecoder.decode(feature)
      case FloatType => FloatFeatureDecoder.decode(feature)
      case DoubleType => DoubleFeatureDecoder.decode(feature)
      case DecimalType() => DecimalFeatureDecoder.decode(feature)
      case StringType => StringFeatureDecoder.decode(feature)
      case BinaryType => BinaryFeatureDecoder.decode(feature)
      case ArrayType(IntegerType, _) => IntListFeatureDecoder.decode(feature)
      case ArrayType(LongType, _) => LongListFeatureDecoder.decode(feature)
      case ArrayType(FloatType, _) => FloatListFeatureDecoder.decode(feature)
      case ArrayType(DoubleType, _) => DoubleListFeatureDecoder.decode(feature)
      case ArrayType(DecimalType(), _) => DecimalListFeatureDecoder.decode(feature)
      case ArrayType(StringType, _) => StringListFeatureDecoder.decode(feature)
      case ArrayType(BinaryType, _) => BinaryListFeatureDecoder.decode(feature)
      case VectorType =>  Vectors.dense(DoubleListFeatureDecoder.decode(feature).toArray)
      case _ => throw new scala.RuntimeException(s"Cannot convert Feature to unsupported data type ${colDataType}")
    }
  }

  // Decode FeatureList to Scala Type based on field in schema
  private def decodeFeatureList(featureList: FeatureList, schema: StructType, fieldIndex: Int): Any = {
    val colDataType = schema.fields(fieldIndex).dataType
    colDataType match {
      case ArrayType(ArrayType(IntegerType, _), _) => IntFeatureListDecoder.decode(featureList)
      case ArrayType(ArrayType(LongType, _), _) => LongFeatureListDecoder.decode(featureList)
      case ArrayType(ArrayType(FloatType, _), _) => FloatFeatureListDecoder.decode(featureList)
      case ArrayType(ArrayType(DoubleType, _), _) => DoubleFeatureListDecoder.decode(featureList)
      case ArrayType(ArrayType(DecimalType(), _), _) => DecimalFeatureListDecoder.decode(featureList)
      case ArrayType(ArrayType(StringType, _), _) => StringFeatureListDecoder.decode(featureList)
      case ArrayType(ArrayType(BinaryType, _), _) => BinaryFeatureListDecoder.decode(featureList)
      case _ => throw new scala.RuntimeException(s"Cannot convert FeatureList to unsupported data type ${colDataType}")
    }
  }
}
