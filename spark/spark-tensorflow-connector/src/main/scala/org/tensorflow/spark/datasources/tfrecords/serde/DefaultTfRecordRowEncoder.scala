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
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
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
  def encodeExample(row: Row): Example

  /**
   * Encodes each Row as TensorFlow "SequenceExample"
   *
   * Maps each column in Row to one of Int64List, FloatList, BytesList or FeatureList based on the column data type
   *
   * @param row a DataFrame row
   * @return TensorFlow SequenceExample
   */
  def encodeSequenceExample(row: Row): SequenceExample
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
  def encodeExample(row: Row): Example = {
    val features = Features.newBuilder()
    val example = Example.newBuilder()

    row.schema.zipWithIndex.foreach {
      case (structField, index) =>
        if (row.get(index) != null) {
          val feature = encodeFeature(row, structField, index)
          features.putFeature(structField.name, feature)
        }
        else if (!structField.nullable) {
          throw new NullPointerException(s"${structField.name} does not allow null values")
        }
    }

    example.setFeatures(features.build())
    example.build()
  }

  /**
   * Encodes each Row as TensorFlow "SequenceExample"
   *
   * Maps each column in Row to one of Int64List, FloatList, BytesList or FeatureList based on the column data type
   *
   * @param row a DataFrame row
   * @return TensorFlow SequenceExample
   */
  def encodeSequenceExample(row: Row): SequenceExample = {
    val features = Features.newBuilder()
    val featureLists = FeatureLists.newBuilder()
    val sequenceExample = SequenceExample.newBuilder()

    row.schema.zipWithIndex.foreach {
      case (structField, index) if row.get(index) == null => {
        if (!structField.nullable) {
          throw new NullPointerException(s"${structField.name}  does not allow null values")
        }
      }
      case (structField, index) => structField.dataType match {
        case ArrayType(ArrayType(_, _), _) =>
          val featureList = encodeFeatureList(row, structField, index)
          featureLists.putFeatureList(structField.name, featureList)
        case _ =>
          val feature = encodeFeature(row, structField, index)
          features.putFeature(structField.name, feature)
      }
    }

    sequenceExample.setContext(features.build())
    sequenceExample.setFeatureLists(featureLists.build())
    sequenceExample.build()
  }

  //Encode field in row to TensorFlow Feature
  private def encodeFeature(row: Row, structField: StructField, index: Int): Feature = {
    val feature = structField.dataType match {
      case IntegerType => Int64ListFeatureEncoder.encode(Seq(row.getInt(index).toLong))
      case BooleanType =>   Int64ListFeatureEncoder.encode(Seq(if (row.getBoolean(index)) 1.toLong else 0.toLong))
      case LongType => Int64ListFeatureEncoder.encode(Seq(row.getLong(index)))
      case FloatType => FloatListFeatureEncoder.encode(Seq(row.getFloat(index)))
      case DoubleType => FloatListFeatureEncoder.encode(Seq(row.getDouble(index).toFloat))
      case DecimalType() => FloatListFeatureEncoder.encode(Seq(row.getAs[Decimal](index).toFloat))
      case StringType => BytesListFeatureEncoder.encode(Seq(row.getString(index).getBytes))
      case BinaryType => BytesListFeatureEncoder.encode(Seq(row.getAs[Array[Byte]](index)))
      case ArrayType(BooleanType, _) =>
        Int64ListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index)).toBooleanArray().map(if (_) 1.toLong else 0.toLong))
      case ArrayType(IntegerType, _)  =>
        Int64ListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index)).toIntArray().map(_.toLong))
      case ArrayType(LongType, _) =>
        Int64ListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index)).toLongArray())
      case ArrayType(FloatType, _) =>
        FloatListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index)).toFloatArray())
      case ArrayType(DoubleType, _) =>
        FloatListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index)).toDoubleArray().map(_.toFloat))
      case ArrayType(DecimalType(), _) =>
        val decimalArray = ArrayData.toArrayData(row.get(index)).toArray[Decimal](DataTypes.createDecimalType())
        FloatListFeatureEncoder.encode(decimalArray.map(_.toFloat))
      case ArrayType(StringType, _) =>
        BytesListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index))
          .toArray[String](ObjectType(classOf[String])).map(_.getBytes))
      case ArrayType(BinaryType, _) =>
        BytesListFeatureEncoder.encode(ArrayData.toArrayData(row.get(index)).toArray[Array[Byte]](BinaryType))
      case VectorType => {
        val field = row.get(index)
        field match {
          case v: SparseVector => FloatListFeatureEncoder.encode(v.toDense.toArray.map(_.toFloat))
          case v: DenseVector => FloatListFeatureEncoder.encode(v.toArray.map(_.toFloat))
          case _ => throw new RuntimeException(s"Cannot convert $field to vector")
        }
      }
      case _ => throw new RuntimeException(s"Cannot convert field to unsupported data type ${structField.dataType}")
    }
    feature
  }

  //Encode field in row to TensorFlow FeatureList
  def encodeFeatureList(row: Row, structField: StructField, index: Int): FeatureList = {
    val featureList = structField.dataType match {
      case ArrayType(ArrayType(IntegerType, _), _) =>
        val longArrays = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toIntArray().map(_.toLong).toSeq
        }
        Int64FeatureListEncoder.encode(longArrays)

      case ArrayType(ArrayType(LongType, _), _) =>
        val longArrays = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toLongArray().toSeq
        }
        Int64FeatureListEncoder.encode(longArrays)

      case ArrayType(ArrayType(FloatType, _), _) =>
        val floatArrays = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toFloatArray().toSeq
        }
        FloatFeatureListEncoder.encode(floatArrays)

      case ArrayType(ArrayType(DoubleType, _), _) =>
        val floatArrays = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toDoubleArray().map(_.toFloat).toSeq
        }
        FloatFeatureListEncoder.encode(floatArrays)

      case ArrayType(ArrayType(DecimalType(), _), _) =>
        val floatArrays = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toArray[Decimal](DataTypes.createDecimalType()).map(_.toFloat).toSeq
        }
        FloatFeatureListEncoder.encode(floatArrays)

      case ArrayType(ArrayType(StringType, _), _) =>
        val arrayData = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toArray[String](ObjectType(classOf[String])).toSeq.map(_.getBytes)
        }.toSeq
        BytesFeatureListEncoder.encode(arrayData)

      case ArrayType(ArrayType(BinaryType, _), _) =>
        val arrayData = ArrayData.toArrayData(row.get(index)).array.map {arr =>
          ArrayData.toArrayData(arr).toArray[Array[Byte]](BinaryType).toSeq
        }.toSeq
        BytesFeatureListEncoder.encode(arrayData)

      case _ => throw new RuntimeException(s"Cannot convert row element ${row.get(index)} to FeatureList.")
    }
    featureList
  }
}
