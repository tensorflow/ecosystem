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
package org.tensorflow.spark.datasources.tfrecords

import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.sql._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.StructType
import org.tensorflow.hadoop.io.TFRecordFileOutputFormat
import org.tensorflow.spark.datasources.tfrecords.serde.DefaultTfRecordRowEncoder

/**
 * Provides access to TensorFlow record source
 */
class DefaultSource extends DataSourceRegister
      with CreatableRelationProvider
      with RelationProvider
      with SchemaRelationProvider{

  /**
   * Short alias for spark-tensorflow data source.
   */
  override def shortName(): String = "tensorflow"

  // Writes DataFrame as TensorFlow Records
  override def createRelation(
    sqlContext: SQLContext,
    mode: SaveMode,
    parameters: Map[String, String],
    data: DataFrame): BaseRelation = {

    val path = parameters("path")

    //Export DataFrame as TFRecords
    val features = data.rdd.map(row => {
      val example = DefaultTfRecordRowEncoder.encodeTfRecord(row)
      (new BytesWritable(example.toByteArray), NullWritable.get())
    })
    features.saveAsNewAPIHadoopFile[TFRecordFileOutputFormat](path)

    TensorflowRelation(parameters)(sqlContext.sparkSession)
  }

  override def createRelation(sqlContext: SQLContext,
                      parameters: Map[String, String],
                      schema: StructType): BaseRelation = {
    TensorflowRelation(parameters, Some(schema))(sqlContext.sparkSession)
  }

  // Reads TensorFlow Records into DataFrame
  override def createRelation(sqlContext: SQLContext, parameters: Map[String, String]): TensorflowRelation = {
    TensorflowRelation(parameters)(sqlContext.sparkSession)
  }
}
