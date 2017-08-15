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
package org.tensorflow.spark.datasources.tfrecords

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
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
  override def shortName(): String = "tfrecords"

  // Writes DataFrame as TensorFlow Records
  override def createRelation(
    sqlContext: SQLContext,
    mode: SaveMode,
    parameters: Map[String, String],
    data: DataFrame): BaseRelation = {

    val path = parameters("path")

    val recordType = parameters.getOrElse("recordType", "Example")

    //Export DataFrame as TFRecords
    val features = data.rdd.map(row => {
      recordType match {
        case "Example" =>
          val example = DefaultTfRecordRowEncoder.encodeExample(row)
          (new BytesWritable(example.toByteArray), NullWritable.get())
        case "SequenceExample" =>
          val sequenceExample = DefaultTfRecordRowEncoder.encodeSequenceExample(row)
          (new BytesWritable(sequenceExample.toByteArray), NullWritable.get())
        case _ =>
          throw new IllegalArgumentException(s"Unsupported recordType ${recordType}: recordType can be Example or SequenceExample")
      }
    })

    val hadoopConf = sqlContext.sparkContext.hadoopConfiguration
    val outputPath = new Path(path)
    val fs = outputPath.getFileSystem(hadoopConf)
    val qualifiedOutputPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)

    val pathExists = fs.exists(qualifiedOutputPath)

    mode match {
        case SaveMode.Overwrite =>
            fs.delete(qualifiedOutputPath, true)
            save(features, path)

        case SaveMode.Append =>
            throw new IllegalArgumentException("Append mode is not supported")

        case SaveMode.ErrorIfExists =>
            if (pathExists)
                throw new IllegalStateException(
                    s"Path $path already exists. SaveMode: ErrorIfExists.")
            save(features, path)

        case SaveMode.Ignore =>
        // With `SaveMode.Ignore` mode, if data already exists, the save operation is expected
        // to not save the contents of the DataFrame and to not change the existing data.
        // Therefore, it is okay to do nothing here and then just return the relation below.
            if (pathExists == false)
                save(features, path)
    }

    TensorflowRelation(parameters)(sqlContext.sparkSession)
  }

  private def save(features: RDD[(BytesWritable, NullWritable)], path: String) = {
      features.saveAsNewAPIHadoopFile[TFRecordFileOutputFormat](path)
  }

  // Reads TensorFlow Records into DataFrame with Custom Schema
  override def createRelation(sqlContext: SQLContext,
                      parameters: Map[String, String],
                      schema: StructType): BaseRelation = {
    TensorflowRelation(parameters, Some(schema))(sqlContext.sparkSession)
  }

  // Reads TensorFlow Records into DataFrame with schema inferred
  override def createRelation(sqlContext: SQLContext, parameters: Map[String, String]): TensorflowRelation = {
    TensorflowRelation(parameters)(sqlContext.sparkSession)
  }
}
