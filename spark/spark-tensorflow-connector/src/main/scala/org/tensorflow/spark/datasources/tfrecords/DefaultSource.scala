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

import java.io._
import java.nio.file.Paths

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.StructType
import org.tensorflow.hadoop.io.TFRecordFileOutputFormat
import org.tensorflow.hadoop.util._
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
    val codec = parameters.getOrElse("codec", "")

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

    parameters.getOrElse("writeLocality", "distributed") match {
      case "distributed" =>
        saveDistributed(features, path, sqlContext, mode, codec)
      case "local" =>
        saveLocal(features, path, mode, codec)
      case s: String =>
        throw new IllegalArgumentException(
          s"Expected 'distributed' or 'local', got $s")
    }
    TensorflowRelation(parameters)(sqlContext.sparkSession)
  }

  private def save(sqlContext: SQLContext, features: RDD[(BytesWritable, NullWritable)], path: String, codec: String) = {
    val hadoopConf = new Configuration(sqlContext.sparkContext.hadoopConfiguration)
    if (!codec.isEmpty) {
      hadoopConf.set("mapreduce.output.fileoutputformat.compress", "true")
      hadoopConf.set("mapreduce.output.fileoutputformat.compress.codec", codec)
    }
    features.saveAsNewAPIHadoopFile(
      path,
      classOf[NullWritable],
      classOf[BytesWritable],
      classOf[TFRecordFileOutputFormat],
      hadoopConf
    )
  }

  private def saveDistributed(
      features: RDD[(BytesWritable, NullWritable)],
      path: String,
      sqlContext: SQLContext,
      mode: SaveMode,
      codec: String): Unit = {
    val hadoopConf = sqlContext.sparkContext.hadoopConfiguration
    val outputPath = new Path(path)
    val fs = outputPath.getFileSystem(hadoopConf)
    val qualifiedOutputPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)

    val pathExists = fs.exists(qualifiedOutputPath)

    mode match {
      case SaveMode.Overwrite =>
        fs.delete(qualifiedOutputPath, true)
        save(sqlContext, features, path, codec)

      case SaveMode.Append =>
        throw new IllegalArgumentException("Append mode is not supported")

      case SaveMode.ErrorIfExists =>
        if (pathExists)
          throw new IllegalStateException(
            s"Path $path already exists. SaveMode: ErrorIfExists.")
        save(sqlContext, features, path, codec)

      case SaveMode.Ignore =>
        // With `SaveMode.Ignore` mode, if data already exists, the save operation is expected
        // to not save the contents of the DataFrame and to not change the existing data.
        // Therefore, it is okay to do nothing here and then just return the relation below.
        if (pathExists == false)
          save(sqlContext, features, path, codec)
    }
  }

  private def saveLocal(
      features: RDD[(BytesWritable, NullWritable)],
      localPath: String,
      mode: SaveMode,
      codec: String): Unit = {
    val cleanedPath = Paths.get(localPath).toAbsolutePath.toString
    if (!codec.isEmpty) {
      throw new IllegalArgumentException("codec can not be used in local write mode")
    }
    if (mode == SaveMode.Append) {
      throw new IllegalArgumentException("Append mode is not supported in local write mode")
    }
    // Not supported now, but it should be a small fix eventually.
    if (mode == SaveMode.Overwrite) {
      throw new IllegalArgumentException("Overwrite mode is not supported in local write mode")
    }

    val f = DefaultSource.writePartitionLocalFun(localPath, mode)

    // Perform the action.
    features.mapPartitionsWithIndex(f).collect()
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

object DefaultSource {
  // The function run on each worker.
  // Writes the partition to a file and returns the number of records output.
  private def writePartitionLocal(
      index: Int,
      part: Iterator[(BytesWritable, NullWritable)],
      localPath: String,
      mode: SaveMode): Iterator[Int] = {
    val dir = new File(localPath)
    if (dir.exists()) {
      if (mode == SaveMode.ErrorIfExists) {
        throw new IllegalStateException(
          s"LocalPath $localPath already exists. SaveMode: ErrorIfExists.")
      }
      if (mode == SaveMode.Ignore) {
        return Iterator.empty
      }
    }

    // Make the directory if it does not exist
    dir.mkdirs()
    // The path to the partition file.
    val filePath = localPath + s"/part-" + String.format("%05d", new java.lang.Integer(index))
    val fos = new DataOutputStream(new FileOutputStream(filePath))
    var count = 0
    try {
      val tfw = new TFRecordWriter(fos)
      for((bw, _) <- part) {
        tfw.write(bw.getBytes)
        count += 1
      }
    } finally {
      fos.close()
    }
    Iterator(count)
  }

  // Working around the closure variable captures.
  private def writePartitionLocalFun(
      localPath: String,
      mode: SaveMode): (Int, Iterator[(BytesWritable, NullWritable)]) => Iterator[Int] = {
    def mapFun(index: Int, part: Iterator[(BytesWritable, NullWritable)]) = {
      writePartitionLocal(index, part, localPath, mode)
    }
    mapFun
  }

}
