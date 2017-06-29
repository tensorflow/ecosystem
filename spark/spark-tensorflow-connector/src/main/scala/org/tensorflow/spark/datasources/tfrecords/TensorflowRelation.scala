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
package org.tensorflow.spark.datasources.tfrecords

import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.sources.{BaseRelation, TableScan}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import org.tensorflow.example.{SequenceExample, Example}
import org.tensorflow.hadoop.io.TFRecordFileInputFormat
import org.tensorflow.spark.datasources.tfrecords.serde.DefaultTfRecordRowDecoder


case class TensorflowRelation(options: Map[String, String], customSchema: Option[StructType]=None)
                             (@transient val session: SparkSession) extends BaseRelation with TableScan {

  //Import TFRecords as DataFrame happens here
  lazy val (tfRdd, tfSchema) = {
    val rdd = session.sparkContext.newAPIHadoopFile(options("path"), classOf[TFRecordFileInputFormat], classOf[BytesWritable], classOf[NullWritable])

    val recordType = options.getOrElse("recordType", "Example")

    recordType match {
      case "Example" =>
        val exampleRdd = rdd.map{case (bytesWritable, nullWritable) =>
          Example.parseFrom(bytesWritable.getBytes)
        }
        val finalSchema = customSchema.getOrElse(TensorFlowInferSchema(exampleRdd))
        val rowRdd = exampleRdd.map(example => DefaultTfRecordRowDecoder.decodeExample(example, finalSchema))
        (rowRdd, finalSchema)
      case "SequenceExample" =>
        val sequenceExampleRdd = rdd.map{case (bytesWritable, nullWritable) =>
          SequenceExample.parseFrom(bytesWritable.getBytes)
        }
        val finalSchema = customSchema.getOrElse(TensorFlowInferSchema(sequenceExampleRdd))
        val rowRdd = sequenceExampleRdd.map(example => DefaultTfRecordRowDecoder.decodeSequenceExample(example, finalSchema))
        (rowRdd, finalSchema)
      case _ =>
        throw new IllegalArgumentException(s"Unsupported recordType ${recordType}: recordType can be Example or SequenceExample")
    }
  }

  override def sqlContext: SQLContext = session.sqlContext

  override def schema: StructType = tfSchema

  override def buildScan(): RDD[Row] = tfRdd
}

