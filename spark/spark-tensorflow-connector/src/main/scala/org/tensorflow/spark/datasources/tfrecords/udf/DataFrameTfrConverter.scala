package org.tensorflow.spark.datasources.tfrecords.udf

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.tensorflow.spark.datasources.tfrecords.serde.DefaultTfRecordRowEncoder

object DataFrameTfrConverter {
  def getRowToTFRecordExampleUdf: UserDefinedFunction = udf(rowToTFRecordExampleUdf _ )

  private def rowToTFRecordExampleUdf(row: Row): Array[Byte] = {
    DefaultTfRecordRowEncoder.encodeExample(row).toByteArray
  }

  def getRowToTFRecordSequenceExampleUdf: UserDefinedFunction = udf(rowToTFRecordSequenceExampleUdf _ )

  private def rowToTFRecordSequenceExampleUdf(row: Row): Array[Byte] = {
    DefaultTfRecordRowEncoder.encodeSequenceExample(row).toByteArray
  }
}
