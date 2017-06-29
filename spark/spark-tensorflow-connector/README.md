# spark-tensorflow-connector

This repo contains a library for loading and storing TensorFlow records with [Apache Spark](http://spark.apache.org/).
The library implements data import from the standard TensorFlow record format ([TFRecords](https://www.tensorflow.org/how_tos/reading_data/)) into Spark SQL DataFrames, and data export from DataFrames to TensorFlow records.

## What's new

This is the initial release of the `spark-tensorflow-connector` repo.

## Known issues

None.

## Prerequisites

1. [Apache Spark 2.0 (or later)](http://spark.apache.org/)

2. [Apache Maven](https://maven.apache.org/)

3. [TensorFlow Hadoop](../../hadoop) - Provided as Maven dependency. You can also build the latest version as described [here.](../../hadoop)

## Building the library
You can build the library using both Maven and SBT build tools

#### Maven
Build the library using Maven(3.3) as shown below

```sh
mvn clean install
```

To use a different version of TensorFlow Hadoop, use:
```sh
mvn clean install -Dtensorflow.hadoop.version=1.0-SNAPSHOT
```

#### SBT 
Build the library using SBT(0.13.13) as show below
```sh
sbt clean assembly
```

## Using Spark Shell
Run this library in Spark using the `--jars` command line option in `spark-shell` or `spark-submit`. For example:

Maven Jars
```sh
$SPARK_HOME/bin/spark-shell --jars target/spark-tensorflow-connector-1.0-SNAPSHOT.jar,target/lib/tensorflow-hadoop-1.0-06262017-SNAPSHOT-shaded-protobuf.jar
```

SBT Jars
```sh
$SPARK_HOME/bin/spark-shell --jars target/scala-2.11/spark-tensorflow-connector-assembly-1.0.0.jar
```

## Features
This library allows reading TensorFlow records in local or distributed filesystem as [Spark DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html).
When reading TensorFlow records into Spark DataFrame, the API accepts several options:
* `load`: input path to TensorFlow records. Similar to Spark can accept standard Hadoop globbing expressions.
* `schema`: schema of TensorFlow records. Optional schema defined using Spark StructType. If not provided, the schema is inferred from TensorFlow records.
* `recordType`: input format of TensorFlow records. By default it is Example. Possible values are:
  * `Example`: TensorFlow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records
  * `SequenceExample`: TensorFlow [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records

When writing Spark DataFrame to TensorFlow records, the API accepts several options:
* `save`: output path to TensorFlow records. Output path to TensorFlow records on local or distributed filesystem.
* `recordType`: output format of TensorFlow records. By default it is Example. Possible values are:
  * `Example`: TensorFlow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records
  * `SequenceExample`: TensorFlow [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records

## Schema inference
This library supports automatic schema inference when reading TensorFlow records into Spark DataFrames.
Schema inference is expensive since it requires an extra pass through the data.

The schema inference rules are described in the table below:

| TFRecordType             | Feature Type  | Inferred Spark Data Type  |
| ------------------------ |:--------------|:--------------------------|
| Example, SequenceExample | Int64List     | LongType if all lists have length=1, else ArrayType(LongType) |
| Example, SequenceExample | FloatList     | FloatType if all lists have length=1, else ArrayType(FloatType) |
| Example, SequenceExample | BytesList     | StringType if all lists have length=1, else ArrayType(StringType) |
| SequenceExample          | FeatureList of Int64List | ArrayType(ArrayType(LongType)) |
| SequenceExample          | FeatureList of FloatList | ArrayType(ArrayType(FloatType)) |
| SequenceExample          | FeatureList of BytesList | ArrayType(ArrayType(StringType)) |

## Usage Examples

The following code snippet demonstrates usage on test data.

```scala
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._

val path = "test-output.tfrecord"
val testRows: Array[Row] = Array(
new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1")),
new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2")))
val schema = StructType(List(StructField("id", IntegerType), 
                             StructField("IntegerTypeLabel", IntegerType),
                             StructField("LongTypeLabel", LongType),
                             StructField("FloatTypeLabel", FloatType),
                             StructField("DoubleTypeLabel", DoubleType),
                             StructField("VectorLabel", ArrayType(DoubleType, true)),
                             StructField("name", StringType)))
                             
val rdd = spark.sparkContext.parallelize(testRows)

//Save DataFrame as TFRecords
val df: DataFrame = spark.createDataFrame(rdd, schema)
df.write.format("tfrecords").option("recordType", "Example").save(path)

//Read TFRecords into DataFrame.
//The DataFrame schema is inferred from the TFRecords if no custom schema is provided.
val importedDf1: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").load(path)
importedDf1.show()

//Read TFRecords into DataFrame using custom schema
val importedDf2: DataFrame = spark.read.format("tfrecords").schema(schema).load(path)
importedDf2.show()
```

#### Loading YouTube-8M dataset to Spark
Here's how to import the [YouTube-8M](https://research.google.com/youtube8m/) dataset into a Spark DataFrame.

```sh
curl http://us.data.yt8m.org/1/video_level/train/train-0.tfrecord > /tmp/video_level-train-0.tfrecord
curl http://us.data.yt8m.org/1/frame_level/train/train-0.tfrecord > /tmp/frame_level-train-0.tfrecord
```

```scala
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._

//Import Video-level Example dataset into DataFrame
val videoSchema = StructType(List(StructField("video_id", StringType),
                             StructField("labels", ArrayType(IntegerType, true)),
                             StructField("mean_rgb", ArrayType(FloatType, true)),
                             StructField("mean_audio", ArrayType(FloatType, true))))
val videoDf: DataFrame = spark.read.format("tfrecords").schema(videoSchema).option("recordType", "Example").load("file:///tmp/video_level-train-0.tfrecord")
videoDf.show()
videoDf.write.format("tfrecords").option("recordType", "Example").save("youtube-8m-video.tfrecord")
val importedDf1: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").schema(videoSchema).load("youtube-8m-video.tfrecords")
importedDf1.show()

//Import Frame-level SequenceExample dataset into DataFrame
val frameSchema = StructType(List(StructField("video_id", StringType),
                             StructField("labels", ArrayType(IntegerType, true)),
                             StructField("rgb", ArrayType(ArrayType(StringType, true),true)),
                             StructField("audio", ArrayType(ArrayType(StringType, true),true))))
val frameDf: DataFrame = spark.read.format("tfrecords").schema(frameSchema).option("recordType", "SequenceExample").load("file:///tmp/frame_level-train-0.tfrecord")
frameDf.show()
frameDf.write.format("tfrecords").option("recordType", "SequenceExample").save("youtube-8m-frame.tfrecord")
val importedDf2: DataFrame = spark.read.format("tfrecords").option("recordType", "SequenceExample").schema(frameSchema).load("youtube-8m-frame.tfrecords")
importedDf2.show()
```