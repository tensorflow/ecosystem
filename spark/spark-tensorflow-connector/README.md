# spark-tensorflow-connector

This repo contains a library for loading and storing TensorFlow records with [Apache Spark](http://spark.apache.org/).
The library implements data import from the standard TensorFlow record format ([TFRecords](https://www.tensorflow.org/how_tos/reading_data/))
into Spark SQL DataFrames, and data export from DataFrames to TensorFlow records.

The library supports both the Scala and PySpark APIs. See [Usage examples](#usage-examples) for sample PySpark and Scala code.

## Breaking changes

* 08/20/2018 - Reverted artifactId back to `org.tensorflow.spark-tensorflow-connector`
* 05/29/2018 - Changed the artifactId from `org.tensorflow.spark-tensorflow-connector` to `org.tensorflow.spark-connector`

## Prerequisites

1. [Apache Spark 2.0 (or later)](http://spark.apache.org/)

2. [Apache Maven](https://maven.apache.org/)

3. [TensorFlow Hadoop](../../hadoop) - Provided as Maven dependency. You can also build the latest version as described [here.](../../hadoop)

## Building the library
Build the library using Maven 3.3.9 or newer as shown below:

```sh
# Build TensorFlow Hadoop
cd ../../hadoop
mvn clean install

# Build Spark TensorFlow connector
cd ../spark/spark-tensorflow-connector
mvn clean install
```

To build the library for a different version of TensorFlow, e.g., 1.5.0, use:
```sh
# Build TensorFlow Hadoop
cd ../../hadoop
mvn versions:set -DnewVersion=1.5.0
mvn clean install

# Build Spark TensorFlow connector
cd ../spark/spark-tensorflow-connector
mvn versions:set -DnewVersion=1.5.0
mvn clean install
```

To build the library for a different version of Apache Spark, e.g., 2.2.1, use:

```
# From spark-tensorflow-connector directory
mvn clean install -Dspark.version=2.2.1
```

After installation (or deployment), the package can be used with the following dependency:

    ```xml
    <dependency>
      <groupId>org.tensorflow</groupId>
      <artifactId>spark-tensorflow-connector_2.11</artifactId>
      <version>1.10.0</version>
    </dependency>
    ```

## Using Spark Shell
Run this library in Spark using the `--jars` command line option in `spark-shell`, `pyspark` or `spark-submit`. For example:

```sh
$SPARK_HOME/bin/spark-shell --jars target/spark-tensorflow-connector_2.11-1.10.0.jar
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
* `codec`: codec for compressinng Tensorflow records. For example, `option("codec", "org.apache.hadoop.io.compress.GzipCodec")` enables gzip
compression. While reading compressed TensorFlow records, `codec` can be inferred automatically, so this option is not required for reading.
* `recordType`: output format of TensorFlow records. By default it is Example. Possible values are:
  * `Example`: TensorFlow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records
  * `SequenceExample`: TensorFlow [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records
* `writeLocality`: determines whether the TensorFlow records are written locally on the workers
or on a distributed file system. Possible values are:
  * `distributed` (default): the dataframe is written using Spark's default file system.
  * `local`: writes the content on the disks of each the Spark workers, in a partitioned manner
  (see details in the paragraph below).

_Local mode write_ each of the workers stores on the local disk a subset of the data.
The subset that is stored on each worker is determined by the partitioning of the Dataframe.
Each of the partitions is coalesced into a single TFRecord file and written on the node where
the partition lives.
This is useful in the context of distributed training, in which each of the workers gets a
subset of the data to work on.
When this mode is activated, the path provided to the writer is interpreted as a base path that is
created on each of the worker nodes, and that will be populated with data from the dataframe. For
 example, the following code:

```scala
myDataFrame.write.format("tfrecords").option("writeLocality", "local").save("/path")
```

will lead to each worker nodes to have the following files:
  - worker1: /path/part-0001.tfrecord, /path/part-0002.tfrecord, ...
  - worker2: /path/part-0042.tfrecord, ...


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

## Supported data types

The supported Spark data types are listed in the table below:

| Type            | Spark DataTypes                          |
| --------------- |:------------------------------------------|
| Scalar          | IntegerType, LongType, FloatType, DoubleType, DecimalType, StringType, BinaryType |
| Array           | VectorType, ArrayType of IntegerType, LongType, FloatType, DoubleType, DecimalType, BinaryType, or StringType |
| Array of Arrays | ArrayType of ArrayType of IntegerType, LongType, FloatType, DoubleType, DecimalType, BinaryType, or StringType |

## Usage Examples

### Python API
Run PySpark with the spark_connector in the jars argument as shown below:

`$SPARK_HOME/bin/pyspark --jars target/spark-tensorflow-connector_2.11-1.10.0.jar`

The following Python code snippet demonstrates usage on test data.

```
from pyspark.sql.types import *

path = "test-output.tfrecord"

fields = [StructField("id", IntegerType()), StructField("IntegerCol", IntegerType()),
          StructField("LongCol", LongType()), StructField("FloatCol", FloatType()),
          StructField("DoubleCol", DoubleType()), StructField("VectorCol", ArrayType(DoubleType(), True)),
          StructField("StringCol", StringType())]
schema = StructType(fields)
test_rows = [[11, 1, 23, 10.0, 14.0, [1.0, 2.0], "r1"], [21, 2, 24, 12.0, 15.0, [2.0, 2.0], "r2"]]
rdd = spark.sparkContext.parallelize(test_rows)
df = spark.createDataFrame(rdd, schema)
df.write.format("tfrecords").option("recordType", "Example").save(path)

df = spark.read.format("tfrecords").option("recordType", "Example").load(path)
df.show()
```

### Scala API
Run Spark shell with the spark_connector in the jars argument as shown below:
```sh
$SPARK_HOME/bin/spark-shell --jars target/spark-tensorflow-connector_2.11-1.10.0.jar
```

The following Scala code snippet demonstrates usage on test data.

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
                             StructField("IntegerCol", IntegerType),
                             StructField("LongCol", LongType),
                             StructField("FloatCol", FloatType),
                             StructField("DoubleCol", DoubleType),
                             StructField("VectorCol", ArrayType(DoubleType, true)),
                             StructField("StringCol", StringType)))
                             
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
mkdir -p /tmp/youtube-8m-frames
pushd /tmp/youtube-8m-frames
curl data.yt8m.org/download.py | shard=1,3844 partition=2/frame/train mirror=us python

mkdir -p /tmp/youtube-8m-videos
cd /tmp/youtube-8m-videos
curl data.yt8m.org/download.py | shard=1,3844 partition=2/video/train mirror=us python
popd

$SPARK_HOME/bin/spark-shell --jars target/spark-tensorflow-connector_2.11-1.10.0.jar
```

```scala
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._

//Import Video-level Example dataset into DataFrame
val videoSchema = StructType(List(StructField("id", StringType),
                             StructField("labels", ArrayType(IntegerType, true)),
                             StructField("mean_rgb", ArrayType(FloatType, true)),
                             StructField("mean_audio", ArrayType(FloatType, true))))
val videoDf: DataFrame = spark.read.format("tfrecords").schema(videoSchema).option("recordType", "Example").load("file:///tmp/youtube-8m-videos/*.tfrecord")
videoDf.show()
videoDf.write.format("tfrecords").option("recordType", "Example").save("youtube-8m-video.tfrecord")
val importedDf1: DataFrame = spark.read.format("tfrecords").option("recordType", "Example").schema(videoSchema).load("youtube-8m-video.tfrecord")
importedDf1.show()

//Import Frame-level SequenceExample dataset into DataFrame
val frameSchema = StructType(List(StructField("id", StringType),
                             StructField("labels", ArrayType(IntegerType, true)),
                             StructField("rgb", ArrayType(ArrayType(BinaryType, true),true)),
                             StructField("audio", ArrayType(ArrayType(BinaryType, true),true))))
val frameDf: DataFrame = spark.read.format("tfrecords").schema(frameSchema).option("recordType", "SequenceExample").load("file:///tmp/youtube-8m-frames/*.tfrecord")
frameDf.show()
frameDf.write.format("tfrecords").option("recordType", "SequenceExample").save("youtube-8m-frame.tfrecord")
val importedDf2: DataFrame = spark.read.format("tfrecords").option("recordType", "SequenceExample").schema(frameSchema).load("youtube-8m-frame.tfrecord")
importedDf2.show()
```
