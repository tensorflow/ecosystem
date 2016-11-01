# Hadoop MapReduce InputFormat/OutputFormat for TFRecord file

This directory contains [Apache Hadoop](http://hadoop.apache.org/) MapReduce
InputFormat/OutputFormat implementation for Tensorflow TFRecord feature files.
This can also be used with [Apache Spark](http://spark.apache.org/).

## Prerequisites

1. [protoc 3.1.0](https://developers.google.com/protocol-buffers/) must be
installed.

2. [Apache Maven](https://maven.apache.org/) must be installed.

3. This is tested with Hadoop 2.6.0, so you'd better have hadoop 2.6.0 or 
compitable YARN/Spark cluster.

## Build and install

1. Compile Tensorflow Example protos

    ```sh
    # Suppose $TF_SRC_ROOT is the source code root of tensorflow project
    protoc --proto_path=$TF_SRC_ROOT --java_out=src/main/java/ $TF_SRC_ROOT/tensorflow/core/example/{example,feature}.proto
    ```

2. Compile the code

    ```sh
    mvn clean package
    ```

3. Optionally install (or deploy) the jars

    ```sh
    mvn install
    ```

    After installed (or deployed), the package can be used with following dependency:

    ```xml
    <dependency>
      <groupId>org.tensorflow</groupId>
      <artifactId>tensorflow-hadoop</artifactId>
      <version>1.0-SNAPSHOT</version>
    </dependency>
    ```

## Use with MapReduce
The Hadoop MapReduce example can be found [here](src/main/java/org/tensorflow/hadoop/example/TFRecordFileMRExample.java).

## Use with Spark
Spark support reading/writing files with Hadoop InputFormat/OutputFormat, the
following code snippet demostrate the usage.

```scala
import com.google.protobuf.ByteString
import org.apache.hadoop.io.{NullWritable, BytesWritable}
import org.apache.spark.{SparkConf, SparkContext}
import org.tensorflow.example.{BytesList, Int64List, Feature, Features, Example}
import org.tensorflow.hadoop.io.TFRecordFileOutputFormat

val inputPath = "path/to/input.txt"
val outputPath = "path/to/output.tfr"

val sparkConf = new SparkConf().setAppName("TFRecord Demo")
val sc = new SparkContext(sparkConf)

var features = sc.textFile(inputPath).map(line => {
  val text = BytesList.newBuilder().addValue(ByteString.copyFrom(line.getBytes)).build()
  val features = Features.newBuilder()
    .putFeature("text", Feature.newBuilder().setBytesList(text).build())
    .build()
  val example = Example.newBuilder()
    .setFeatures(features)
    .build()
  (new BytesWritable(example.toByteArray), NullWritable.get())
})

features.saveAsNewAPIHadoopFile[TFRecordFileOutputFormat](outputPath)
```
