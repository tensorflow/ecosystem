# Hadoop MapReduce InputFormat/OutputFormat for TFRecords

This directory contains a [Apache Hadoop](http://hadoop.apache.org/) MapReduce
InputFormat/OutputFormat implementation for TensorFlow's TFRecords format.
This can also be used with [Apache Spark](http://spark.apache.org/).

## Prerequisites

1. [protoc 3.3.0](https://developers.google.com/protocol-buffers/)
installed.

2. [Apache Maven](https://maven.apache.org/)

3. Download and compile protos from [TensorFlow](https://github.com/tensorflow/tensorflow).

4. Tested with Hadoop 2.6.0. Patches are welcome if there are incompatibilities
   with your Hadoop version.

## Build and install

1. Compile TensorFlow Example protos

    ```sh
    # Suppose $TF_SRC_ROOT is the source code root of TensorFlow project
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

    After installation (or deployment), the package can be used with the following dependency:

    ```xml
    <dependency>
      <groupId>org.tensorflow</groupId>
      <artifactId>tensorflow-hadoop</artifactId>
      <version>1.0-SNAPSHOT</version>
    </dependency>
    ```

    Alternatively, use the shaded version of the package in case of incompatibilities with the version
    of the protobuf library in your application. For example, Apache Spark uses an older version of the
    protobuf library which can cause conflicts. The shaded package can be used as follows:

    ```xml
    <dependency>
      <groupId>org.tensorflow</groupId>
      <artifactId>tensorflow-hadoop</artifactId>
      <version>1.0-SNAPSHOT</version>
      <classifier>shaded-protobuf</classifier>
    </dependency>
    ```

## Use with MapReduce
The Hadoop MapReduce example can be found [here](src/main/java/org/tensorflow/hadoop/example/TFRecordFileMRExample.java).

## Use with Spark
Spark supports reading/writing files with Hadoop InputFormat/OutputFormat. Use the shaded version of the
package to avoid conflicts with the protobuf version included in Spark.

The following command demonstrates how to use the package with spark-shell:

```bash
$spark-shell --master local --jars target/tensorflow-hadoop-1.0-SNAPSHOT-shaded-protobuf.jar
```


The following code snippet demonstrates the usage.

```scala
import org.tensorflow.hadoop.shaded.protobuf.ByteString
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
