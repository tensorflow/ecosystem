# Hadoop MapReduce InputFormat/OutputFormat for TFRecords

This directory contains a [Apache Hadoop](http://hadoop.apache.org/) MapReduce
InputFormat/OutputFormat implementation for TensorFlow's TFRecords format.
This can also be used with [Apache Spark](http://spark.apache.org/).

## Prerequisites

1. [Apache Maven](https://maven.apache.org/)

2. Tested with Hadoop 2.6.0. Patches are welcome if there are incompatibilities
   with your Hadoop version.

## Breaking changes

* 08/20/2018 - Reverted artifactId back to `org.tensorflow.tensorflow-hadoop`
* 05/29/2018 - Changed the artifactId from `org.tensorflow.tensorflow-hadoop` to `org.tensorflow.hadoop`

## Build and install

1. Compile the code

    ```sh
    mvn clean package
    ```

    Alternatively, if you would like to build jars for a different version of TensorFlow, e.g., 1.5.0:

    ```sh
   mvn versions:set -DnewVersion=1.5.0
   mvn clean package
    ```

2. Optionally install (or deploy) the jars

    ```sh
    mvn install
    ```

    After installation (or deployment), the package can be used with the following dependency:

    ```xml
    <dependency>
      <groupId>org.tensorflow</groupId>
      <artifactId>tensorflow-hadoop</artifactId>
      <version>1.10.0</version>
    </dependency>
    ```


## Use with MapReduce
The Hadoop MapReduce example can be found [here](src/main/java/org/tensorflow/hadoop/example/TFRecordFileMRExample.java).

## Use with Apache Spark
The [Spark-TensorFlow-Connector](../spark/spark-tensorflow-connector) uses TensorFlow Hadoop to load and save
TensorFlow's TFRecords format using Spark DataFrames.