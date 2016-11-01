/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.hadoop.example;

import com.google.protobuf.ByteString;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.tensorflow.example.*;
import org.tensorflow.hadoop.io.TFRecordFileInputFormat;
import org.tensorflow.hadoop.io.TFRecordFileOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.Map;

public class TFRecordFileMRExample {
  /**
   * Convert from text file to TFRecord file. Each line is converted into two dummy features: the
   * content of each line and the starting offset of each line.
   */
  static class ToTFRecordMapper extends Mapper<LongWritable, Text, BytesWritable, NullWritable> {
    ToTFRecordMapper(){}

    @Override protected void map(LongWritable key, Text value,
        Context context) throws IOException, InterruptedException {
      Int64List int64List = Int64List.newBuilder().addValue(key.get()).build();
      Feature offset = Feature.newBuilder().setInt64List(int64List).build();

      ByteString byteString = ByteString.copyFrom(value.copyBytes());
      BytesList bytesList = BytesList.newBuilder().addValue(byteString).build();
      Feature text = Feature.newBuilder().setBytesList(bytesList).build();

      Features features = Features.newBuilder()
          .putFeature("offset", offset)
          .putFeature("text", text)
          .build();
      Example example = Example.newBuilder().setFeatures(features).build();
      context.write(new BytesWritable(example.toByteArray()), NullWritable.get());
    }
  }

  /**
   * Convert from previous TFRecord file to text file.
   */
  static class FromTFRecordMapper extends Mapper<BytesWritable, NullWritable, NullWritable, Text> {
    FromTFRecordMapper(){}

    @Override protected void map(BytesWritable key, NullWritable value,
        Context context) throws IOException, InterruptedException {
      Example example = Example.parseFrom(key.getBytes());
      Map<String, Feature> featureMap = example.getFeatures().getFeatureMap();
      byte[] text = featureMap.get("text").getBytesList().getValue(0).toByteArray();
      context.write(NullWritable.get(), new Text(text));
    }
  }

  public static boolean convert(String jobName,
      Class<? extends Mapper> mapperClass,
      Class<? extends Writable> outputKeyClass,
      Class<? extends Writable> outputValueClass,
      Class<? extends InputFormat> inFormatClass,
      Class<? extends OutputFormat> outFormatClass,
      Path input,
      Path output) throws InterruptedException, IOException, ClassNotFoundException {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, jobName);
    job.setJarByClass(mapperClass);
    job.setMapperClass(mapperClass);
    job.setNumReduceTasks(0);

    job.setInputFormatClass(inFormatClass);
    job.setOutputFormatClass(outFormatClass);
    job.setOutputKeyClass(outputKeyClass);
    job.setOutputValueClass(outputValueClass);

    final FileSystem fs = FileSystem.get(output.toUri(), conf);
    fs.delete(output, true);
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    return job.waitForCompletion(true);
  }

  public static void main(String[] args) throws Exception {
    String testRoot  = "/tmp/tfrecord-file-test";
    if (args.length == 1) {
      testRoot = args[0];
    } else if (args.length > 1) {
      System.out.println("Usage: TFRecordFileMRExample [path]");
    }

    Path testRootPath = new Path(testRoot);
    Path input = new Path(testRootPath, "input.txt");
    Path tfrout = new Path(testRootPath, "output.tfr");
    Path txtout = new Path(testRootPath, "output.txt");

    convert("ToTFR", ToTFRecordMapper.class, BytesWritable.class, NullWritable.class,
        TextInputFormat.class, TFRecordFileOutputFormat.class, input, tfrout);
    convert("FromTFR", FromTFRecordMapper.class, NullWritable.class, Text.class,
        TFRecordFileInputFormat.class, TextOutputFormat.class, tfrout, txtout);
  }
}
