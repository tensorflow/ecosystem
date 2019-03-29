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

package org.tensorflow.hadoop.io;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.task.MapContextImpl;
import org.junit.Test;
import org.tensorflow.example.Example;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.Int64List;

import java.io.File;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;

public class TFRecordFileTest {
  private static final int RECORDS = 10000;

  @Test
  public void testZippedInputOutputFormat() throws Exception {
    testInputOutputFormat(true);
  }

  @Test
  public void testInputOutputFormat() throws Exception {
    testInputOutputFormat(false);
  }

  private void testInputOutputFormat(boolean zipped) throws Exception {
    Job job = Job.getInstance(new Configuration());
    TaskAttemptContext context =
            MapReduceTestUtil.createDummyMapTaskAttemptContext(job.getConfiguration());

    Random rand = new Random(1234);
    Map<Long, Long> expectedRecords = new TreeMap<Long, Long>();
    for (int i = 0; i < RECORDS; ++i) {
      long randValue = rand.nextLong();
      expectedRecords.put((long) i, randValue);
    }

    Path dir = new Path(getTmpDirectory().toString(), "tfr-test-zipped-" + zipped);
    writeTFRecords(job,
            context,
            dir,
            expectedRecords,
            zipped);

    Map<Long, Long> records = readTFRecords(job,
            context,
            dir);

    assertEquals(expectedRecords, records);
    deleteDirectory(job, dir);
  }

  @Test
  public void testWriteSmallTfRecords() throws Exception {
    Job job = Job.getInstance(new Configuration());
    TaskAttemptContext context =
            MapReduceTestUtil.createDummyMapTaskAttemptContext(job.getConfiguration());

    Path dir = new Path(getTmpDirectory().toString(), "tfr-test-small");
    writeTFRecords(job, context, dir, getExpectedRecords(), false);

    String fileName = getFileName(job, new Path(getResourcesDirectory().toString(), "tf-records"));

    assertEquals(FileUtils.readFileToString(new File(new File(getResourcesDirectory(), "tf-records"), fileName)),
            FileUtils.readFileToString(new File(dir.toString(), fileName)));

    deleteDirectory(job, dir);
  }

  @Test
  public void testReadSmallTfRecords() throws Exception {
    Job job = Job.getInstance(new Configuration());
    TaskAttemptContext context =
            MapReduceTestUtil.createDummyMapTaskAttemptContext(job.getConfiguration());

    Path dir = new Path(getResourcesDirectory().toString(), "tf-records");
    Map<Long, Long> records = readTFRecords(job, context, dir);

    assertEquals(getExpectedRecords(), records);
  }

  @Test
  public void testReadSmallZippedTfRecords() throws Exception {
    Job job = Job.getInstance(new Configuration());
    TaskAttemptContext context =
            MapReduceTestUtil.createDummyMapTaskAttemptContext(job.getConfiguration());

    Path dir = new Path(getResourcesDirectory().toString(), "zipped-tf-records");
    Map<Long, Long> records = readTFRecords(job, context, dir);

    assertEquals(getExpectedRecords(), records);
  }

  private void writeTFRecords(Job job,
                              TaskAttemptContext context,
                              Path dir,
                              Map<Long, Long> records,
                              boolean zipped) throws Exception {
    TFRecordFileOutputFormat.setOutputPath(job, dir);

    if (zipped) {
      TFRecordFileOutputFormat.setCompressOutput(job, true);
      TFRecordFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
    }

    OutputFormat<BytesWritable, NullWritable> outputFormat =
            new TFRecordFileOutputFormat();
    OutputCommitter committer = outputFormat.getOutputCommitter(context);
    committer.setupJob(job);
    RecordWriter<BytesWritable, NullWritable> writer = outputFormat.
            getRecordWriter(context);

    try {
      for (Map.Entry<Long, Long> entry : records.entrySet()) {
        Int64List data = Int64List.newBuilder().addValue(entry.getKey()).addValue(entry.getValue()).build();
        Feature feature = Feature.newBuilder().setInt64List(data).build();
        Features features = Features.newBuilder().putFeature("data", feature).build();
        Example example = Example.newBuilder().setFeatures(features).build();
        BytesWritable key = new BytesWritable(example.toByteArray());
        writer.write(key, NullWritable.get());
      }
    } finally {
      writer.close(context);
    }
    committer.commitTask(context);
    committer.commitJob(job);
  }

  private Map<Long, Long> readTFRecords(Job job, TaskAttemptContext context, Path dir) throws Exception {
    Map<Long, Long> records = new TreeMap<Long, Long>();
    TFRecordFileInputFormat.setInputPaths(job, dir);
    InputFormat<BytesWritable, NullWritable> inputFormat = new TFRecordFileInputFormat();
    for (InputSplit split : inputFormat.getSplits(job)) {
      RecordReader<BytesWritable, NullWritable> reader =
              inputFormat.createRecordReader(split, context);
      MapContext<BytesWritable, NullWritable, BytesWritable, NullWritable> mcontext =
              new MapContextImpl<BytesWritable, NullWritable, BytesWritable, NullWritable>
                      (job.getConfiguration(), context.getTaskAttemptID(), reader, null, null,
                              MapReduceTestUtil.createDummyReporter(),
                              split);
      reader.initialize(split, mcontext);
      try {
        while (reader.nextKeyValue()) {
          BytesWritable bytes = reader.getCurrentKey();
          Example example = Example.parseFrom(bytes.getBytes());
          Int64List data = example.getFeatures().getFeatureMap().get("data").getInt64List();
          Long key = data.getValue(0);
          Long value = data.getValue(1);
          records.put(key, value);
        }
      } finally {
        reader.close();
      }
    }
    return records;
  }

  private Map<Long, Long> getExpectedRecords() {
    Map<Long, Long> records = new TreeMap<Long, Long>();
    records.put(0L, -6519408338692630574L);
    records.put(1L, -897291810407650440L);
    records.put(2L, -2627029093267243214L);
    records.put(3L, 8452912497529882771L);
    records.put(4L, 6197228047171027195L);
    return records;
  }

  private File getResourcesDirectory() {
    return new File("src/test/resources");
  }

  private File getTmpDirectory() {
    return new File(System.getProperty("test.build.data", "/tmp"));
  }

  private void deleteDirectory(Job job, Path dir) throws Exception {
    FileSystem fs = dir.getFileSystem(job.getConfiguration());
    fs.delete(dir, true);
  }

  private String getFileName(Job job, Path dir) throws Exception {
    FileSystem fs = dir.getFileSystem(job.getConfiguration());
    return fs.listFiles(dir, false).next().getPath().getName();
  }
}
