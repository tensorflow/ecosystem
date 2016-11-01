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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.task.MapContextImpl;
import org.junit.Test;
import org.tensorflow.example.Example;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.Int64List;

import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;

public class TFRecordFileTest {
  private static final int RECORDS = 10000;

  @Test
  public void testInputOutputFormat() throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf);

    Path outdir = new Path(System.getProperty("test.build.data", "/tmp"), "tfr-test");

    TFRecordFileOutputFormat.setOutputPath(job, outdir);

    TaskAttemptContext context =
        MapReduceTestUtil.createDummyMapTaskAttemptContext(job.getConfiguration());
    OutputFormat<BytesWritable, NullWritable> outputFormat =
        new TFRecordFileOutputFormat();
    OutputCommitter committer = outputFormat.getOutputCommitter(context);
    committer.setupJob(job);
    RecordWriter<BytesWritable, NullWritable> writer = outputFormat.
        getRecordWriter(context);

    // Write Example with random numbers
    Random rand = new Random();
    Map<Long, Long> records = new TreeMap<Long, Long>();
    try {
      for (int i = 0; i < RECORDS; ++i) {
        long randValue = rand.nextLong();
        records.put((long) i, randValue);
        Int64List data = Int64List.newBuilder().addValue(i).addValue(randValue).build();
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

    // Read and compare
    TFRecordFileInputFormat.setInputPaths(job, outdir);
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
          assertEquals(records.get(key), value);
          records.remove(key);
        }
      } finally {
        reader.close();
      }
    }
    assertEquals(0, records.size());
  }
}
