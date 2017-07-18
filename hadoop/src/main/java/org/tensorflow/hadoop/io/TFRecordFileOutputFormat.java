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

import org.tensorflow.hadoop.util.TFRecordWriter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

public class TFRecordFileOutputFormat extends FileOutputFormat<BytesWritable, NullWritable> {
  @Override public RecordWriter<BytesWritable, NullWritable> getRecordWriter(
      TaskAttemptContext context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    Path file = getDefaultWorkFile(context, "");
    FileSystem fs = file.getFileSystem(conf);

    int bufferSize = TFRecordIOConf.getBufferSize(conf);
    final FSDataOutputStream fsdos = fs.create(file, true, bufferSize);
    final TFRecordWriter writer = new TFRecordWriter(fsdos);
    return new RecordWriter<BytesWritable, NullWritable>() {
      @Override public void write(BytesWritable key, NullWritable value)
          throws IOException, InterruptedException {
        writer.write(key.getBytes(), 0, key.getLength());
      }

      @Override public void close(TaskAttemptContext context)
          throws IOException, InterruptedException {
        fsdos.close();
      }
    };
  }
}
