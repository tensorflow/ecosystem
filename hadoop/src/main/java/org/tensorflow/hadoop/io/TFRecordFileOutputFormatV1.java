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

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.util.ReflectionUtils;
import org.tensorflow.hadoop.util.TFRecordWriter;

import java.io.DataOutputStream;
import java.io.IOException;

public class TFRecordFileOutputFormatV1 extends FileOutputFormat<BytesWritable, Writable> {
  @Override
  public RecordWriter<BytesWritable, Writable> getRecordWriter(FileSystem ignored,
                                                               JobConf job, String name,
                                                               Progressable progress) throws IOException {
    boolean isCompressed = getCompressOutput(job);
    CompressionCodec codec = null;
    String extension = "";
    if (isCompressed) {
      Class<? extends CompressionCodec> codecClass = getOutputCompressorClass(job, GzipCodec.class);
      codec = ReflectionUtils.newInstance(codecClass, job);
      extension = codec.getDefaultExtension();
    }
    Path file = FileOutputFormat.getTaskOutputPath(job, name + extension);
    FileSystem fs = file.getFileSystem(job);
    int bufferSize = TFRecordIOConf.getBufferSize(job);
    FSDataOutputStream fsDataOutputStream = fs.create(file, true, bufferSize);
    final DataOutputStream fsdos = isCompressed ?
      new DataOutputStream(codec.createOutputStream(fsDataOutputStream)) : fsDataOutputStream;
    final TFRecordWriter writer = new TFRecordWriter(fsdos);
    return new RecordWriter<BytesWritable, Writable>() {
      @Override
      public void write(BytesWritable key, Writable value)
        throws IOException {
        writer.write(key.getBytes(), 0, key.getLength());
      }

      @Override
      public void close(Reporter reporter)
        throws IOException {
        fsdos.close();
      }
    };
  }
}
