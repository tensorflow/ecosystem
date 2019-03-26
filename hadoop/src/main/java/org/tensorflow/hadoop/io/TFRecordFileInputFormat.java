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

import org.apache.hadoop.fs.Seekable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.tensorflow.hadoop.util.TFRecordReader;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.io.InputStream;

public class TFRecordFileInputFormat extends FileInputFormat<BytesWritable, NullWritable> {
  @Override public RecordReader<BytesWritable, NullWritable> createRecordReader(
      InputSplit inputSplit, final TaskAttemptContext context) throws IOException, InterruptedException {

    return new RecordReader<BytesWritable, NullWritable>() {
      private InputStream fsdis;
      private TFRecordReader reader;
      private long length;
      private long begin;
      private byte[] current;

      @Override public void initialize(InputSplit split, TaskAttemptContext context)
          throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        FileSplit fileSplit = (FileSplit) split;
        length = fileSplit.getLength();
        begin = fileSplit.getStart();

        final Path file = fileSplit.getPath();
        CompressionCodecFactory compressionCodecFactory = new CompressionCodecFactory(conf);
        CompressionCodec codec = compressionCodecFactory.getCodec(file);
        FileSystem fs = file.getFileSystem(conf);

        FSDataInputStream fsIn = fs.open(file, TFRecordIOConf.getBufferSize(conf));
        if (codec != null) {
          fsdis = codec.createInputStream(fsIn);
        } else {
          fsdis = fsIn;
        }
        reader = new TFRecordReader(fsdis, TFRecordIOConf.getDoCrc32Check(conf));
      }

      @Override public boolean nextKeyValue() throws IOException, InterruptedException {
        current = reader.read();
        return current != null;
      }

      @Override public BytesWritable getCurrentKey() throws IOException, InterruptedException {
        return new BytesWritable(current);
      }

      @Override public NullWritable getCurrentValue() throws IOException, InterruptedException {
        return NullWritable.get();
      }

      @Override public float getProgress() throws IOException, InterruptedException {
        return (((Seekable)fsdis).getPos() - begin) / (length + 1e-6f);
      }

      @Override public void close() throws IOException {
        fsdis.close();
      }
    };
  }

  @Override
  protected boolean isSplitable(JobContext context, Path file) {
    return false;
  }
}
