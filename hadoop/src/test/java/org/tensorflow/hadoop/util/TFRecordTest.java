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

package org.tensorflow.hadoop.util;

import java.io.*;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class TFRecordTest {
  @Test
  public void testTFRecord() throws IOException {
    int count = 1000;
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    TFRecordWriter writer = new TFRecordWriter(new DataOutputStream(baos));
    for (int i = 0; i < count; ++i) {
      writer.write((Integer.toString(i)).getBytes());
    }
    baos.close();

    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    TFRecordReader reader = new TFRecordReader(new DataInputStream(bais), true);
    for (int i = 0; i < count; ++i) {
      assertEquals(Integer.toString(i), new String(reader.read()));
    }
    assertNull(reader.read()); // EOF
  }
}
