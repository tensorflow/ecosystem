/**
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.spark.datasources.tfrecords.serde

import org.scalatest.{Matchers, WordSpec}
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._
import scala.collection.JavaConverters._

class FeatureEncoderTest extends WordSpec with Matchers {

  "Int64List feature encoder" should {
    "Encode inputs to Int64List" in {
      val longFeature = Int64ListFeatureEncoder.encode(Seq(10L))
      val longListFeature = Int64ListFeatureEncoder.encode(Seq(3L,5L,6L))

      assert(longFeature.getInt64List.getValueList.asScala.toSeq === Seq(10L))
      assert(longListFeature.getInt64List.getValueList.asScala.toSeq === Seq(3L, 5L, 6L))
    }

    "Encode empty list to empty feature" in {
      val longListFeature = Int64ListFeatureEncoder.encode(Seq.empty[Long])
      assert(longListFeature.getInt64List.getValueList.size() === 0)
    }
  }

  "FloatList feature encoder" should {
    "Encode inputs to FloatList" in {
      val floatFeature = FloatListFeatureEncoder.encode(Seq(2.5F))
      val floatListFeature = FloatListFeatureEncoder.encode(Seq(1.5F,6.8F,-3.2F))

      assert(floatFeature.getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== Seq(2.5F))
      assert(floatListFeature.getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== Seq(1.5F,6.8F,-3.2F))
    }

    "Encode empty list to empty feature" in {
      val floatListFeature = FloatListFeatureEncoder.encode(Seq.empty[Float])
      assert(floatListFeature.getFloatList.getValueList.size() === 0)
    }
  }

  "ByteList feature encoder" should {
    "Encode inputs to ByteList" in {
      val strFeature = BytesListFeatureEncoder.encode(Seq("str-input"))
      val strListFeature = BytesListFeatureEncoder.encode(Seq("alice", "bob"))

      assert(strFeature.getBytesList.getValueList.asScala.map(_.toStringUtf8) === Seq("str-input"))
      assert(strListFeature.getBytesList.getValueList.asScala.map(_.toStringUtf8) === Seq("alice", "bob"))
    }

    "Encode empty list to empty feature" in {
      val strListFeature = BytesListFeatureEncoder.encode(Seq.empty[String])
      assert(strListFeature.getBytesList.getValueList.size() === 0)
    }
  }
}
