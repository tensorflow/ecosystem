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
      val intFeature = Int64ListFeatureEncoder.encode(Seq(5))
      val longFeature = Int64ListFeatureEncoder.encode(Seq(10L))
      val longListFeature = Int64ListFeatureEncoder.encode(Seq(3L,5L,6L))

      intFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(5L))
      longFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(10L))
      longListFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(3L, 5L, 6L))
    }

    "Encode null elements as zeros" in {
      val longListFeature = Int64ListFeatureEncoder.encode(Seq(1, null.asInstanceOf[Long]))
      longListFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(1L, 0))
    }
  }

  "FloatList feature encoder" should {
    "Encode inputs to FloatList" in {
      val intFeature = FloatListFeatureEncoder.encode(Seq(5))
      val longFeature = FloatListFeatureEncoder.encode(Seq(10L))
      val floatFeature = FloatListFeatureEncoder.encode(Seq(2.5F))
      val floatListFeature = FloatListFeatureEncoder.encode(Seq(1.5F,6.8F,-3.2F))

      intFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(5F))
      longFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(10F))
      floatFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(2.5F))
      floatListFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(1.5F,6.8F,-3.2F))
    }

    "Encode null elements as zeros" in {
      val floatListFeature = FloatListFeatureEncoder.encode(Seq(6.5F, null.asInstanceOf[Float]))
      assert(floatListFeature.getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== Seq(6.5F, 0F))
    }
  }

  "ByteList feature encoder" should {
    "Encode inputs to ByteList" in {
      val strFeature = BytesListFeatureEncoder.encode(Seq("str-input"))
      val strListFeature = BytesListFeatureEncoder.encode(Seq("alice", "bob"))

      strFeature.getBytesList.getValueList.asScala.map(_.toStringUtf8.trim) should equal (Seq("str-input"))
      strListFeature.getBytesList.getValueList.asScala.map(_.toStringUtf8.trim) should equal (Seq("alice", "bob"))
    }

    "Throw an exception when inputs contain null" in {
      intercept[Exception] {
        BytesListFeatureEncoder.encode(Seq(null, "test"))
      }
    }
  }
}
