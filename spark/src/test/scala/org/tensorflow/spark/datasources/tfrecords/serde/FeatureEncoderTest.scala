/**
 * Copyright (c) 2016 Intel Corporation 
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

import scala.collection.JavaConverters._

class FeatureEncoderTest extends WordSpec with Matchers {

  "Int64List feature encoder" should {
    "Encode inputs to Int64List" in {
      val intFeature = Int64ListFeatureEncoder.encode(5)
      val longFeature = Int64ListFeatureEncoder.encode(10L)
      val longListFeature = Int64ListFeatureEncoder.encode(Seq(3L,5L,6L))

      intFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(5L))
      longFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(10L))
      longListFeature.getInt64List.getValueList.asScala.toSeq should equal (Seq(3L, 5L, 6L))
    }

    "Throw an exception when inputs contain null" in {
      intercept[Exception] {
        Int64ListFeatureEncoder.encode(null)
      }
      intercept[Exception] {
        Int64ListFeatureEncoder.encode(Seq(3,null,6))
      }
    }

    "Throw an exception for non-numeric inputs" in {
      intercept[Exception] {
        Int64ListFeatureEncoder.encode("bad-input")
      }
    }
  }

  "FloatList feature encoder" should {
    "Encode inputs to FloatList" in {
      val intFeature = FloatListFeatureEncoder.encode(5)
      val longFeature = FloatListFeatureEncoder.encode(10L)
      val floatFeature = FloatListFeatureEncoder.encode(2.5F)
      val doubleFeature = FloatListFeatureEncoder.encode(14.6)
      val floatListFeature = FloatListFeatureEncoder.encode(Seq(1.5F,6.8F,-3.2F))

      intFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(5F))
      longFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(10F))
      floatFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(2.5F))
      doubleFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(14.6F))
      floatListFeature.getFloatList.getValueList.asScala.toSeq should equal (Seq(1.5F,6.8F,-3.2F))
    }

    "Throw an exception when inputs contain null" in {
      intercept[Exception] {
        FloatListFeatureEncoder.encode(null)
      }
      intercept[Exception] {
        FloatListFeatureEncoder.encode(Seq(3,null,6))
      }
    }

    "Throw an exception for non-numeric inputs" in {
      intercept[Exception] {
        FloatListFeatureEncoder.encode("bad-input")
      }
    }
  }

  "ByteList feature encoder" should {
    "Encode inputs to ByteList" in {
      val longFeature = BytesListFeatureEncoder.encode(10L)
      val longListFeature = BytesListFeatureEncoder.encode(Seq(3L,5L,6L))
      val strFeature = BytesListFeatureEncoder.encode("str-input")

      longFeature.getBytesList.toByteString.toStringUtf8.trim should equal ("10")
      longListFeature.getBytesList.toByteString.toStringUtf8.trim should equal ("List(3, 5, 6)")
      strFeature.getBytesList.toByteString.toStringUtf8.trim should equal ("str-input")
    }

    "Throw an exception when inputs contain null" in {
      intercept[Exception] {
        BytesListFeatureEncoder.encode(null)
      }
    }
  }
}
