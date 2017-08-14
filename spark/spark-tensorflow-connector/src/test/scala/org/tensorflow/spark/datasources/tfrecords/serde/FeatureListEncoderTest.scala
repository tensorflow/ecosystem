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

class FeatureListEncoderTest extends WordSpec with Matchers {

  "Int64 feature list encoder" should {

    "Encode inputs to feature list of Int64" in {
      val longListOfLists = Seq(Seq(3L,5L,Int.MaxValue+6L), Seq(-1L,-6L))
      val longFeatureList = Int64FeatureListEncoder.encode(longListOfLists)

      longFeatureList.getFeatureList.asScala.map(_.getInt64List.getValueList.asScala.toSeq) should equal (longListOfLists)
    }

    "Encode empty array to empty feature list" in {
      val longFeatureList = Int64FeatureListEncoder.encode(Seq.empty[Seq[Long]])
      assert(longFeatureList.getFeatureList.size() === 0)
    }
  }

  "Float feature list encoder" should {

    "Encode inputs to feature list of Float" in {
      val floatListOfLists = Seq(Seq(-2.67F, 1.5F, 0F), Seq(-1.4F,-6F))
      val floatFeatureList = FloatFeatureListEncoder.encode(floatListOfLists)

      assert(floatFeatureList.getFeatureList.asScala.map(_.getFloatList.getValueList.asScala.map(_.toFloat).toSeq) ~== floatListOfLists)
    }

    "Encode empty array to empty feature list" in {
      val floatFeatureList = FloatFeatureListEncoder.encode(Seq.empty[Seq[Float]])
      assert(floatFeatureList.getFeatureList.size() === 0)
    }
  }

  "Bytes feature list encoder" should {

    "Encode inputs to feature list of bytes" in {
      val bytesListOfLists = Seq(Seq("alice".getBytes, "bob".getBytes), Seq("charles".getBytes))
      val bytesFeatureList = BytesFeatureListEncoder.encode(bytesListOfLists)

      assert(bytesFeatureList.getFeatureList.asScala.map(_.getBytesList.getValueList.asScala.toSeq.map(_.toByteArray.deep)) === bytesListOfLists.map(_.map(_.deep)))
    }

    "Encode empty array to empty feature list" in {
      val bytesFeatureList = BytesFeatureListEncoder.encode(Seq.empty[Seq[Array[Byte]]])
      assert(bytesFeatureList.getFeatureList.size() === 0)
    }
  }
}
