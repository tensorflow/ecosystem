/**
 *  Copyright (c) 2016 Intel Corporation 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.tensorflow.spark.datasources.tfrecords.serde

import org.scalatest.{Matchers, WordSpec}
import org.tensorflow.example.{BytesList, Feature, FloatList, Int64List}
import org.tensorflow.hadoop.shaded.protobuf.ByteString

class FeatureDecoderTest extends WordSpec with Matchers {

  "Int Feature decoder" should {

    "Decode Feature to Int" in {
      val int64List = Int64List.newBuilder().addValue(4).build()
      val intFeature = Feature.newBuilder().setInt64List(int64List).build()
      IntFeatureDecoder.decode(intFeature) should equal(4)
    }

    "Throw an exception if length of feature array exceeds 1" in {
      intercept[Exception] {
        val int64List = Int64List.newBuilder().addValue(4).addValue(7).build()
        val intFeature = Feature.newBuilder().setInt64List(int64List).build()
        IntFeatureDecoder.decode(intFeature)
      }
    }

    "Throw an exception if feature is not an Int64List" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(4).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        IntFeatureDecoder.decode(floatFeature)
      }
    }
  }

  "Int List Feature decoder" should {

    "Decode Feature to Int List" in {
      val int64List = Int64List.newBuilder().addValue(3).addValue(9).build()
      val intFeature = Feature.newBuilder().setInt64List(int64List).build()
      IntListFeatureDecoder.decode(intFeature) should equal(Seq(3,9))
    }

    "Throw an exception if feature is not an Int64List" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(4).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        IntListFeatureDecoder.decode(floatFeature)
      }
    }
  }

  "Long Feature decoder" should {

    "Decode Feature to Long" in {
      val int64List = Int64List.newBuilder().addValue(5L).build()
      val intFeature = Feature.newBuilder().setInt64List(int64List).build()
      LongFeatureDecoder.decode(intFeature) should equal(5L)
    }

    "Throw an exception if length of feature array exceeds 1" in {
      intercept[Exception] {
        val int64List = Int64List.newBuilder().addValue(4L).addValue(10L).build()
        val intFeature = Feature.newBuilder().setInt64List(int64List).build()
        LongFeatureDecoder.decode(intFeature)
      }
    }

    "Throw an exception if feature is not an Int64List" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(4).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        LongFeatureDecoder.decode(floatFeature)
      }
    }
  }

  "Long List Feature decoder" should {

    "Decode Feature to Long List" in {
      val int64List = Int64List.newBuilder().addValue(3L).addValue(Int.MaxValue+10L).build()
      val intFeature = Feature.newBuilder().setInt64List(int64List).build()
      LongListFeatureDecoder.decode(intFeature) should equal(Seq(3L,Int.MaxValue+10L))
    }

    "Throw an exception if feature is not an Int64List" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(4).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        LongListFeatureDecoder.decode(floatFeature)
      }
    }
  }

  "Float Feature decoder" should {

    "Decode Feature to Float" in {
      val floatList = FloatList.newBuilder().addValue(2.5F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      FloatFeatureDecoder.decode(floatFeature) should equal(2.5F)
    }

    "Throw an exception if length of feature array exceeds 1" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(1.5F).addValue(3.33F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        FloatFeatureDecoder.decode(floatFeature)
      }
    }

    "Throw an exception if feature is not a FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        FloatFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "Float List Feature decoder" should {

    "Decode Feature to Float List" in {
      val floatList = FloatList.newBuilder().addValue(2.5F).addValue(4.3F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      FloatListFeatureDecoder.decode(floatFeature) should equal(Seq(2.5F, 4.3F))
    }

    "Throw an exception if feature is not a FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        FloatListFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "Double Feature decoder" should {

    "Decode Feature to Double" in {
      val floatList = FloatList.newBuilder().addValue(2.5F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      DoubleFeatureDecoder.decode(floatFeature) should equal(2.5d)
    }

    "Throw an exception if length of feature array exceeds 1" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(1.5F).addValue(3.33F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        DoubleFeatureDecoder.decode(floatFeature)
      }
    }

    "Throw an exception if feature is not a FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        DoubleFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "Double List Feature decoder" should {

    "Decode Feature to Double List" in {
      val floatList = FloatList.newBuilder().addValue(2.5F).addValue(4.0F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      DoubleListFeatureDecoder.decode(floatFeature) should equal(Seq(2.5d, 4.0d))
    }

    "Throw an exception if feature is not a DoubleList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        FloatListFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "Bytes List Feature decoder" should {

    "Decode Feature to Bytes List" in {
      val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
      val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
      StringFeatureDecoder.decode(bytesFeature) should equal("str-input")
    }

    "Throw an exception if feature is not a BytesList" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(2.5F).addValue(4.0F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        StringFeatureDecoder.decode(floatFeature)
      }
    }
  }
}

