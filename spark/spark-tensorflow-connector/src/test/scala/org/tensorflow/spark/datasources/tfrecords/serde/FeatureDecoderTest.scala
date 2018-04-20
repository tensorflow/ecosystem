/**
 *  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
import com.google.protobuf.ByteString
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._

class FeatureDecoderTest extends WordSpec with Matchers {
  val epsilon = 1E-6

  "Int Feature decoder" should {

    "Decode Feature to Int" in {
      val int64List = Int64List.newBuilder().addValue(4).build()
      val intFeature = Feature.newBuilder().setInt64List(int64List).build()
      assert(IntFeatureDecoder.decode(intFeature) === 4)
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
      assert(IntListFeatureDecoder.decode(intFeature) === Seq(3,9))
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
      assert(LongFeatureDecoder.decode(intFeature) === 5L)
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
      assert(LongListFeatureDecoder.decode(intFeature) === Seq(3L,Int.MaxValue+10L))
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
      assert(FloatFeatureDecoder.decode(floatFeature) === 2.5F +- epsilon.toFloat)
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
      assert(FloatListFeatureDecoder.decode(floatFeature) ~== Seq(2.5F, 4.3F))
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
      assert(DoubleFeatureDecoder.decode(floatFeature) === 2.5d +- epsilon)
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
      assert(DoubleListFeatureDecoder.decode(floatFeature) ~== Seq(2.5d, 4.0d))
    }

    "Throw an exception if feature is not a FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        FloatListFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "Decimal Feature decoder" should {

    "Decode Feature to Decimal" in {
      val floatList = FloatList.newBuilder().addValue(2.55F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      assert(DecimalFeatureDecoder.decode(floatFeature).toDouble === 2.55d +- epsilon)
    }

    "Throw an exception if length of feature array exceeds 1" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(1.5F).addValue(3.33F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        DecimalFeatureDecoder.decode(floatFeature)
      }
    }

    "Throw an exception if feature is not a FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        DecimalFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "Decimal List Feature decoder" should {

    "Decode Feature to Decimal List" in {
      val floatList = FloatList.newBuilder().addValue(2.5F).addValue(4.0F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      assert(DecimalListFeatureDecoder.decode(floatFeature).map(_.toDouble) ~== Seq(2.5d, 4.0d))
    }

    "Throw an exception if feature is not a FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        FloatListFeatureDecoder.decode(bytesFeature)
      }
    }
  }

  "String Feature decoder" should {

    "Decode Feature to String" in {
      val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
      val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
      assert(StringFeatureDecoder.decode(bytesFeature) === "str-input")
    }

    "Throw an exception if length of feature array exceeds 1" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("alice".getBytes))
          .addValue(ByteString.copyFrom("bob".getBytes)).build()
        val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
        StringFeatureDecoder.decode(bytesFeature)
      }
    }

    "Throw an exception if feature is not a BytesList" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(2.5F).addValue(4.0F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        StringFeatureDecoder.decode(floatFeature)
      }
    }
  }

  "String List Feature decoder" should {

    "Decode Feature to String List" in {
      val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("alice".getBytes))
        .addValue(ByteString.copyFrom("bob".getBytes)).build()
      val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
      assert(StringListFeatureDecoder.decode(bytesFeature) === Seq("alice", "bob"))
    }

    "Throw an exception if feature is not a BytesList" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(2.5F).addValue(4.0F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        StringListFeatureDecoder.decode(floatFeature)
      }
    }
  }
}
