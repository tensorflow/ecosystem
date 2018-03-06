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

import com.google.protobuf.ByteString
import org.tensorflow.spark.datasources.tfrecords.TestingUtils._
import org.scalatest.{Matchers, WordSpec}
import org.tensorflow.example._

class FeatureListDecoderTest extends WordSpec with Matchers{

  "Int FeatureList decoder" should {

    "Decode FeatureList to 2-dimensional integer array" in {
      val int64List1 = Int64List.newBuilder().addValue(1).addValue(3).build()
      val int64List2 = Int64List.newBuilder().addValue(-2).addValue(5).addValue(10).build()
      val feature1 = Feature.newBuilder().setInt64List(int64List1).build()
      val feature2 = Feature.newBuilder().setInt64List(int64List2).build()
      val featureList = FeatureList.newBuilder().addFeature(feature1).addFeature(feature2).build()

     assert(IntFeatureListDecoder.decode(featureList) === Seq(Seq(1,3), Seq(-2,5,10)))
    }

    "Decode empty feature list to empty array" in {
      val featureList = FeatureList.newBuilder().build()
      assert(IntFeatureListDecoder.decode(featureList).size === 0)
    }

    "Throw an exception if FeatureList is not of type Int64List" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(4).build()
        val feature = Feature.newBuilder().setFloatList(floatList).build()
        val featureList = FeatureList.newBuilder().addFeature(feature).build()
        IntFeatureListDecoder.decode(featureList)
      }
    }
  }

  "Long FeatureList decoder" should {

    "Decode FeatureList to 2-dimensional long array" in {
      val int64List1 = Int64List.newBuilder().addValue(1).addValue(Int.MaxValue+10L).build()
      val int64List2 = Int64List.newBuilder().addValue(Int.MinValue-20L).build()
      val intFeature1 = Feature.newBuilder().setInt64List(int64List1).build()
      val intFeature2 = Feature.newBuilder().setInt64List(int64List2).build()
      val featureList = FeatureList.newBuilder().addFeature(intFeature1).addFeature(intFeature2).build()

      assert(LongFeatureListDecoder.decode(featureList) === Seq(Seq(1L,Int.MaxValue+10L), Seq(Int.MinValue-20L)))
    }

    "Decode empty feature list to empty array" in {
      val featureList = FeatureList.newBuilder().build()
      assert(LongFeatureListDecoder.decode(featureList).size === 0)
    }

    "Throw an exception if FeatureList is not of type Int64List" in {
      intercept[Exception] {
        val floatList = FloatList.newBuilder().addValue(4).build()
        val feature = Feature.newBuilder().setFloatList(floatList).build()
        val featureList = FeatureList.newBuilder().addFeature(feature).build()
        LongFeatureListDecoder.decode(featureList)
      }
    }
  }

  "Float FeatureList decoder" should {

    "Decode FeatureList to 2-dimensional float array" in {
      val floatList1 = FloatList.newBuilder().addValue(1.3F).addValue(3.85F).build()
      val floatList2 = FloatList.newBuilder().addValue(-2.0F).build()
      val feature1 = Feature.newBuilder().setFloatList(floatList1).build()
      val feature2 = Feature.newBuilder().setFloatList(floatList2).build()
      val featureList = FeatureList.newBuilder().addFeature(feature1).addFeature(feature2).build()

      assert(FloatFeatureListDecoder.decode(featureList) ~== Seq(Seq(1.3F,3.85F), Seq(-2.0F)))
    }

    "Decode empty feature list to empty array" in {
      val featureList = FeatureList.newBuilder().build()
      assert(FloatFeatureListDecoder.decode(featureList).size === 0)
    }

    "Throw an exception if FeatureList is not of type FloatList" in {
      intercept[Exception] {
        val intList = Int64List.newBuilder().addValue(4).build()
        val feature = Feature.newBuilder().setInt64List(intList).build()
        val featureList = FeatureList.newBuilder().addFeature(feature).build()
        FloatFeatureListDecoder.decode(featureList)
      }
    }
  }

  "Double FeatureList decoder" should {

    "Decode FeatureList to 2-dimensional double array" in {
      val floatList1 = FloatList.newBuilder().addValue(4.3F).addValue(13.8F).build()
      val floatList2 = FloatList.newBuilder().addValue(-12.0F).build()
      val feature1 = Feature.newBuilder().setFloatList(floatList1).build()
      val feature2 = Feature.newBuilder().setFloatList(floatList2).build()
      val featureList = FeatureList.newBuilder().addFeature(feature1).addFeature(feature2).build()

      assert(DoubleFeatureListDecoder.decode(featureList) ~== Seq(Seq(4.3d,13.8d), Seq(-12.0d)))
    }

    "Decode empty feature list to empty array" in {
      val featureList = FeatureList.newBuilder().build()
      assert(DoubleFeatureListDecoder.decode(featureList).size === 0)
    }

    "Throw an exception if FeatureList is not of type FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("charles".getBytes)).build()
        val feature = Feature.newBuilder().setBytesList(bytesList).build()
        val featureList = FeatureList.newBuilder().addFeature(feature).build()
        DoubleFeatureListDecoder.decode(featureList)
      }
    }
  }

  "Decimal FeatureList decoder" should {

    "Decode FeatureList to 2-dimensional decimal array" in {
      val floatList1 = FloatList.newBuilder().addValue(4.3F).addValue(13.8F).build()
      val floatList2 = FloatList.newBuilder().addValue(-12.0F).build()
      val feature1 = Feature.newBuilder().setFloatList(floatList1).build()
      val feature2 = Feature.newBuilder().setFloatList(floatList2).build()
      val featureList = FeatureList.newBuilder().addFeature(feature1).addFeature(feature2).build()

      val arr = DecimalFeatureListDecoder.decode(featureList)
      assert(arr(0).map(_.toDouble) ~== Seq(4.3d,13.8d))
      assert(arr(1).map(_.toDouble) ~== Seq(-12.0d))
    }

    "Decode empty feature list to empty array" in {
      val featureList = FeatureList.newBuilder().build()
      assert(DecimalFeatureListDecoder.decode(featureList).size === 0)
    }

    "Throw an exception if FeatureList is not of type FloatList" in {
      intercept[Exception] {
        val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("charles".getBytes)).build()
        val feature = Feature.newBuilder().setBytesList(bytesList).build()
        val featureList = FeatureList.newBuilder().addFeature(feature).build()
        DecimalFeatureListDecoder.decode(featureList)
      }
    }
  }

  "String FeatureList decoder" should {

    "Decode FeatureList to 2-dimensional string array" in {
      val bytesList1 = BytesList.newBuilder().addValue(ByteString.copyFrom("alice".getBytes))
        .addValue(ByteString.copyFrom("bob".getBytes)).build()
      val bytesList2 = BytesList.newBuilder().addValue(ByteString.copyFrom("charles".getBytes)).build()

      val feature1 = Feature.newBuilder().setBytesList(bytesList1).build()
      val feature2 = Feature.newBuilder().setBytesList(bytesList2).build()
      val featureList = FeatureList.newBuilder().addFeature(feature1).addFeature(feature2).build()

      assert(StringFeatureListDecoder.decode(featureList) === Seq(Seq("alice", "bob"), Seq("charles")))
    }

    "Decode empty feature list to empty array" in {
      val featureList = FeatureList.newBuilder().build()
      assert(StringFeatureListDecoder.decode(featureList).size === 0)
    }

    "Throw an exception if FeatureList is not of type BytesList" in {
      intercept[Exception] {
        val intList = Int64List.newBuilder().addValue(4).build()
        val feature = Feature.newBuilder().setInt64List(intList).build()
        val featureList = FeatureList.newBuilder().addFeature(feature).build()
        StringFeatureListDecoder.decode(featureList)
      }
    }
  }
}
