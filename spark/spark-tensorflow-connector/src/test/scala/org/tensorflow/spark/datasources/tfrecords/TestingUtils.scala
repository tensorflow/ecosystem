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
package org.tensorflow.spark.datasources.tfrecords

import org.scalatest.Matchers
import org.scalatest.matchers.{MatchResult, Matcher}

object TestingUtils extends Matchers {

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class FloatArrayWithAlmostEquals(val left: Seq[Float]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Float], epsilon : Float = 1E-6F): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a === (b +- epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class DoubleArrayWithAlmostEquals(val left: Seq[Double]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Double], epsilon : Double = 1E-6): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a === (b +- epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class FloatMatrixWithAlmostEquals(val left: Seq[Seq[Float]]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Seq[Float]], epsilon : Float = 1E-6F): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a ~== (b, epsilon) }
      }
      else false
    }
  }


  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class DoubleMatrixWithAlmostEquals(val left: Seq[Seq[Double]]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Seq[Double]], epsilon : Double = 1E-6): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a ~== (b, epsilon) }
      }
      else false
    }
  }
}