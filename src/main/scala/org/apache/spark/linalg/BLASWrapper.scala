package org.apache.spark.linalg

import org.apache.spark.ml.linalg.{BLAS, Vector}

object BLASWrapper {

  def scal(alpha: Double, x: Vector): Unit = BLAS.scal(alpha, x)
  def axpy(alpha: Double, x: Vector, y: Vector): Unit = BLAS.axpy(alpha, x, y)
  def dot(x: Vector, y: Vector): Double = BLAS.dot(x, y)

}
