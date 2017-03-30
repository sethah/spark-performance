package com.sethah.sparkperf.datagen

import org.apache.spark.linalg.BLASWrapper
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.random.RandomDataGenerator

class LinearRegressionGenerator[F <: RandomDataGenerator[Vector]](
    private val coefficients: Vector,
    private val intercept: Double,
    private val featuresGenerator: F,
    private val errorSigma: Double,
    private val seed: Long) extends RandomDataGenerator[LabeledPoint] {

  private val rng = new java.util.Random(seed)
  rng.nextGaussian()

  override def nextValue(): LabeledPoint = {
    val features = featuresGenerator.nextValue()
    val label = BLASWrapper.dot(features, coefficients) + intercept
    val noisyLabel = label + errorSigma * rng.nextGaussian()
    LabeledPoint(noisyLabel, features)
  }

  def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): LinearRegressionGenerator[F] = {
    new LinearRegressionGenerator(coefficients, intercept, featuresGenerator, errorSigma, seed)
  }

}
