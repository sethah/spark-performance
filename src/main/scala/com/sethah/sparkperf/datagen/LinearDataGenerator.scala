package com.sethah.sparkperf.datagen

import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.random.RandomDataGenerator

case class LinearDataGenerator(
       private val means: Array[Double],
       private val stds: Array[Double],
       private val seed: Long) extends RandomDataGenerator[Vector] {

  val numFeatures = means.length
  val rng = new scala.util.Random(seed)

  override def nextValue(): Vector = {
    new DenseVector((0 until numFeatures).map { j =>
      rng.nextGaussian * stds(j) + means(j)
    }.toArray)
  }

  def setSeed(seed: Long): Unit = {
    rng.setSeed(seed)
  }

  override def copy(): LinearDataGenerator = {
    LinearDataGenerator(means, stds, seed)
  }

}
