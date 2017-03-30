package com.sethah.sparkperf.datagen

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.random.{RandomDataGenerator, RandomRDDs}
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class LSHGenerator(
                    private val numFeatures: Int,
    private val seed: Long) extends RandomDataGenerator[LabeledPoint] {

  private val rng = new java.util.Random(seed)
  val numCenters = 3
  val centers = (0 until numCenters).map { k =>
    new DenseVector(Array.fill(numFeatures)(rng.nextGaussian * 10))
  }


  override def nextValue(): LabeledPoint = {
    val center = rng.nextInt(numCenters)
    val x = new DenseVector(centers(center).values.map(v => v + rng.nextGaussian()))
    val cdist = centers.map(c => LSHGenerator.dist(c, x))
    println(cdist.mkString(","))
    LabeledPoint(center.toDouble, x)
  }

  def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): LSHGenerator = new LSHGenerator(numFeatures, seed)

}

object LSHGenerator {
  def dist(x: Vector, y: Vector): Double = {
    val res = x.copy
//    BLAS.axpy(-1.0, y, res)
//    math.sqrt(BLAS.dot(res, res))
    0.0
  }
}

//object MultinomialDataGenerator {
//  def makeData(
//                spark: SparkSession,
//                numClasses: Int,
//                numFeatures: Int,
//                fitIntercept: Boolean,
//                numPoints: Int,
//                seed: Long): RDD[LabeledPoint] = {
//    val rng = scala.util.Random
//    rng.setSeed(seed)
//    val coefWithInterceptLength = if (fitIntercept) numFeatures + 1 else numFeatures
//    val coefficients = Array.tabulate((numClasses - 1) * coefWithInterceptLength) { i =>
//      rng.nextDouble() - 0.5
//    }
//    val xMean = Array.tabulate(numFeatures) { i => (rng.nextDouble() - 0.5) * 5}
//    val xVariance = Array.tabulate(numFeatures) { i => rng.nextDouble() * 2 + 1}
//    val generator = new MultinomialDataGenerator(coefficients, xMean, xVariance, fitIntercept, seed)
//    RandomRDDs.randomRDD(spark.sparkContext,
//      generator, numPoints, spark.sparkContext.defaultParallelism, seed)
//  }
//}