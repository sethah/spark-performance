package com.sethah.sparkperf

import com.github.tototoshi.csv.CSVWriter
import com.sethah.sparkperf.datagen.{LinearDataGenerator, LinearRegressionGenerator}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.optim.optimizers.{EMSO, LBFGS, OWLQN, SGD}
import org.apache.spark.ml.optim.aggregator.LeastSquaresAggregator
import org.apache.spark.ml.optim.loss.LossFunction
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, LabeledPoint}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.functions._

import scala.util.Try

object LinearRegressionMain {

  def main(args: Array[String]): Unit = {
    val logger = Logger.getLogger("org")
    logger.setLevel(Level.WARN)
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("spark session example")
      .getOrCreate()
    import spark.implicits._
    Thread.sleep(2000)

    try {
      val seed = 42
      val numFeatures = args(1).toInt
      val numPoints = args(2).toInt
      val numPartitions = args(3).toInt
      val regParam = args(4).toDouble
      val elasticNetParam = args(5).toDouble
      val gamma = args(6).toDouble
      val noise = args(7).toDouble

      val maxIter = 30

      val timingFile = Try(args(0)).getOrElse("/root/timings.csv")
      val rng = new scala.util.Random(seed)
      val featureMeans = Array.fill(numFeatures)(rng.nextDouble())
      val featureStds = Array.fill(numFeatures)(rng.nextDouble() * 0.1)
      val coefficients = Vectors.dense(Array.fill(numFeatures)((rng.nextDouble - 0.5) * 5))
      val featuresGenerator = LinearDataGenerator(featureMeans, featureStds, seed)
      val generator = new LinearRegressionGenerator(coefficients, 3.1, featuresGenerator, noise, seed)
      val rdd = RandomRDDs.randomRDD(spark.sparkContext,
        generator, numPoints, numPartitions, seed)
      val counts = rdd.mapPartitions { it => Iterator.single(it.count(_ => true))}.collect()
      val tmp = rdd.mapPartitionsWithIndex((i, it) => Iterator.single(i)).collect().mkString(",")
//      val df = rdd.mapPartitionsWithIndex((i, it) => if (i == 0) it else {it.next(); it}).toDF()
//      val df = rdd.mapPartitionsWithIndex((i, it) => if (true) it else Iterator.single(it.next())).toDF()
//      val firstPartition = rdd.mapPartitionsWithIndex((i, it) => if (i == 0) it else Iterator()).collect()
//      val firstPartRDD = spark.sparkContext.parallelize(firstPartition).toDF()
      val df = spark.read.parquet(s"/Users/sethhendrickson/" +
        s"Development/datasets/linearRegression_${numFeatures}_${numPoints}_${noise}")
      df.show()
      val base = new LinearRegression()
        .setSolver("l-bfgs")
        .setRegParam(regParam)
        .setElasticNetParam(elasticNetParam)
//      val baseModel = base.fit(df)
//      val baseModel2 = base.setSolver("normal").fit(df)
      (1 to 10).foreach { i =>
        val model = base.fit(df.sample(false, 0.5))
        println(model.coefficients)
      }

      val numTrials = 1
      val times = new Array[Long](numTrials)
      var coefs = List[Vector]()
      for (trial <- 0 until numTrials) {
        val lr = new LinearRegression()
          .setSolver("l-bfgs")
          .setMaxIter(maxIter)
          .setMinimizer(new EMSO[LossFunction[RDD, LeastSquaresAggregator]](
            new LBFGS().setMaxIter(50), (i: Int) => gamma * (i + 1.0) / 2.0).setMaxIter(30).setTol(1e-10))
//            new SGD().setMaxIter(10), gamma).setMaxIter(20).setTol(1e-16))
//          .setMinimizer(new LBFGS())
          .setElasticNetParam(elasticNetParam)
          .setRegParam(regParam)
        val t0 = System.nanoTime()
        val model = lr.fit(df)
//        println("MOOOOODEL", model.summary.totalIterations, coefficients, model.coefficients)
//        println(featureStds.mkString(","))
        println("numIters", model.summary.totalIterations, model.summary.meanSquaredError)
        coefs = model.coefficients :: coefs
        val t1 = System.nanoTime()
        times(trial) = t1 - t0
      }
      println("Summary")
      println(coefficients)
//      println(baseModel.coefficients)
//      println(baseModel.summary.meanSquaredError)
//      println(baseModel.summary.objectiveHistory.mkString(","))
//      println(baseModel2.coefficients)
//      println(baseModel2.summary.meanSquaredError)
      coefs.foreach(println)
//      Thread.sleep(1000000)
//      val writer = CSVWriter.open(timingFile, append = true)
//      writer.writeRow(List(numFeatures, numPoints, maxIter, regParam, elasticNetParam)
//        ::: times.toList)
//      writer.close()
      spark.stop()
    } finally {
      spark.stop()
    }
  }
}

