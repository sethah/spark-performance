package com.sethah.sparkperf

import com.sethah.sparkperf.datagen.LSHGenerator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.functions._

object Main {

  def main(args: Array[String]): Unit = {
//    val logger = Logger.getLogger("org")
//    logger.setLevel(Level.WARN)
//    val spark = SparkSession.builder
//      .master("local[*]")
//      .appName("spark session example")
//      .getOrCreate()
//    import spark.implicits._
//
//
//    try {
//      val seed = 42
//      val numFeatures = args(0).toInt
//      val numPoints = args(1).toInt
//      val generator = new LSHGenerator(numFeatures, seed)
//      val rdd = RandomRDDs.randomRDD(spark.sparkContext,
//        generator, numPoints, spark.sparkContext.defaultParallelism, seed)
//      val df = rdd.toDF()
//      df.cache().count()
//      val brp = new BucketedRandomProjectionLSH()
//        .setNumHashTables(2)
//        .setBucketLength(10)
//        .setInputCol("features")
//        .setOutputCol("hashes")
//      def sameBucket(x: Seq[Vector], y: Seq[Vector]): Boolean = {
//        x.zip(y).exists(tuple => tuple._1 == tuple._2)
//      }
//
//      val key = df.first().getAs[Vector](1)
//      val model = brp.fit(df)
//      val keyHash = model.hashFunc(key)
//      // In the origin dataset, find the hash value that hash the same bucket with the key
//      val sameBucketWithKeyUDF = udf((x: Seq[Vector]) =>
//        sameBucket(x, keyHash), DataTypes.BooleanType)
//
//      df.groupBy("label").count().show()
//      println(model.transform(df).filter(sameBucketWithKeyUDF(col("hashes"))).count())
////      println(first)
////      val nn = model.approxNearestNeighbors(df, first, 5)
////      println(nn.count())
//      spark.stop()
//    } finally {
//      spark.stop()
//    }
  }
}
