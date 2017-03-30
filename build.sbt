name := "spark-performance"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0-SNAPSHOT" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.2.0-SNAPSHOT" % "provided",
  "org.apache.spark" %% "spark-mllib-local" % "2.2.0-SNAPSHOT" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.2.0-SNAPSHOT" % "provided",
  "com.github.tototoshi" %% "scala-csv" % "1.3.1"
)

fork := true
