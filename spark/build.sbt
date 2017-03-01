name := "spark-tensorflow-connector"

organization := "org.trustedanalytics"

scalaVersion in Global := "2.11.8"

spName := "tapanalyticstoolkit/spark-tensorflow-connector"

sparkVersion := "2.1.0"

sparkComponents ++= Seq("sql", "mllib")

version := "1.0.0"

def ProjectName(name: String,path:String): Project =  Project(name, file(path))

resolvers in Global ++= Seq("https://tap.jfrog.io/tap/public" at "https://tap.jfrog.io/tap/public" ,
           "https://tap.jfrog.io/tap/public-snapshots" at "https://tap.jfrog.io/tap/public-snapshots" ,
           "https://repo.maven.apache.org/maven2" at "https://repo.maven.apache.org/maven2" )

val `junit_junit` = "junit" % "junit" % "4.12"

val `org.apache.hadoop_hadoop-yarn-api` = "org.apache.hadoop" % "hadoop-yarn-api" % "2.7.3"

val `org.apache.spark_spark-core_2.11` = "org.apache.spark" % "spark-core_2.11" % "2.1.0"

val `org.apache.spark_spark-sql_2.11` = "org.apache.spark" % "spark-sql_2.11" % "2.1.0"

val `org.apache.spark_spark-mllib_2.11` = "org.apache.spark" % "spark-mllib_2.11" % "2.1.0"

val `org.scalatest_scalatest_2.11` = "org.scalatest" % "scalatest_2.11" % "2.2.6"

val `org.tensorflow_tensorflow-hadoop` = "org.tensorflow" % "tensorflow-hadoop" % "1.0-01232017-SNAPSHOT"

libraryDependencies in Global ++= Seq(`org.tensorflow_tensorflow-hadoop` classifier "shaded-protobuf",
   `org.scalatest_scalatest_2.11` % "test" ,
   `org.apache.spark_spark-sql_2.11` % "provided" ,
   `org.apache.spark_spark-mllib_2.11` % "test" classifier "tests",
   `org.apache.spark_spark-core_2.11` % "provided" ,
   `org.apache.hadoop_hadoop-yarn-api` % "provided" ,
   `junit_junit` % "test" )

assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp filterNot {x => List("spark-tensorflow-connector-1.0-SNAPSHOT.jar",
    "tensorflow-hadoop-1.0-01232017-SNAPSHOT-shaded-protobuf.jar").contains(x.data.getName)}
}

/********************
  * Release settings *
  ********************/

spIgnoreProvided := true

spAppendScalaVersion := true

// If you published your package to Maven Central for this release (must be done prior to spPublish)
spIncludeMaven := false

publishMavenStyle := true

licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

pomExtra :=
  <url>https://github.com/tapanalyticstoolkit/spark-tensorflow-connector</url>
    <scm>
      <url>git@github.com:tapanalyticstoolkit/spark-tensorflow-connector.git</url>
      <connection>scm:git:git@github.com:tapanalyticstoolkit/spark-tensorflow-connector.git</connection>
    </scm>
    <developers>
      <developer>
        <id>karthikvadla</id>
        <name>Karthik Vadla</name>
        <url>https://github.com/karthikvadla</url>
      </developer>
      <developer>
        <id>skavulya</id>
        <name>Soila Kavulya</name>
        <url>https://github.com/skavulya</url>
      </developer>
      <developer>
        <id>joyeshmishra</id>
        <name>Joyesh Mishra</name>
        <url>https://github.com/joyeshmishra</url>
      </developer>
    </developers>

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials") // A file containing credentials

// Add assembly jar to Spark package
test in assembly := {}

spShade := true
