
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;

import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.SQLContext;

public class DurationPredictor {
	
	
	public static void main(String[] args) throws URISyntaxException, IOException {
	    SparkConf sconf = new SparkConf().setAppName("Duration Predictor").setMaster("local[*]");
	    JavaSparkContext jsc = new JavaSparkContext(sconf);
	    SQLContext sqlContext = new SQLContext(jsc);
	    
	    

	    //String input = "hdfs://192.168.1.15:9000/btr/ctba/data/prediction_data.json";
	    
	    // Load training data
//	    JavaRDD<String> trainingRDD = jsc.textFile(args[0]);
//	    DataFrame training = sqlContext.read().format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(input);
	    DataFrameReader dReader = sqlContext.read();
	    DataFrame training = dReader.json(args[0]);
	    
	    training.show(1);
//
//	    LinearRegression lr = new LinearRegression()
//	      .setMaxIter(10)
//	      .setRegParam(0.3)
//	      .setElasticNetParam(1.0);
//
//	    // Fit the model
//	    LinearRegressionModel lrModel = lr.fit(training);
//
//	    // Print the coefficients and intercept for linear regression
//	    System.out.println("Coefficients: "
//	      + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
//
//	    // Summarize the model over the training set and print out some metrics
//	    LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
//	    System.out.println("numIterations: " + trainingSummary.totalIterations());
//	    System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
//	    trainingSummary.residuals().show();
//	    System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
//	    System.out.println("r2: " + trainingSummary.r2());
	    // $example off$

	    jsc.stop();
	
	}
}
