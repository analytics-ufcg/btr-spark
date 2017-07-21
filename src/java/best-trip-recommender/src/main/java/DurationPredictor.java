
import java.io.IOException;
import java.net.URISyntaxException;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


//import com.google.common.collect.ImmutableMap;

public class DurationPredictor {

	public static void main(String[] args) throws URISyntaxException, IOException {
//		SparkConf sconf = new SparkConf().setAppName("Duration Predictor").setMaster("local[*]");
//		JavaSparkContext jsc = new JavaSparkContext(sconf);
//		SQLContext sqlContext = new SQLContext(jsc);
		
		SparkSession spark = SparkSession
				  .builder()
				  .appName("Duration Predictor")
				  .config("master", "local[*]")
				  .config("spark.sql.files.ignoreCorruptFiles","true")
				  .getOrCreate();

		// Load training data
		Dataset<Row> training = spark.read()
				.format("com.databricks.spark.csv")
				.option("header", "true")
				.option("inferSchema", "true")
				.load(args[0]);

		training.show(2);
		
//		training = training.na().replace(new String[] {"difference_previous_schedule", "difference_next_schedule"}, ImmutableMap.of("NA", "-1"));
		training = training.filter("difference_previous_schedule != 'NA' AND difference_next_schedule != 'NA'");
		training = training.withColumn("difference_previous_schedule", training.col("difference_previous_schedule").cast("double"));
		training = training.withColumn("difference_next_schedule", training.col("difference_next_schedule").cast("double"));
		
		training.show(2);

		training.printSchema();
		
		// Automatically identify categorical features, and index them.
		// Set maxCategories so features with > 4 distinct values are treated as
		// continuous.
		String[] categoricalFeatures = {"route", "week_day"};
		PipelineStage[] featuresIndexers = new PipelineStage[categoricalFeatures.length];
		
		for (int i = 0; i < categoricalFeatures.length; i++) {
			featuresIndexers[i] = new StringIndexer() //VectorIndexer()
					.setInputCol(categoricalFeatures[i])
					.setOutputCol(categoricalFeatures[i] + "_index");
		}
		
		Pipeline pipeline = new Pipeline()
				  .setStages(featuresIndexers);
		
		Dataset<Row> trainingDF = pipeline
				.fit(training)
				.transform(training);
		
		trainingDF.show(1);

		trainingDF.printSchema();

		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] {"departure", "arrival", "route_index", "week_day_index",
						"difference_previous_schedule", "difference_next_schedule"})
				.setOutputCol("features");

		trainingDF = assembler.transform(trainingDF);

		trainingDF.show(1);

		trainingDF.printSchema();

		LinearRegression lr = new LinearRegression()
				.setMaxIter(10)
				.setRegParam(0.3)
				.setElasticNetParam(1.0)
				.setLabelCol("duration")
				.setFeaturesCol("features");
		
		// Fit the model
		LinearRegressionModel lrModel = lr.fit(trainingDF);		

		lrModel.save(args[1]);

		// Print the coefficients and intercept for linear regression
		System.out.println("Coefficients: " + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

		// Summarize the model over the training set and print out some metrics
		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
		System.out.println("numIterations: " + trainingSummary.totalIterations());
		System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
		trainingSummary.residuals().show();
		System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
		System.out.println("r2: " + trainingSummary.r2());

		System.out.println("==================== Loaded Model ====================");
		
		
		LinearRegressionModel lrModelLoaded = LinearRegressionModel.load(args[1]);
		
		Dataset<Row> predictions = lrModelLoaded.transform(trainingDF);
		
		predictions.show(5);
		
		// Select (prediction, true label) and compute test error.
		RegressionEvaluator evaluator = new RegressionEvaluator()
		  .setLabelCol("duration")
		  .setPredictionCol("prediction")
		  .setMetricName("rmse");
		double rmse = evaluator.evaluate(predictions);
		System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
		
		System.out.println("Coefficients: " + lrModelLoaded.coefficients() + " Intercept: " + lrModelLoaded.intercept());
		
		spark.stop();

	}
}
