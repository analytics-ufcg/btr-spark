import pyspark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RegressionMetrics

#MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression

import sys
from pyspark import SparkContext
import os

def getPredictionsLabels(model, test_data):
	predictions = model.transform(test_data)

	return predictions.rdd.map(lambda row: (row.prediction, row.duration))

def save_train_info(model, predictions_and_labels, output, filepath = "hdfs://localhost:9000/btr/ctba/output.txt"):
	with open(filepath, 'a') as outfile:
		output += "Model:\n"
		output += "Coefficients: %s\n" % str(model.coefficients)
		output += "Intercept: %s\n" % str(model.intercept)
		output += "Model info\n"

		trainingSummary = RegressionMetrics(predictions_and_labels)

		output += "RMSE: %f\n" % trainingSummary.rootMeanSquaredError
		output += "MAE: %f\n" % trainingSummary.meanAbsoluteError
		output += "r2: %f\n" % trainingSummary.r2

		outfile.write(output)

def read_data(sqlContext, filepath = "../data/curitiba/prediction_data.csv"):
	df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema","true").load(filepath)

	df = df.withColumn("totalpassengers", df['totalpassengers'].cast('Double'))

	#df = df.withColumn('date_timestamp', df['date'].cast('Integer'))

	return df

def data_pre_proc(df, string_columns = ["route", "week_day", "difference_previous_schedule", "difference_next_schedule"],
				  features = ["route_index", "week_day_index", "group_15_minutes", "difference_next_schedule_index", "difference_previous_schedule_index"]):
	indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in string_columns]
	pipeline = Pipeline(stages=indexers)
	df_r = pipeline.fit(df).transform(df)

	assembler = VectorAssembler(
	inputCols=features,
	outputCol='features')

	assembled_df = assembler.transform(df_r)

	return assembled_df

def train_duration_model(training_df):
	duration_lr = LinearRegression(maxIter=10, regParam=0.01, elasticNetParam=1.0).setLabelCol("duration").setFeaturesCol("features")

	duration_lr_model = duration_lr.fit(training_df)

	return duration_lr_model

def train_crowdedness_model(training_df):
	crowdedness_lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=1.0).setLabelCol("totalpassengers").setFeaturesCol("features")

	crowdedness_lr_model = crowdedness_lr.fit(training_df)

	return crowdedness_lr_model

def save_model(model, filepath):
	model.write().overwrite().save(filepath)

if __name__ == "__main__":
	if len(sys.argv) < 5:
        	print "Error: Wrong parameter specification!"
		print "Your command should be something like:"
		print "spark-submit com.databricks:spark-csv_2.10:1.5.0 <%s-directory> <training-data-filepath> <train-info-output-filepath> <duration-model-path-to-save> <crowdedness-model-path-to-save>" % (sys.argv[0])
		sys.exit(1)
	elif not os.path.exists(sys.argv[1]):
		print "Error: training-data-filepath doesn't exist! You must specify a valid one!"
                sys.exit(1)		

	training_data_filepath, train_info_output_filepath, duration_model_path_to_save, crowdedness_model_path_to_save = sys.argv[1:5]

	sc = SparkContext("local[*]", appName="train_btr")
	sqlContext = pyspark.SQLContext(sc)
	data = read_data(sqlContext, training_data_filepath)
	preproc_data = data_pre_proc(data)

	# Duration
	duration_model = train_duration_model(preproc_data)

	predictions_and_labels = getPredictionsLabels(duration_model, preproc_data)
	save_train_info(duration_model, predictions_and_labels, "Duration model\n", train_info_output_filepath)

	save_model(duration_model, duration_model_path_to_save)

	duration_model_loaded = LinearRegressionModel.load(duration_model_path_to_save)

	print duration_model_loaded.coefficients[0]

	# Crowdedness
	crowdedness_model = train_crowdedness_model(preproc_data)

	predictions_and_labels = getPredictionsLabels(duration_model, preproc_data)
	save_train_info(crowdedness_model, predictions_and_labels, "Crowdedness model\n", train_info_output_filepath)

	save_model(crowdedness_model, crowdedness_model_path_to_save)

	crowdedness_model_loaded = LinearRegressionModel.load(crowdedness_model_path_to_save)

	print crowdedness_model_loaded.coefficients[0]

	sc.stop()
