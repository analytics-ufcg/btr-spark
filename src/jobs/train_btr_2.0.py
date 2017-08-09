import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RegressionMetrics

# MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression

import sys
import os


def read_data(sqlContext, filepath):
    return sqlContext.read.format("com.databricks.spark.csv")\
        .option("header", "true")\
        .option("inferSchema", "true") \
        .option("nullValue", "-")\
        .load(filepath)


def data_pre_proc(df,
                  string_columns = ["BUS_CODE", "PERIOD_ORIG", "PERIOD_DEST", "WEEK_DAY"],
                  features=["TRIP_NUM_ORIG", "ROUTE", "SHAPE_ID", "SHAPE_SEQ", "LAT_SHAPE_ORIG", "LON_SHAPE_ORIG",
                            "STOP_ID_ORIG", "STOP_ID_DEST", "TRIP_NUM_DEST", "LAT_SHAPE_DEST", "LON_SHAPE_DEST",
                            "HOUR_ORIG", "HOUR_DEST", "IS_RUSH_ORIG", "IS_RUSH_DEST", "WEEK_OF_YEAR", "DAY_OF_MONTH",
                            "MONTH", "IS_HOLIDAY", "IS_WEEKEND", "IS_REGULAR_DAY", "TOTAL_DISTANCE"]):
    indexers = [StringIndexer(inputCol = column, outputCol = column + "_index").fit(df) for column in string_columns]
    pipeline = Pipeline(stages=indexers)
    df_r = pipeline.fit(df).transform(df)

    assembler = VectorAssembler(
    inputCols=features,
    outputCol='features')

    assembled_df = assembler.transform(df_r)

    return assembled_df


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 %s <training-data-path> " \
              "<train-info-output-filepath> <duration-model-path-to-save>" % (sys.argv[0])
        sys.exit(1)
    elif not os.path.exists(sys.argv[1]):
        print "Error: training-data-filepath doesn't exist! You must specify a valid one!"
        sys.exit(1)

    training_data_path, train_info_output_filepath, duration_model_path_to_save = sys.argv[1:5]

    sc = SparkContext("local[*]", appName="train_btr_2.0")
    sqlContext = pyspark.SQLContext(sc)
    data = read_data(sqlContext, training_data_path)

    preproc_data = data_pre_proc(data)

    train, test = preproc_data.randomSplit([0.7, 0.3], 24)

    #
    # # Duration
    # duration_model = train_duration_model(preproc_data)
    #
    # predictions_and_labels = getPredictionsLabels(duration_model, preproc_data)
    # save_train_info(duration_model, predictions_and_labels, "Duration model\n", train_info_output_filepath)
    #
    # save_model(duration_model, duration_model_path_to_save)
    #
    # duration_model_loaded = LinearRegressionModel.load(duration_model_path_to_save)
    #
    # print duration_model_loaded.coefficients[0]
    #
    # # Crowdedness
    # crowdedness_model = train_crowdedness_model(preproc_data)
    #
    # predictions_and_labels = getPredictionsLabels(duration_model, preproc_data)
    # save_train_info(crowdedness_model, predictions_and_labels, "Crowdedness model\n", train_info_output_filepath)
    #
    # save_model(crowdedness_model, crowdedness_model_path_to_save)
    #
    # crowdedness_model_loaded = LinearRegressionModel.load(crowdedness_model_path_to_save)
    #
    # print crowdedness_model_loaded.coefficients[0]

    sc.stop()
