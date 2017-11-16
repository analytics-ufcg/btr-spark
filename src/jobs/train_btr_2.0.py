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
    df = sqlContext.read.csv(filepath, header=True, inferSchema=True, nullValue="-")

    df = df.withColumn("duration", df.duration.cast('Double'))

    return df


def data_pre_proc(df, pipeline_path,
                  string_columns = ["periodOrig", "weekDay", "route"],
                  features=["shapeLatOrig", "shapeLonOrig",
                            "busStopIdOrig", "busStopIdDest", "shapeLatDest", "shapeLonDest",
                            "hourOrig", "isRushOrig", "weekOfYear", "dayOfMonth",
                            "month", "isHoliday", "isWeekend", "isRegularDay", "distance"]):

    df = df.na.drop(subset = string_columns + features)

    indexers = [StringIndexer(inputCol = column, outputCol = column + "_index").fit(df) for column in string_columns]
    pipeline = Pipeline(stages = indexers)

    pipeline.write().overwrite().save(pipeline_path)

    df_r = pipeline.fit(df).transform(df)

    assembler = VectorAssembler(
    inputCols = features + map(lambda c: c + "_index", string_columns),
    outputCol = 'features')

    assembled_df = assembler.transform(df_r)

    return assembled_df


def train_duration_model(training_df):
    duration_lr = LinearRegression(maxIter=10, regParam=0.01, elasticNetParam=1.0).setLabelCol("duration").setFeaturesCol("features")

    duration_lr_model = duration_lr.fit(training_df)

    return duration_lr_model

def train_crowdedness_model(training_df):
    crowdedness_lr = LinearRegression(maxIter=10, regParam=0.01, elasticNetParam=1.0).setLabelCol("probableNumPassengers").setFeaturesCol("features")

    crowdedness_lr_model = crowdedness_lr.fit(training_df)

    return crowdedness_lr_model


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


def save_model(model, filepath):
    model.write().overwrite().save(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <training-data-path> " \
              "<duration-model-path-to-save> <crowdedness-model-path-to-save> <pipeline-path-to-save>" % (sys.argv[0])
        sys.exit(1)
    #elif not os.path.exists(sys.argv[1]):
    #    print "Error: training-data-filepath doesn't exist! You must specify a valid one!"
    #    sys.exit(1)

    training_data_path, duration_model_path_to_save, crowdedness_model_path_to_save, pipeline_path = sys.argv[1:5]

    sc = SparkContext(appName="train_btr_2.0")
    sqlContext = pyspark.SQLContext(sc)
    data = read_data(sqlContext, training_data_path)

    preproc_data = data_pre_proc(data, pipeline_path)

    train, test = preproc_data.randomSplit([0.7, 0.3], 24)

    # Duration
    duration_model = train_duration_model(train)
    # Crowdedness
    crowdedness_model = train_crowdedness_model(train)

    duration_predictions_and_labels = getPredictionsLabels(duration_model, test)
    crowdedness_predictions_and_labels = getPredictionsLabels(crowdedness_model, test)

    #save_train_info(duration_model, duration_predictions_and_labels, "Duration model\n", train_info_output_filepath)
    #save_train_info(crowdedness_model, crowdedness_predictions_and_labels, "Crowdedness model\n", train_info_output_filepath)

    save_model(duration_model, duration_model_path_to_save)
    save_model(crowdedness_model, crowdedness_model_path_to_save)

    #duration_model_loaded = LinearRegressionModel.load(duration_model_path_to_save)

    sc.stop()
