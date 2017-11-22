import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
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
                  features=["busStopIdOrig", "busStopIdDest", "scaledCoordinates",
                            "hourOrig", "isRushOrig", "weekOfYear", "dayOfMonth",
                            "month", "isHoliday", "isWeekend", "isRegularDay", "distance"]):

    df = df.na.drop(subset = string_columns + features)

    pipelineStages = []

    indexers = [StringIndexer(inputCol = column, outputCol = column + "_index") for column in string_columns]

    assembler = VectorAssembler(
        inputCols = features + map(lambda c: c + "_index", string_columns),
        outputCol = 'features')

    pipelineStages = pipelineStages + indexers
    pipelineStages.append(assembler)

    pipeline = Pipeline(stages = pipelineStages)

    pipeline.write().overwrite().save(pipeline_path)

    assembled_df = featuresAssembler.transform(df)

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

    trainingSummary = RegressionMetrics(predictions.rdd.map(lambda row: (row.prediction, row.duration)))

    return (predictions, trainingSummary)


def save_train_info(model, test_data, filepath):
    predictions, trainingSummary = getPredictionsLabels(model, test_data)

    result = sqlContext.createDataFrame([
        "Model:",
        #"Coefficients: %s" % str(model.coefficients),
        #"Intercept: %s" % str(model.intercept),
        "Model info",
        "RMSE: %f" % trainingSummary.rootMeanSquaredError,
        "MAE: %f" % trainingSummary.meanAbsoluteError,
        "r2: %f" % trainingSummary.r2
        ], StringType())


    result.write.text(filepath + "info/")


def save_model(model, filepath):
    model.write().overwrite().save(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <training-data-path> " \
              "<duration-model-path-to-save> <crowdedness-model-path-to-save> <pipeline-path-to-save>, <train-info-path>" % (sys.argv[0])
        sys.exit(1)
    #elif not os.path.exists(sys.argv[1]):
    #    print "Error: training-data-filepath doesn't exist! You must specify a valid one!"
    #    sys.exit(1)

    training_data_path, duration_model_path_to_save, crowdedness_model_path_to_save, pipeline_path, train_info_path = sys.argv[1:6]


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

    save_train_info(duration_model, duration_predictions_and_labels, "Duration model\n", train_info_output_filepath)
    save_train_info(crowdedness_model, crowdedness_predictions_and_labels, "Crowdedness model\n", train_info_output_filepath)

    save_model(duration_model, duration_model_path_to_save)
    save_model(crowdedness_model, crowdedness_model_path_to_save)


    sc.stop()
