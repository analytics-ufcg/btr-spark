import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql.types import StringType

# MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression, RandomForestRegressor, GBTRegressor

import sys
import os

def read_data(sqlContext, filepath):
    df = sqlContext.read.csv(filepath, header=True, inferSchema=True, nullValue="-")

    df = df.withColumn("duration", df.duration.cast('Double'))

    return df

# df = df.na.drop(subset = string_columns + features)

def execute_preproc_pipeline(df, pipeline_path, string_columns = ["periodOrig", "weekDay"],#, "route"],
                            features=["shapeLatOrig", "shapeLonOrig", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "isHoliday", "isWeekend", "isRegularDay", "distance", "month", "weekOfYear", "dayOfMonth"]):
    pipelineStages = []

    coordinatesFeatures = ["shapeLatOrig", "shapeLonOrig", "shapeLatDest", "shapeLonDest"]

    # Assemble a vector with coordinates that will be the input of the scaler
    scalerVectorAssembler = VectorAssembler(inputCols=coordinatesFeatures,
                                  outputCol="coordinates")
    # Normalize the scale of coordinates
    coordinatesScaler = MinMaxScaler(inputCol="coordinates", outputCol="scaledCoordinates")

    pipelineStages.append(scalerVectorAssembler)
    pipelineStages.append(coordinatesScaler)

    # Add _index to categorical variables
    for column in string_columns:
        indexer = StringIndexer(inputCol = column, outputCol = column + "_index")
        pipelineStages.append(indexer)

    features = [f for f in features if f not in coordinatesFeatures]
    features.append('scaledCoordinates')
    # Assemble a vector with all the features that will be the input of the model
    featuresAssembler = VectorAssembler(
            inputCols = features + map(lambda c: c + "_index", string_columns),
            outputCol = 'features')

    pipelineStages.append(featuresAssembler)

    pipeline = Pipeline(stages = pipelineStages)
    pipelineModel =  pipeline.fit(df)
    pipelineModel.write().overwrite().save(pipeline_path)
    df = pipelineModel.transform(df)

    return df

def train_duration_model(training_df):
    duration_lr = LinearRegression(maxIter=10, regParam=0.01, elasticNetParam=1.0).setLabelCol("duration").setFeaturesCol("features")
    duration_lr_model = duration_lr.fit(training_df)
    return duration_lr_model

def train_crowdedness_model(training_df):
    crowdedness_lr = LinearRegression(maxIter=10, regParam=0.01, elasticNetParam=1.0).setLabelCol("crowdedness").setFeaturesCol("features")
    crowdedness_lr_model = crowdedness_lr.fit(training_df)
    return crowdedness_lr_model

def train_pacing_model(training_df):
    pacing_regressor = GBTRegressor(maxDepth = 4, maxBins = 300).setLabelCol("pacing").setFeaturesCol("features")
    pacing_model = pacing_regressor.fit(training_df)
    return pacing_model

def train_speed_model(training_df):
    speed_regressor = LinearRegression(maxIter=10, regParam=0.01, elasticNetParam=1.0).setLabelCol("speed").setFeaturesCol("features")
    speed_model = speed_regressor.fit(training_df)
    return speed_model

def getPredictionsLabels(model, test_data):
    predictions = model.transform(test_data)
    trainingSummary = RegressionMetrics(predictions.rdd.map(lambda row: (row.prediction, row.duration)))
    return (predictions, trainingSummary)

def save_train_info(model, test_data, model_name, filepath):
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


    result.write.mode("overwrite").text(filepath + model_name)


def save_model(model, filepath):
    model.write().overwrite().save(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <training-data-path> " \
              "<output-folder-path>" % (sys.argv[0])
        sys.exit(1)
    #elif not os.path.exists(sys.argv[1]):
    #    print "Error: training-data-filepath doesn't exist! You must specify a valid one!"
    #    sys.exit(1)

    training_data_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    duration_model_path_to_save = output_folder_path + "duration-model/"
    crowdedness_model_path_to_save = output_folder_path + "crowdedness-model/"
    pacing_model_path_to_save = output_folder_path + "pacing-model/"
    speed_model_path_to_save = output_folder_path + "speed-model/"
    pipeline_path = output_folder_path + "pipeline/"
    train_info_path = output_folder_path + "train-info/"

    sc = SparkContext(appName="train_btr_2.0")
    sqlContext = pyspark.SQLContext(sc)
    data = read_data(sqlContext, training_data_path)

    preproc_data = execute_preproc_pipeline(data, pipeline_path)

    train, test = preproc_data.randomSplit([0.7, 0.3], 24)

    # Duration
    duration_model = train_duration_model(train)
    # Crowdedness
    crowdedness_model = train_crowdedness_model(train.na.drop(subset=["crowdedness"]))
    # Pacing
    #pacing_model = train_pacing_model(train.na.drop(subset=["pacing"]))
    # Speed
    #speed_model = train_speed_model(train.na.drop(subset=["speed"]))

    save_train_info(duration_model, test, "duration", train_info_path)
    save_train_info(crowdedness_model, test.na.drop(subset=["crowdedness"]), "crowdedness", train_info_path)
    #save_train_info(pacing_model, test, "pacing", train_info_path)
    #save_train_info(speed_model, test, "speed", train_info_path)

    save_model(duration_model, duration_model_path_to_save)
    save_model(crowdedness_model, crowdedness_model_path_to_save)
    #save_model(pacing_model, pacing_model_path_to_save)
    #save_model(speed_model, speed_model_path_to_save)

    sc.stop()
