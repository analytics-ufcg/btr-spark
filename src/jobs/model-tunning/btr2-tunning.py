import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType

# MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

import sys, os
from datetime import datetime

def train_lasso(train_data, test_data, label, file_to_save):
    duration_lr = LinearRegression(elasticNetParam=1.0).setLabelCol(label).setFeaturesCol("features")

    paramGrid = ParamGridBuilder() \
    .addGrid(duration_lr.maxIter, [10, 20]) \
    .addGrid(duration_lr.regParam, [0.1, 0.01]) \
    .build()

    crossval = CrossValidator(estimator=duration_lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(labelCol=label),
                              numFolds=5)

    cvModel = crossval.fit(train_data)

    bestModel = cvModel.bestModel
    regParam = bestModel._java_obj.getRegParam()
    maxIter = bestModel._java_obj.getMaxIter()

    with open(params_folder_path + 'lassoParams.csv','wb') as file:
        file.write("param, value" + '\n')
        file.write("maxIter, " + str(maxIter) + '\n')
        file.write("regParam, " + str(regParam) + '\n')

    save_test_info(cvModel.bestModel, test_data, label + "-lasso", file_to_save)

    return cvModel

def train_rf(train_data, test_data, label, file_to_save):
    route_arity = train_data.select('route').distinct().count()

    duration_rf = RandomForestRegressor(labelCol=label, featuresCol="features")

    paramGrid = (ParamGridBuilder()
             .addGrid(duration_rf.maxDepth, [2, 4, 6])
             .addGrid(duration_rf.maxBins, [300])
             .addGrid(duration_rf.numTrees, [5, 20])
             .build())

    crossval = CrossValidator(estimator=duration_rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(labelCol=label),
                        numFolds=2)

    cvModel = crossval.fit(train_data)

    bestModel = cvModel.bestModel
    maxDepth = bestModel._java_obj.getMaxDepth()
    maxBins = bestModel._java_obj.getMaxBins()
    numTrees = bestModel._java_obj.getnumTrees()

    with open(params_folder_path + 'randomForestParams.csv','wb') as file:
        file.write("param, value" + '\n')
        file.write("maxDepth, " + str(maxDepth) + '\n')
        file.write("maxBins, " + str(maxBins) + '\n')
        file.write("numTrees, " + str(numTrees) + '\n')

    save_test_info(bestModel, test_data, label + "-random-forest", file_to_save)

    return cvModel;

def train_gbt(train_data, test_data, label, file_to_save):
    route_arity = train_data.select('route').distinct().count()

    duration_gbt = GBTRegressor(labelCol=label, featuresCol="features")

    paramGrid = (ParamGridBuilder()
             .addGrid(duration_gbt.maxDepth, [2, 4, 6])
             .addGrid(duration_gbt.maxBins, [300])
             .build())

    crossval = CrossValidator(estimator=duration_gbt,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(labelCol=label),
                        numFolds=5)

    cvModel = crossval.fit(train_data)

    bestModel = cvModel.bestModel
    maxDepth = bestModel._java_obj.getMaxDepth()
    maxBins = bestModel._java_obj.getMaxBins()

    with open(params_folder_path + 'GBTParams.csv','wb') as file:
        file.write("param, value" + '\n')
        file.write("maxDepth, " + str(maxDepth) + '\n')
        file.write("maxBins, " + str(maxBins) + '\n')

    save_test_info(cvModel.bestModel, test_data, label + "-gbt", file_to_save)

    return cvModel;

# exemplo de chamada: save_train_info(cvModel, "Lasso", <nome do arquivo para salvar os resultados>)
def save_test_info(model, test_data, model_name, output_path):

    predictions, trainingSummary = getPredictionsLabels(model, test_data)

    predictions = predictions.select("duration", "crowdedness", "pacing", "speed", "prediction")

    print predictions.describe().show()

    # output = model_name + "\n"
    # with open(output_path + model_name, 'a') as outfile:
    #     output += "Model:\n"
    #     output += "Coefficients: %s\n" % str(model.coefficients)
    #     output += "Intercept: %s\n" % str(model.intercept)
    #     output += "Model info\n"
    #
    #     output += "RMSE: %f\n" % trainingSummary.rootMeanSquaredError
    #     output += "MAE: %f\n" % trainingSummary.meanAbsoluteError
    #     output += "r2: %f\n" % trainingSummary.r2
    #
    #     outfile.write(output)

    result = sqlContext.createDataFrame([
        "Model:",
        #"Coefficients: %s" % str(model.coefficients),
        #"Intercept: %s" % str(model.intercept),
        "Model info",
        "RMSE: %f" % trainingSummary.rootMeanSquaredError,
        "MAE: %f" % trainingSummary.meanAbsoluteError,
        "r2: %f" % trainingSummary.r2
        ], StringType())

    predictions.write.csv(output_path + "predictions/" + model_name, mode="overwrite", header = True)

    model.write().overwrite().save(output_path + "model/" + model_name)

    result.write.mode("overwrite").text(output_path + "info/" + model_name)


def getPredictionsLabels(model, test_data):
    predictions = model.transform(test_data)

    trainingSummary = RegressionMetrics(predictions.rdd.map(lambda row: (row.prediction, row.duration)))

    return (predictions, trainingSummary)


def execute_preproc_pipeline(df, output_path, string_columns = ["periodOrig", "weekDay", "route"],
                            features=["busStopIdOrig", "busStopIdDest", "shapeLatOrig", "shapeLonOrig", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "isHoliday", "isWeekend", "isRegularDay", "month", "weekOfYear", "dayOfMonth"]):
    pipelineStages = []
    pipelinePath = output_path + "pipeline/"

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
    pipelineModel.write().overwrite().save(pipelinePath)
    df = pipelineModel.transform(df)

    return df

def read_data(datapath, string_columns = ["periodOrig", "weekDay", "route"],
                            features=["shapeLatOrig", "shapeLonOrig", "busStopIdOrig",
                             "busStopIdDest", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "weekOfYear", "dayOfMonth", "month", "isHoliday",
                             "isWeekend", "isRegularDay", "distance", "crowdedness",
                             "pacing", "speed"]):

    training_raw = sqlContext.read.csv(datapath, header=True, inferSchema=True, nullValue="-")

    training = training_raw.withColumn("duration", training_raw.duration.cast('Double'))\
    	.na.drop(subset = string_columns + features)

    return training

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <training-data-path> <btr2-tunning-output-path>" \
              "<train-start-date(YYYY-MM-DD)>" "<train-end-date(YYYY-MM-DD)>"\
              "<test-end-date(YYYY-MM-DD)>" % (sys.argv[0])
        sys.exit(1)

    training_data_path = sys.argv[1]
    output_path = sys.argv[2]
    train_start_date = sys.argv[3]
    train_end_date = sys.argv[4]
    test_end_date = sys.argv[5]

    if train_start_date >= train_end_date or train_end_date >= test_end_date:
        print "Error: Dates cannot be equals and must be in chronological order"
        print "Your command should be something like:"
        print "spark-submit %s <training-data-path> <btr2-tunning-output-path>" \
              "<train-start-date(YYYY-MM-DD)>" "<train-end-date(YYYY-MM-DD)>"\
              "<test-end-date(YYYY-MM-DD)>" % (sys.argv[0])
        sys.exit(1)

    global params_folder_path
    params_folder_path = output_path + 'cv-params/'
    if not os.path.exists(params_folder_path):
        os.makedirs(params_folder_path)

    sc = SparkContext(appName="tunning_btr_2.0")
    global sqlContext
    sqlContext = pyspark.SQLContext(sc)

    raw_data = read_data(training_data_path)

    transformed_data = execute_preproc_pipeline(raw_data, output_path, features=["busStopIdOrig", "busStopIdDest", "hourOrig",
                             "isRushOrig", "weekOfYear", "dayOfMonth", "month", "isHoliday", "isWeekend", "isRegularDay"])

    train_data = transformed_data.filter("date >= '" + train_start_date + "' and date <= '" + train_end_date + "'")
    test_data = transformed_data.filter("date > '" + train_end_date + "' and date <= '" + test_end_date + "'")

    duration = "duration"
    crowdedness = "crowdedness"
    pacing = "pacing"
    speed = "speed"

    # Lasso
    # duration_lasso_model = train_lasso(train_data, test_data, pacing, output_path)
    # pass_lasso_model = train_lasso(train_data, test_data, crowdedness, output_path)
    pacing_lasso_model = train_lasso(transformed_data, transformed_data, duration, output_path)
    speed_lasso_model = train_lasso(transformed_data, transformed_data, speed, output_path)


    # Random Forest
    # duration_rf_model = train_rf(train_data, test_data, duration, output_path)
    # pass_rf_model = train_rf(train_data, test_data, crowdedness, output_path)
    pacing_rf_model = train_rf(train_data, test_data, pacing, output_path)
    speed_rf_model = train_rf(train_data, test_data, speed, output_path)

    # Gradient Boosted Trees
    # duration_gbt_model = train_gbt(train_data, test_data, duration, output_path)
    # pass_gbt_model = train_gbt(train_data, test_data, crowdedness, output_path)
    pacing_gbt_model = train_gbt(train_data, test_data, pacing, output_path)
    speed_gbt_model = train_gbt(train_data, test_data, speed, output_path)
