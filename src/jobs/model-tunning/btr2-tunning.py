import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType

# MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

import sys
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

    save_test_info(cvModel.bestModel, test_data, label + "-lasso", file_to_save)

    return cvModel

def train_rf(train_data, test_data, label, file_to_save):
    route_arity = train_data.select('route').distinct().count()

    duration_rf = RandomForestRegressor(labelCol=label, featuresCol="features")

    #RandomForest.trainRegressor(train_data, {0:route_arity, 31:3, 33: 7}, 5).setLabelCol(label).setFeaturesCol("features")

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

    save_test_info(cvModel.bestModel, test_data, label + "-random-forest", file_to_save)

    return cvModel;

def train_gbt(train_data, test_data, label, file_to_save):
    route_arity = train_data.select('route').distinct().count()

    duration_gbt = GBTRegressor(labelCol=label, featuresCol="features")

    #GradientBoostedTrees.trainRegressor(train_data, {0:route_arity, 31:3, 33: 7}, numIterations=10).setLabelCol(label).setFeaturesCol("features")

    paramGrid = (ParamGridBuilder()
             .addGrid(duration_gbt.maxDepth, [2, 4, 6])
             .addGrid(duration_gbt.maxBins, [300])
             .build())

    crossval = CrossValidator(estimator=duration_gbt,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(labelCol=label),
                        numFolds=5)

    cvModel = crossval.fit(train_data)

    save_test_info(cvModel.bestModel, test_data, label + "-gbt", file_to_save)

    return cvModel;

# exemplo de chamada: save_train_info(cvModel, "Lasso", <nome do arquivo para salvar os resultados>)
def save_test_info(model, test_data, model_name, filepath):

    predictions, trainingSummary = getPredictionsLabels(model, test_data)

    predictions = predictions.select("duration", "probableNumPassengers", "prediction")

    print predictions.describe().show()

    # output = model_name + "\n"
    # with open(filepath + model_name, 'a') as outfile:
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

    predictions.write.csv(filepath + "predictions/" + model_name, mode="overwrite", header = True)

    model.write().overwrite().save(filepath + "model/" + model_name)

    result.write.text(filepath + "info/" + model_name)


def getPredictionsLabels(model, test_data):
    predictions = model.transform(test_data)

    trainingSummary = RegressionMetrics(predictions.rdd.map(lambda row: (row.prediction, row.duration)))

    return (predictions, trainingSummary)


def build_features_pipeline(string_columns = ["periodOrig", "weekDay", "route"],
                            features=["busStopIdOrig", "busStopIdDest", "shapeLatOrig", "shapeLonOrig", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "isHoliday", "isWeekend", "isRegularDay", "distance", "month", "weekOfYear", "dayOfMonth"]):

    pipelineStages = []

    indexers = [StringIndexer(inputCol = column, outputCol = column + "_index") for column in string_columns]

    assembler = VectorAssembler(
        inputCols = features + map(lambda c: c + "_index", string_columns),
        outputCol = 'features')

    pipelineStages = pipelineStages + indexers
    pipelineStages.append(assembler)

    pipeline = Pipeline(stages = pipelineStages)

    return pipeline

def is_train_or_test(month, dayOfMonth, train_start_date, train_end_date, test_end_date):
    if month >=  train_start_date.month & dayOfMonth >= train_start_date.day & month <= train_end_date.month & dayOfMonth <= train_end_date.day:
        return "train"
    elif month > train_end_date.month & dayOfMonth > train_end_date.day & month <= test_end_date.month & dayOfMonth <= test_end_date.day:
        return "test"
    else:
        return None

def split_partitions(df, train_start_date, train_end_date, test_end_date):
    udf_is_train_or_test = udf(is_train_or_test, StringType())
    df = df.withColumn("partition", udf_is_train_or_test("month", "dayOfMonth", train_start_date, train_end_date, test_end_date))
    return df

def read_data(datapath, string_columns = ["periodOrig", "weekDay", "route"],
                            features=["shapeLatOrig", "shapeLonOrig", "busStopIdOrig",
                             "busStopIdDest", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "weekOfYear", "dayOfMonth", "month", "isHoliday", "isWeekend", "isRegularDay", "distance"]):

    training_raw = sqlContext.read.csv(datapath, header=True, inferSchema=True, nullValue="-")

    training = training_raw.withColumn("duration", training_raw.duration.cast('Double'))\
    	.na.drop(subset = string_columns + features)

    return training

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <training-data-path> <train-start-date(YYYY-MM-DD)> " \
              "<train-end-date(YYYY-MM-DD)> <test-end-date(YYYY-MM-DD)> <tunning-info-path>" % (sys.argv[0])
        sys.exit(1)

    training_data_path = sys.argv[1]
    train_start_date = sys.argv[2]
    train_end_date = sys.argv[3]
    test_end_date = sys.argv[4]
    filepath = sys.argv[5]

    sc = SparkContext(appName="tunning_btr_2.0")
    global sqlContext
    sqlContext = pyspark.SQLContext(sc)

    raw_data = read_data(training_data_path)

    pipeline = build_features_pipeline(features=["busStopIdOrig", "busStopIdDest", "hourOrig",
                             "isRushOrig", "weekOfYear", "dayOfMonth", "month", "isHoliday", "isWeekend", "isRegularDay", "distance"])

    transformed_data = pipeline.fit(raw_data).transform(raw_data)

    train_data = transformed_data.filter("date > '" + train_start_date + "' and date < '" + train_end_date + "'")

    test_data = transformed_data.filter("date > '" + train_end_date + "' and date < '" + test_end_date + "'")


    #train_data = split_partitions(train_data, train_start_date, train_end_date, test_end_date)

    duration = "duration"
    crowdedness = "probableNumPassengers"

    # Lasso
    duration_lasso_model = train_lasso(train_data, test_data, duration, filepath)
    # pass_lasso_model = train_lasso(train_data, test_data, crowdedness, filepath)

    # Random Forest
    duration_rf_model = train_rf(train_data, test_data, duration, filepath)
    # pass_rf_model = train_rf(train_data, test_data, crowdedness, filepath)

    # Gradient Boosted Trees
    duration_gbt_model = train_gbt(train_data, test_data, duration, filepath)
    # pass_gbt_model = train_gbt(train_data, test_data, crowdedness, filepath)
