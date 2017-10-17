import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

import sys

def train_lasso(train_data):
    duration_lr = LinearRegression(elasticNetParam=1.0, labelCol="duration")

    paramGrid = ParamGridBuilder() \
    .addGrid(duration_lr.maxIter, [10, 20]) \
    .addGrid(duration_lr.regParam, [0.1, 0.01]) \
    .build()

    crossval = CrossValidator(estimator=duration_lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(labelCol="duration"),
                              numFolds=2)

    cvModel = crossval.fit(train_data)

    return cvModel

def train_rf(train_data):
    route_arity = train_data.select('route').distinct().count()

    duration_rf = RandomForest.trainRegressor(train_data, {0:route_arity, 31:3, 33: 7}, 5)

    paramGrid = (ParamGridBuilder()
             .addGrid(duration_rf.maxDepth, [2, 4, 6])
             .addGrid(duration_rf.maxBins, [20, 60])
             .addGrid(duration_rf.numTrees, [5, 20])
             .build())

    crossval = CrossValidator(estimator=duration_rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(labelCol="duration"),
                        numFolds=5)

    cvModel = crossval.fit(train_data)
    return cvModel;

def train_gbt(train_data):
    route_arity = train_data.select('route').distinct().count()

    duration_gbt = GradientBoostedTrees.trainRegressor(train_data, {0:route_arity, 31:3, 33: 7}, numIterations=10)

    paramGrid = (ParamGridBuilder()
             .addGrid(duration_gbt.maxDepth, [2, 4, 6])
             .addGrid(duration_gbt.maxBins, [20, 60])
             .build())

    crossval = CrossValidator(estimator=duration_gbt,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(labelCol="duration"),
                        numFolds=5)

    cvModel = crossval.fit(train_data)
    return cvModel;

# exemplo de chamada: save_train_info(cvModel, "Lasso", <nome do arquivo para salvar os resultados>)
def save_train_info(model, model_name, filepath):
    output = model_name + "\n"
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

def getPredictionsLabels(model, test_data):
    predictions = model.transform(test_data)
    return predictions.rdd.map(lambda row: (row.prediction, row.duration))

def build_features_pipeline(string_columns = ["periodOrig", "weekDay", "route"],
                            features=["shapeLatOrig", "shapeLonOrig", "busStopIdOrig",
                             "busStopIdDest", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "weekOfYear", "dayOfMonth", "month", "isHoliday", "isWeekend", "isRegularDay", "distance"]):

    pipelineStages = []

    indexers = [StringIndexer(inputCol = column, outputCol = column + "_index") for column in string_columns]

    assembler = VectorAssembler(
        inputCols = features + map(lambda c: c + "_index", string_columns),
        outputCol = 'features')

    pipelineStages = pipelineStages + indexers
    pipelineStages.append(assembler)

    pipeline = Pipeline(stages = pipelineStages)

    return pipeline

def read_data(datapath, string_columns = ["periodOrig", "weekDay", "route"],
                            features=["shapeLatOrig", "shapeLonOrig", "busStopIdOrig",
                             "busStopIdDest", "shapeLatDest", "shapeLonDest", "hourOrig",
                             "isRushOrig", "weekOfYear", "dayOfMonth", "month", "isHoliday", "isWeekend", "isRegularDay", "distance"]):

    training_raw = sqlContext.read.format("com.databricks.spark.csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("nullValue", "-")\
        .load(datapath)

    training = training_raw.withColumn("duration", training_raw.duration.cast('Double'))\
    	.na.drop(subset = string_columns + features)

    return training

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print "Error: Wrong parameter specification!"
    #     print "Your command should be something like:"
    #     print "spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 %s <training-data-path> <train-start-date> <train-end-date> <test-end-date> <duration-model-path>" % (sys.argv[0])
    #     sys.exit(1)

    training_data_path = sys.argv[1]
    # train_start_date = sys.argv[2]
    # train_end_date = sys.argv[3]
    # test_end_date = sys.argv[4]
    # filepath = sys.argv[5]

    sc = SparkContext(appName="train_btr_2.0")
    sqlContext = pyspark.SQLContext(sc)

    raw_data = read_data(training_data_path)

    pipeline = build_features_pipeline()

    train_data = pipeline.fit(raw_data).transform(raw_data)

    bestModel = train_lasso(train_data).bestModel

    print bestModel.coefficients

    # A partir daqui deve-se rodar os modelos e salvar os resultados separadamente
    # model = train_lasso(train_data)
    #
    # model.bestModel.write().overwrite().save(filepath)
