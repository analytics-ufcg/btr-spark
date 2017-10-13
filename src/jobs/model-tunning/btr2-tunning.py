import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# MLlib
from pyspark.ml.regression import LinearRegressionModel, LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

string_columns = ["periodOrig", "weekDay", "route"]
features=["shapeLatOrig", "shapeLonOrig", "busStopIdOrig",
            "busStopIdDest", "shapeLatDest", "shapeLonDest",
			"hourOrig", "isRushOrig", "weekOfYear", "dayOfMonth",
			"month", "isHoliday", "isWeekend", "isRegularDay", "distance"]
			
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 %s <training-data-path> " % (sys.argv[0])
        sys.exit(1)
		
    training_data_path = sys.argv[1]
	pipelineStages = []
			
    sc = SparkContext(appName="train_btr_2.0")
    sqlContext = pyspark.SQLContext(sc)


    training_raw = sqlContext.read.format("com.databricks.spark.csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("nullValue", "-")\
        .load(training_data_path)
	
	training = training_raw.withColumn("duration", training_raw.duration.cast('Double'))\
		.na.drop(subset = string_columns + features)
		
    indexers = [StringIndexer(inputCol = column, outputCol = column + "_index") for column in string_columns]
    assembler = VectorAssembler(
        inputCols = features + map(lambda c: c + "_index", string_columns),
        outputCol = 'features')
    duration_lr = LinearRegression(elasticNetParam=1.0, labelCol="duration")
	
	pipelineStages = pipelineStages + indexers
	pipelineStages.append(assembler)
	
	pipeline = Pipeline(stages = pipelineStages)
	
	train_data = pipeline.fit(training).transform(training)
	
	paramGrid = ParamGridBuilder() \
	.addGrid(duration_lr.maxIter, [10, 20]) \
    .addGrid(duration_lr.regParam, [0.1, 0.01]) \
    .build()
	
	crossval = CrossValidator(estimator=duration_lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="duration"),
                          numFolds=2)
						  
	cvModel = crossval.fit(train_data)
	
	cvModel.bestModel.write().overwrite().save(filepath)