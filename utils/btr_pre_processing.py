import sys
import os.path

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

def read_files(path, sqlContext):
    extension = os.path.splitext(path)[1]

    data_frame = sqlContext.read.format("com.databricks.spark.csv")\
            .option("header", "true")\
            .option("inferSchema","true")\
            .option("nullValue", "-")

    if extension == "":
        return data_frame\
            .load(path + "/*.csv")
    else:
        return data_frame\
            .load(path)

def as_tuple_list(data_frame):
    rdd = data_frame.rdd.map(
        lambda row: (row.asDict().get("SHAPE_ID"), {i: row.asDict()[i] for i in row.asDict() if i != "SHAPE_ID"})
    )

    return rdd

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Error! Your command must be something like:"
        print "spark-submit %s <btr-input-path>" % (sys.argv[0])
        sys.exit(1)

    btr_input_path = sys.argv[1]

    sc = SparkContext("local[*]", appName="train_btr")
    sqlContext = pyspark.SQLContext(sc)

    data_frame = read_files(btr_input_path, sqlContext)

    print "==================================================="
    data_frame.show(5)
    print "==================================================="


    print as_tuple_list(data_frame).take(5)


