#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join, splitext

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import lit

def read_file(filepath, sqlContext):
    data_frame = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("nullValue", "-") \
        .load(filepath)

    # adicionar data
    date = "-".join(filepath.split("/")[-1].split("_")[:3])

    data_frame = data_frame.withColumn("DATE", lit(date))

    return data_frame

def read_files(path, sqlContext):
    extension = splitext(path)[1]

    if extension == "":
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        return reduce(lambda df1, df2: df1.unionAll(df2),
                      map(lambda f: read_file(f, sqlContext), files))
    else:
        return read_file(path, sqlContext)

def as_tuple_list(data_frame):
    # rota, data, veículo
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


    #print as_tuple_list(data_frame).take(5)


