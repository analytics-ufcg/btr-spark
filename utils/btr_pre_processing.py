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
    grouping_set = ["DATE", "ROUTE", "BUS_CODE"]

    rdd = data_frame.rdd.map(
        lambda row: (tuple(row.asDict()[i] for i in row.asDict() if i in grouping_set),
                     {i: row.asDict()[i] for i in row.asDict() if i not in grouping_set})
    )

    return rdd


def combine_stops(key, values):
    trips = list(values)
    combined_trips = list()

    for i in range(len(trips)):
       for j in range(i + i, len(trips)):
           if trips[i]["STOP_ID"] == trips[j]["STOP_ID"]:
               break
           else:
               combined_trips.append((trips[i], trips[j]))

    return (key, combined_trips)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Error! Your command must be something like:"
        print "spark-submit %s <btr-input-path>" % (sys.argv[0])
        sys.exit(1)

    btr_input_path = sys.argv[1]

    sc = SparkContext("local[*]", appName="train_btr")
    sqlContext = pyspark.SQLContext(sc)

    trips_df = read_files(btr_input_path, sqlContext)

    stops_df = trips_df.na.drop(subset = ["STOP_ID"])

    stops_rdd = as_tuple_list(stops_df)

    combined_stops_rdd = stops_rdd.groupByKey().map(lambda (key, value): combine_stops(key, value))

    print combined_stops_rdd.take(20)

    print "==================================================="
    stops_df.show(5)
    print "==================================================="
