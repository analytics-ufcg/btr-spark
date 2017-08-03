#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join, splitext

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import lead
from pyspark.sql.window import Window

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

def add_columns_lead(df, list_of_tuples, window):
    """
    :param df: Spark DataFrame

    :param list_of_tuples:
    Ex:

    [
        ("old_column_1", "new_column_1"),
        ("old_column_2", "new_column_2"),
        ("old_column_3", "new_column_3")
    ]

    :param window: Spark Window to iterate over

    :return: Spark DataFrame with new columns
    """

    for (old_col, new_col) in list_of_tuples:
        df = df.withColumn(new_col, lead(old_col).over(window))

    return df

def rename_columns(df, list_of_tuples):
    """
    :param df: Spark DataFrame

    :param list_of_tuples:
    Ex:

    [
        ("old_column_1", "new_column_1"),
        ("old_column_2", "new_column_2"),
        ("old_column_3", "new_column_3")
    ]

    :return: Spark DataFrame columns renamed
    """

    for (old_col, new_col) in list_of_tuples:
        df = df.withColumnRenamed(old_col, new_col)

    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Error! Your command must be something like:"
        print "spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 %s <btr-input-path> " \
              "<btr-pre-processing-output>" % (sys.argv[0])
        sys.exit(1)

    btr_input_path = sys.argv[1]
    btr_pre_processing_output = sys.argv[2]

    sc = SparkContext("local[*]", appName="btr_pre_processing")
    sqlContext = pyspark.SQLContext(sc)

    trips_df = read_files(btr_input_path, sqlContext)

    stops_df = trips_df.na.drop(subset=["STOP_ID"])

    w = Window().partitionBy("DATE", "ROUTE", "SHAPE_ID", "BUS_CODE").orderBy("TRIP_NUM", "TIMESTAMP")

    stops_df_lead = add_columns_lead(
        stops_df,
        [
            ("STOP_ID", "STOP_ID_DEST"),
            ("TIMESTAMP", "TIMESTAMP_DEST"),
            ("TRIP_NUM", "TRIP_NUM_DEST"),
            ("LAT_SHAPE", "LAT_SHAPE_DEST"),
            ("LON_SHAPE", "LON_SHAPE_DEST"),
            ("DISTANCE", "DISTANCE_DEST")
        ],
        w
    )

    stops_df_lead = rename_columns(
        stops_df_lead,
        [
            ("STOP_ID", "STOP_ID_ORIG"),
            ("TIMESTAMP", "TIMESTAMP_ORIG"),
            ("TRIP_NUM", "TRIP_NUM_ORIG"),
            ("LAT_SHAPE", "LAT_SHAPE_ORIG"),
            ("LON_SHAPE", "LON_SHAPE_ORIG"),
            ("DISTANCE", "DISTANCE_ORIG")
        ]
    )

    stops_df_lead = stops_df_lead.na.drop(subset=["STOP_ID_DEST"])

    print stops_df.show(10)

    print stops_df_lead.show(10)

    stops_df_lead.write.format("com.databricks.spark.csv").save(btr_pre_processing_output)

    sc.stop()