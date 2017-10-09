#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from glob import glob
from os.path import isfile, join, splitext

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import lit, lead, lag, udf, unix_timestamp, hour, when, weekofyear, date_format, dayofmonth, month
import pyspark.sql.functions as func
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType

from datetime import datetime

import random

def read_file(filepath, sqlContext):
    data_frame = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "false") \
        .option("inferSchema", "true") \
        .option("nullValue", "-") \
        .load(filepath)

    while len(data_frame.columns) < 16:
        col_name = "_c" + str(len(data_frame.columns))
        data_frame = data_frame.withColumn(col_name, lit(None))

    data_frame = rename_columns(
        data_frame,
        [
            ("_c0", "route"),
            ("_c1", "tripNum"),
            ("_c2", "shapeId"),
            ("_c3", "shapeSequence"),
            ("_c4", "shapeLat"),
            ("_c5", "shapeLon"),
            ("_c6", "distanceTraveledShape"),
            ("_c7", "busCode"),
            ("_c8", "gpsPointId"),
            ("_c9", "gpsLat"),
            ("_c10", "gpsLon"),
            ("_c11", "distanceToShapePoint"),
            ("_c12", "timestamp"),
            ("_c13", "busStopId"),
            ("_c14", "problem"),
            ("_c15", "numPassengers")
        ]
    )

    date = "-".join(filepath.split("/")[-2].split("_")[:3])

    data_frame = data_frame.withColumn("date", lit(date))

    return data_frame

def read_files(path, sqlContext, sc):
    extension = splitext(path)[1]

    if extension == "":
        path_pattern = path + "/*/part-*"
        if "hdfs" in path:
            URI = sc._gateway.jvm.java.net.URI
            Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
            FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
            Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration

            hdfs = "/".join(path_pattern.split("/")[:3])
            dir = "/" + "/".join(path_pattern.split("/")[3:])

            fs = FileSystem.get(URI(hdfs), Configuration())

            status = fs.globStatus(Path(dir))

            files = map(lambda file_status: str(file_status.getPath()), status)

        else:
            files = glob(path_pattern)

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

def add_accumulated_passengers(df, window, probs = [0.025, 0.025, 0.05, 0.06, 0.075, 0.08, 0.11, 0.15, 0.11, 0.08, 0.075, 0.06, 0.05, 0.025, 0.025]):
    """
    :param df: Spark DataFrame
    :param window: Spark Window to iterate over
    :return: Spark DataFrame with accumulated number of passengers
    """

    df = df.withColumn("acumPassengers", func.sum(df.numPassengers).over(window))

    df = df.withColumn("probableNumPassengers", lit(0))
    for i in range(len(probs)):
        df = df.withColumn("probableNumPassengers", df.probableNumPassengers - lag(df.numPassengers, count=i + 1, default=0).over(window) * probs[i])

    df = df.withColumn("probableNumPassengers", df.numPassengers + df.probableNumPassengers)
    df = df.withColumn("probableNumPassengers", func.sum(df.probableNumPassengers).over(window))
    df = df.withColumn("probableNumPassengers", when(df.probableNumPassengers >= 0, df.probableNumPassengers).otherwise(0))

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

def weekday(date, fmt = "%Y-%m-%d"):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd = datetime.strptime(date, fmt).weekday()
    return days[wd]

def extract_features(df):
    print "Inicial features"
    print ["tripNumOrig", "route", "shapeId", "shapeSequence", "shapeLatOrig", "shapeLonOrig", "gpsPointId",
           "busCode", "timestampOrig", "gpsLat", "gpsLon", "distanceToShapePoint", "problem", "busStopIdOrig", "date",
           "busStopIdDest", "timestampDest", "tripNumDest", "shapeLatDest", "shapeLonDest", "distanceTraveledShapeOrig",
           "distanceTraveledShapeDest"]

    # Extract duration in seconds
    time_fmt = "HH:mm:ss"
    time_difference = unix_timestamp("timestampDest", time_fmt) - unix_timestamp("timestampOrig", time_fmt)
    df = df.withColumn("duration", time_difference)

    # Extract total distance
    df = df.withColumn("distance", func.abs(df.distanceTraveledShapeDest - df.distanceTraveledShapeOrig))
    # Extract hour
    df = df.withColumn("hourOrig", hour("timestampOrig"))
    # Extract hour
    df = df.withColumn("hourDest", hour("timestampDest"))
    # Extract is rush hour
    rush_hours_orig = when((df.hourOrig == 6) | (df.hourOrig == 7) | (df.hourOrig == 11) | (df.hourOrig == 12) |
                           (df.hourOrig == 17) | (df.hourOrig == 18), 1).otherwise(0)
    df = df.withColumn("isRushOrig", rush_hours_orig)
    # Extract is rush hour
    rush_hours_dest = when((df.hourDest == 6) | (df.hourDest == 7) | (df.hourDest == 11) | (df.hourDest == 12) |
                           (df.hourDest == 17) | (df.hourDest == 18), 1).otherwise(0)
    df = df.withColumn("isRushDest", rush_hours_dest)
    # Extract period of day
    period_orig = when(df.hourOrig < 12, "morning").otherwise(when((df.hourOrig >= 12) & (df.hourOrig < 18),
                                                                    "afternoon").otherwise("night"))
    df = df.withColumn("periodOrig", period_orig)
    # Extract period of day
    period_dest = when(df.hourDest < 12, "morning").otherwise(when((df.hourDest >= 12) & (df.hourDest < 18),
                                                                    "afternoon").otherwise("night"))
    df = df.withColumn("periodDest", period_dest)
    # Extract week day
    udf_weekday = udf(weekday, StringType())
    df = df.withColumn("weekDay", udf_weekday("date"))
    # Extract week number
    df = df.withColumn("weekOfYear", weekofyear("date"))
    # Extract day of month
    df = df.withColumn("dayOfMonth", dayofmonth("date"))
    # Extract month
    df = df.withColumn("month", month("date"))
    # Extract is holidays
    is_holiday = when((df.month == 1) | (df.month == 6) | (df.month == 7) | (df.month == 12), 1).otherwise(0)
    df = df.withColumn("isHoliday", is_holiday)
    # Extract is weekend
    is_weekend = when((df.weekDay == "Sat") | (df.weekDay == "Sun"), 1).otherwise(0)
    df = df.withColumn("isWeekend", is_weekend)
    # Extract is TUE, WED, THU
    is_regular_day = when((df.weekDay == "Tue") | (df.weekDay == "Wed") | (df.weekDay == "Thu"), 1).otherwise(0)
    df = df.withColumn("isRegularDay", is_regular_day)

    return df

def extract_routes_stops(df, routes_stops_output_path):
    unique_stops_df = df.select("route", "shapeId", "busStopId", "distanceTraveledShape")\
        .distinct()\
        .orderBy("route", "shapeId", "distanceTraveledShape")

    unique_stops_df.write.format("com.databricks.spark.csv") \
        .save(routes_stops_output_path, mode="overwrite", header=True)

def get_normal_distribution_list(mu, sigma, l_size):
    dist = list()
    for i in range(l_size):
        dist.append(random.gauss(mu, sigma))
    dist_sum = sum(dist)
    norm_dist = map(lambda v: v / dist_sum, dist)
    norm_dist.sort()
    norm_dist_sorted = [0] * l_size
    for i in range(l_size):
        if i % 2 == 0:
            norm_dist_sorted[i/2] = norm_dist[i]
        else:
            norm_dist_sorted[-1 * (i + 1) / 2] = norm_dist[i]
    return norm_dist_sorted

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Error! Your command must be something like:"
        print "spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 %s <btr-input-path> " \
              "<btr-pre-processing-output> <routes-stops-output-path>" % (sys.argv[0])
        sys.exit(1)

    btr_input_path = sys.argv[1]
    btr_pre_processing_output_path = sys.argv[2]
    routes_stops_output_path = sys.argv[3]

    sc = SparkContext("local[*]", appName="btr_pre_processing")
    sqlContext = pyspark.SQLContext(sc)

    trips_df = read_files(btr_input_path, sqlContext, sc)

    stops_df = trips_df.na.drop(subset=["busStopId"])

    extract_routes_stops(stops_df, routes_stops_output_path)

    w = Window().partitionBy("date", "route", "shapeId", "busCode", "tripNum").orderBy("timestamp")

    stops_df_lead = add_columns_lead(
        stops_df,
        [
            ("busStopId", "busStopIdDest"),
            ("timestamp", "timestampDest"),
            ("tripNum", "tripNumDest"),
            ("shapeLat", "shapeLatDest"),
            ("shapeLon", "shapeLonDest"),
            ("distanceTraveledShape", "distanceTraveledShapeDest")
        ],
        w
    )



    stops_df_lead = add_accumulated_passengers(
        stops_df_lead,
        w,
        probs=get_normal_distribution_list(40, 10, 66)
    )

    stops_df_lead = rename_columns(
        stops_df_lead,
        [
            ("busStopId", "busStopIdOrig"),
            ("timestamp", "timestampOrig"),
            ("tripNum", "tripNumOrig"),
            ("shapeLat", "shapeLatOrig"),
            ("shapeLon", "shapeLonOrig"),
            ("distanceTraveledShape", "distanceTraveledShapeOrig")
        ]
    )

    stops_df_lead = stops_df_lead.na.drop(subset=["busStopIdDest"])

    print stops_df.show(10)

    print stops_df_lead.show(10)

    stops_df_lead = extract_features(stops_df_lead)

    stops_df_lead.write.format("com.databricks.spark.csv")\
        .save(btr_pre_processing_output_path, mode="overwrite", header = True)

    sc.stop()
