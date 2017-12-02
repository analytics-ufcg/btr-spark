#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
os.environ["PYSPARK_PYTHON"] = "python2.7"
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import lit, lead, lag, udf, unix_timestamp, hour, when, weekofyear, date_format, dayofmonth, month
import pyspark.sql.functions as func
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType

import random

def read_buste_data(filepath, sqlContext):
    data_frame = sqlContext.read.csv(filepath, header=True, inferSchema=True,nullValue="-")
    data_frame = data_frame.withColumn("date", data_frame.date.cast('String'))
    data_frame = data_frame.withColumnRenamed('stopPointId', 'busStopId')
    return data_frame

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

    df = df.withColumnRenamed("ext_num_pass", "crowdedness")

    # Extract total distance
    df = df.withColumn("distance", func.abs(df.distanceTraveledShapeDest - df.distanceTraveledShapeOrig))

    # Derive speed from one bus stop to another in m/s
    df = df.withColumn("speed", df.distance / df.duration)

    # Derive pacing in s/m, i.e., how many seconds a bus takes to move one meter
    df = df.withColumn("pacing", df.duration / df.distance)

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
    df = df.withColumn("weekDay", date_format('date', 'E'))
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

    df = df.withColumn("duration", df.duration.cast('Double'))

    return df

def extract_routes_stops(df, routes_stops_output_path):
    unique_stops_df = df.select("route", "shapeId", "busStopId", "distanceTraveledShape")\
        .distinct()\
        .orderBy("route", "shapeId", "distanceTraveledShape")

    unique_stops_df.write.csv(routes_stops_output_path, mode="overwrite", header=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Error! Your command must be something like:"
        print "spark-submit %s <btr-input-path> " \
              "<btr-pre-processing-output-folder>" % (sys.argv[0])
        sys.exit(1)

    btr_input_path = sys.argv[1]
    btr_pre_processing_output_path = sys.argv[2]

    btr_pre_processing_data_path = btr_pre_processing_output_path + "train_data"    
    routes_stops_output_path = btr_pre_processing_output_path + "routes_stops"
    btr_outliers_output = btr_pre_processing_output_path + "outliers"
    
    sc = SparkContext(appName="btr_pre_processing")
    sqlContext = pyspark.SQLContext(sc)

    trips_df = read_buste_data(btr_input_path, sqlContext)

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
    print stops_df.printSchema()
    print stops_df_lead.printSchema()

    stops_df_lead = extract_features(stops_df_lead)

    output = stops_df_lead.filter("duration < 1200 and duration > 0")

    outliers = stops_df_lead.filter("duration > 1199")

    output.write.csv(btr_pre_processing_data_path, mode="overwrite", header = True)

    outliers.write.csv(btr_outliers_output, mode="overwrite", header = True)

    sc.stop()
