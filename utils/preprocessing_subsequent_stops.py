#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join, splitext

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import lit, lead, udf, unix_timestamp, hour, when, weekofyear, date_format, dayofmonth, month
import pyspark.sql.functions as func
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType

from datetime import datetime

def read_file(filepath, sqlContext):
    data_frame = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("nullValue", "-") \
        .load(filepath)

    date = "-".join(filepath.split("/")[-1].split("_")[:3])

    data_frame = data_frame.withColumn("date", lit(date))

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

    stops_df = trips_df.na.drop(subset=["busStopId"])

    w = Window().partitionBy("date", "route", "shapeId", "busCode").orderBy("tripNum", "timestamp")

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

    stops_df_lead = extract_features(stops_df_lead)

    stops_df_lead.write.format("com.databricks.spark.csv")\
        .save(btr_pre_processing_output, mode="overwrite", header = True)

    sc.stop()