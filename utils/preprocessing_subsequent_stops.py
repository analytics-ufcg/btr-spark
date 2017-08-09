#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join, splitext

import pyspark
from pyspark import SparkContext
from pyspark.sql.functions import lit, lead, udf, unix_timestamp, hour, when, weekofyear, date_format, dayofmonth, month
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType

from math import radians, cos, sin, asin, sqrt
from datetime import datetime

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

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def weekday(date, fmt = "%Y-%m-%d"):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd = datetime.strptime(date, fmt).weekday()
    return days[wd]

def extract_features(df):
    print "Inicial features"
    print ["TRIP_NUM_ORIG", "ROUTE", "SHAPE_ID", "SHAPE_SEQ", "LAT_SHAPE_ORIG", "LON_SHAPE_ORIG", "GPS_POINT_ID",
           "BUS_CODE", "TIMESTAMP_ORIG", "LAT_GPS", "LON_GPS", "DISTANCE_GPS_SHAPE", "THRESHOLD_PROBLEM",
           "TRIP_PROBLEM", "STOP_ID_ORIG", "DATE", "STOP_ID_DEST", "TIMESTAMP_DEST", "TRIP_NUM_DEST", "LAT_SHAPE_DEST",
           "LON_SHAPE_DEST"]

    # Extract duration in seconds
    time_fmt = "HH:mm:ss"
    time_difference = unix_timestamp("TIMESTAMP_DEST", time_fmt) - unix_timestamp("TIMESTAMP_ORIG", time_fmt)
    df = df.withColumn("DURATION", time_difference)

    # Extract hour
    df = df.withColumn("HOUR_ORIG", hour("TIMESTAMP_ORIG"))
    # Extract hour
    df = df.withColumn("HOUR_DEST", hour("TIMESTAMP_DEST"))
    # Extract is rush hour
    rush_hours_orig = when((df.HOUR_ORIG == 6) | (df.HOUR_ORIG == 7) | (df.HOUR_ORIG == 11) | (df.HOUR_ORIG == 12) | (df.HOUR_ORIG == 17) | (df.HOUR_ORIG == 18), 1).otherwise(0)
    df = df.withColumn("IS_RUSH_ORIG", rush_hours_orig)
    # Extract is rush hour
    rush_hours_dest = when((df.HOUR_DEST == 6) | (df.HOUR_DEST == 7) | (df.HOUR_DEST == 11) | (df.HOUR_DEST == 12) | (df.HOUR_DEST == 17) | (df.HOUR_DEST == 18), 1).otherwise(0)
    df = df.withColumn("IS_RUSH_DEST", rush_hours_dest)
    # Extract period of day
    period_orig = when(df.HOUR_ORIG < 12, "morning").otherwise(when((df.HOUR_ORIG >= 12) & (df.HOUR_ORIG < 18), "afternoon").otherwise("night"))
    df = df.withColumn("PERIOD_ORIG", period_orig)
    # Extract period of day
    period_dest = when(df.HOUR_DEST < 12, "morning").otherwise(when((df.HOUR_DEST >= 12) & (df.HOUR_DEST < 18), "afternoon").otherwise("night"))
    df = df.withColumn("PERIOD_DEST", period_dest)
    # Extract week day
    udf_weekday = udf(weekday, StringType())
    df = df.withColumn("WEEK_DAY", udf_weekday("DATE"))
    # Extract week number
    df = df.withColumn("WEEK_OF_YEAR", weekofyear("DATE"))
    # Extract day of month
    df = df.withColumn("DAY_OF_MONTH", dayofmonth("DATE"))
    # Extract month
    df = df.withColumn("MONTH", month("DATE"))
    # Extract is holidays
    is_holiday = when((df.MONTH == 1) | (df.MONTH == 6) | (df.MONTH == 7) | (df.MONTH == 12), 1).otherwise(0)
    df = df.withColumn("IS_HOLIDAY", is_holiday)
    # Extract is weekend
    is_weekend = when((df.WEEK_DAY == "Sat") | (df.WEEK_DAY == "Sun"), 1).otherwise(0)
    df = df.withColumn("IS_WEEKEND", is_weekend)
    # Extract is TUE, WED, THU
    is_regular_day = when((df.WEEK_DAY == "Tue") | (df.WEEK_DAY == "Wed") | (df.WEEK_DAY == "Thu"), 1).otherwise(0)
    df = df.withColumn("IS_REGULAR_DAY", is_regular_day)
    # Extract total distance
    haversine_udf = udf(haversine, DoubleType())
    df = df.withColumn("TOTAL_DISTANCE", haversine_udf("LON_SHAPE_ORIG", "LAT_SHAPE_ORIG", "LON_SHAPE_DEST", "LAT_SHAPE_DEST"))

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
            ("LON_SHAPE", "LON_SHAPE_DEST")
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
            ("DISTANCE", "DISTANCE_GPS_SHAPE")
        ]
    )

    stops_df_lead = stops_df_lead.na.drop(subset=["STOP_ID_DEST"])

    print stops_df.show(10)

    print stops_df_lead.show(10)

    stops_df_lead = extract_features(stops_df_lead)

    #stops_df_lead.write.format("com.databricks.spark.csv").save(btr_pre_processing_output)

    sc.stop()