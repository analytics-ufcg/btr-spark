import pyspark
from pyspark import SparkContext
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from os.path import isfile, join, splitext
from glob import glob
from datetime import datetime

import json
import numpy as np
import sys

def read_files(path, sqlContext, sc, initial_date, final_date):
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

        print files

        files = filter(lambda f: initial_date <= datetime.strptime(f.split("/")[-2], '%Y_%m_%d_veiculos') <=
                                 final_date, files)

        print files

        return reduce(lambda df1, df2: df1.unionAll(df2),
                      map(lambda f: read_buste_data_v3(f, sqlContext), files))
    else:
        return read_file(path, sqlContext)

def get_files(path, sqlContext, sc, initial_date, final_date):

	path_pattern = path + "/*"
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

	files = filter(lambda f: initial_date <= datetime.strptime(f.split("/")[-1], '%Y_%m_%d_veiculos') <=
                                 final_date, files)

	return files

def rename_columns(df, list_of_tuples):
    for (old_col, new_col) in list_of_tuples:
        df = df.withColumnRenamed(old_col, new_col)
    return df

def read_hdfs_folder(sqlContext, folderpath):
	data_frame = sqlContext.read.csv(folderpath, header=True, inferSchema=True, nullValue="-")
	data_frame = rename_columns(data_frame, [("cardNum18", "cardNum"), ("cardNum19", "userGender"),])
	date = "-".join(folderpath.split("/")[-2].split("_")[:3])
	data_frame = data_frame.withColumn("date", F.lit(date))
	data_frame = data_frame.withColumn("date", F.date_sub(F.col("date"),1))
	return data_frame

def read_buste_data_v3( filepath, sqlContext):
    data_frame = sqlContext.read.csv(filepath, header=True, inferSchema=True,nullValue="-")    
    date = "-".join(filepath.split("/")[-2].split("_")[:3])
    data_frame = data_frame.withColumn("date", F.lit(date))
    data_frame = data_frame.withColumn("date", F.unix_timestamp(F.date_sub(F.col("date"),1),'yyyy-MM-dd'))    
    return data_frame

def dist(lat_x, long_x, lat_y, long_y):
    return F.acos(
        F.sin(F.toRadians(lat_x)) * F.sin(F.toRadians(lat_y)) + 
        F.cos(F.toRadians(lat_x)) * F.cos(F.toRadians(lat_y)) * 
            F.cos(F.toRadians(long_x) - F.toRadians(long_y))
    ) * F.lit(6371.0)


def buildODMatrix(buste_data, datapath, filepath):

	# buste_data = read_files(sqlContext, datapath + "/2017_06_21_veiculos/")

	clean_buste_data = buste_data.na.drop(subset=["date","route","busCode","tripNum","stopPointId","timestamp","shapeLon","shapeLat"])

	filtered_boardings = clean_buste_data.na.drop(subset=['cardNum','cardTimestamp']).dropDuplicates(['cardNum','date','cardTimestamp'])
	
	multiple_boardings = filtered_boardings.groupby('cardNum').count().filter(F.col('count') > 1) \
	                        .select(F.col("cardNum").alias("cardNum1"), F.col("count").alias("count1"))

	clean_boardings = filtered_boardings.join(multiple_boardings, filtered_boardings.cardNum == multiple_boardings.cardNum1, 'leftsemi')

	boarding_data = clean_boardings.withColumn('boarding_id',F.monotonically_increasing_id())

	user_boarding_w = Window.partitionBy(boarding_data.cardNum, boarding_data.date).orderBy(boarding_data.cardTimestamp)

	od_matrix_ids = boarding_data.select(F.col('cardNum'),
						F.col('boarding_id'), 
						F.lead('boarding_id',default=-1).over(user_boarding_w).alias('next_boarding_id'),
						F.first('boarding_id',True).over(user_boarding_w).alias('first_boarding')).withColumn('next_boarding_id', 
							F.when(F.col('next_boarding_id') == -1,F.col('first_boarding'))
							.otherwise(F.col('next_boarding_id'))).drop('first_boarding')


	origin_matrix = boarding_data.select(F.col("route").alias("o_route"),
	                                    F.col("busCode").alias("o_bus_code"),
	                                    F.col("date").alias("o_date"),
	                                    F.col("tripNum").alias("o_tripNum"),
	                                    F.col("cardTimestamp").alias("o_timestamp"),
	                                    F.col("shapeId").alias("o_shape_id"),
	                                    F.col("shapeSequence").alias("o_shape_seq"),
	                                    F.col("shapeLat").alias("o_shape_lat"),
	                                    F.col("shapeLon").alias("o_shape_lon"),
	                                    F.col("stopPointId").alias("o_stop_id"),
	                                    F.col("boarding_id").alias("o_boarding_id"))


	next_origin_matrix = boarding_data.select(F.col("route").alias("next_o_route"),
	                                    F.col("busCode").alias("next_o_bus_code"),
	                                    F.col("date").alias("next_o_date"),
	                                    F.col("tripNum").alias("next_o_tripNum"),
	                                    F.col("cardTimestamp").alias("next_o_timestamp"),
	                                    F.col("shapeId").alias("next_o_shape_id"),
	                                    F.col("shapeSequence").alias("next_o_shape_seq"),
	                                    F.col("shapeLat").alias("next_o_shape_lat"),
	                                    F.col("shapeLon").alias("next_o_shape_lon"),
	                                    F.col("stopPointId").alias("next_o_stop_id"),
	                                    F.col("boarding_id").alias("next_o_boarding_id"))



	user_trips_data = origin_matrix.join(od_matrix_ids, origin_matrix.o_boarding_id == od_matrix_ids.boarding_id, 'inner') \
                                .join(next_origin_matrix, od_matrix_ids.next_boarding_id == next_origin_matrix.next_o_boarding_id, 'inner') \
                                .drop('boarding_id').drop('next_boarding_id') \
                                .withColumn('o_unixtimestamp',F.unix_timestamp(F.col('o_timestamp'), 'HH:mm:ss')) \
                                .withColumn('next_o_unixtimestamp',F.unix_timestamp(F.col('next_o_timestamp'), 'HH:mm:ss')) \
                                .withColumn('leg_duration',F.when(F.col('next_o_unixtimestamp') > F.col('o_unixtimestamp'), \
		                        ((F.col('next_o_unixtimestamp') - F.col('o_unixtimestamp'))/60.0)).otherwise(-1)) \
		                        .orderBy(['cardNum','o_date','o_timestamp'])
	                            # .withColumn('o_date',F.from_unixtime(F.unix_timestamp(F.col('o_date'),'yyyy-MM-dd'), 'yyyy-MM-dd'))\
	                            # .withColumn('next_o_date',F.from_unixtime(F.unix_timestamp(F.col('next_o_date'),'yyyy-MM-dd'), 'yyyy-MM-dd')) \

	bus_trip_data = clean_buste_data.orderBy(['route','busCode','tripNum','timestamp']) \
	                            .dropDuplicates(['route','busCode','tripNum','stopPointId']) \
	                            .drop('cardNum') \
	                            .withColumn('id',F.monotonically_increasing_id()) \
	                            .withColumn('route', F.col('route').cast(T.IntegerType())) \
	                            .withColumnRenamed('','cardNum')


	cond = [bus_trip_data.route == user_trips_data.o_route, 
	        bus_trip_data.busCode == user_trips_data.o_bus_code, 
	        bus_trip_data.date == user_trips_data.o_date,
	        bus_trip_data.tripNum == user_trips_data.o_tripNum]

	w = Window().partitionBy(['cardNum','date','route','busCode','tripNum']).orderBy('dist')

	filtered_od_matrix = bus_trip_data.join(user_trips_data, cond, 'left_outer') \
	                            .withColumn('dist',dist(F.col('shapeLat'),F.col('shapeLon'),F.col('next_o_shape_lat'),F.col('next_o_shape_lon'))) \
	                            .filter('timestamp > o_timestamp') \
	                            .withColumn('rn', F.row_number().over(w)) \
	                            .where(F.col('rn') == 1) \
	                            .filter('dist <= 1.0') \
	                            .filter(user_trips_data.cardNum.isNotNull())

	#filtered_od_matrix.write.csv(path=tmppath+'/od/filtered_od',header=True, mode='append')

	trips_origins = filtered_od_matrix \
	                            .select(['o_date','o_route','o_bus_code','o_tripNum','o_stop_id','o_timestamp']) \
	                            .groupBy(['o_date','o_route','o_bus_code','o_tripNum','o_stop_id']) \
	                            .count() \
	                            .withColumnRenamed('count','boarding_cnt') \
	                            .withColumnRenamed('o_date','date') \
	                            .withColumnRenamed('o_route','route') \
	                            .withColumnRenamed('o_bus_code','busCode') \
	                            .withColumnRenamed('o_tripNum','tripNum') \
	                            .withColumnRenamed('o_stop_id','stopPointId')


	trips_destinations = filtered_od_matrix \
	                            .select(['date','route','busCode','tripNum','stopPointId','timestamp']) \
	                            .groupBy(['date','route','busCode','tripNum','stopPointId']) \
	                            .count() \
                                .withColumnRenamed('count','alighting_cnt') 

	trips_origins.write.csv(path=datapath+'/od/trips_origins/' + filepath,header=True, mode='overwrite')
	trips_destinations.write.csv(path=datapath+'/od/trips_destinations' + filepath,header=True, mode='overwrite')

	trips_o = sqlContext.read.csv(datapath + '/od/trips_origins' + filepath, header=True,inferSchema=True,nullValue="-")
	trips_d = sqlContext.read.csv(datapath + '/od/trips_destinations' + filepath, header=True,inferSchema=True,nullValue="-")

	trips_passengers = trips_o.join(trips_d, on = ['date','route','busCode','tripNum','stopPointId'], how='outer')

	trips_window = Window.partitionBy(['date','route','busCode','tripNum']).orderBy('timestamp')

	od_matrix_route_boarding = filtered_od_matrix.groupby(['route']).count() \
	                            .withColumnRenamed('count','odmatrix_boarding')

	od_matrix_route_prop = bus_trip_data.groupby(['route']).count() \
	                                .withColumnRenamed('count','overall_boarding') \
	                                .join(od_matrix_route_boarding, 'route','left_outer') \
	                                .withColumn('extrap_factor',F.when(((F.col('odmatrix_boarding') == 0) | (F.col('odmatrix_boarding').isNull())), 0.0) \
	                                .otherwise(F.col('overall_boarding').cast('float')/F.col('odmatrix_boarding')))

	buste_crowdedness_extrapolated = bus_trip_data.join(trips_passengers, on=['date','route','busCode','tripNum','stopPointId'], how='left_outer') \
	                        .withColumn('crowd_bal', F.col('boarding_cnt') - F.col('alighting_cnt')) \
	                        .withColumn('num_pass',F.sum('crowd_bal').over(trips_window)) \
	                        .drop('numPassengers','gps_timestamp','gps_timestamp_in_secs') \
	                        .orderBy(['date','route','busCode','tripNum','timestamp']) \
	                        .join(od_matrix_route_prop, 'route', 'left') \
	                        .drop('overall_boarding','odmatrix_boarding') \
	                        .withColumn('ext_num_pass', F.col('num_pass')*F.col('extrap_factor'))


	buste_crowdedness_extrapolated.write.csv(path=datapath + '/od/buste_crowdedness/' + filepath,header=True, mode='overwrite')

	# return buste_crowdedness_extrapolated


def execute_job(input_folder, sqlContext, sc, initial_date, final_date):

	files = get_files(input_folder, sqlContext, sc, initial_date, final_date)

	for file in files:
		data = read_buste_data_v3(file, sqlContext)

		buildODMatrix(data, output_folder, file)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <input-folder-path> <output-folder-path> \
                <initial_date> <final_date>" % (sys.argv[0])
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    initial_date = datetime.strptime(sys.argv[3], '%Y-%m-%d')
    final_date = datetime.strptime(sys.argv[4], '%Y-%m-%d')

    global sc, sqlContext

    sc = SparkContext(appName="OD matrix Builder")
    sqlContext = pyspark.SQLContext(sc)

    execute_job(input_folder, sqlContext, sc, initial_date, final_date)

    sc.stop()
