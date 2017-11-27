import pyspark
from pyspark import SparkContext
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

import json
import numpy as np
import sys

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

def read_buste_data_v3(sqlContext, filepath):
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

def printdf(df,l=10):
    return df.limit(l).toPandas()

def buildODMatrix(buste_data):

	# buste_data = read_hdfs_folder(sqlContext, datapath + '/bulma-output/2017_05_11_veiculos/')

	clean_buste_data = buste_data.na.drop(subset=["date","route","busCode","tripNum","busStopId","timestamp","shapeLon","shapeLat"])

	filtered_boardings = clean_buste_data.na.drop(subset=['cardNum','cardTimestamp']).dropDuplicates(['cardNum','date','cardTimestamp'])

	boarding_count = filtered_boardings.groupby('cardNum').count()

	multiple_boardings = boarding_count.filter(F.col('count') > 1)

	multiple_boardings = multiple_boardings.select(F.col("cardNum").alias("cardNum1"), F.col("count").alias("count1"))

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
	                                    F.col("busStopId").alias("o_stop_id"),
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
	                                    F.col("busStopId").alias("next_o_stop_id"),
	                                    F.col("boarding_id").alias("next_o_boarding_id"))


	user_trips = origin_matrix.join(od_matrix_ids, origin_matrix.o_boarding_id == od_matrix_ids.boarding_id, 'inner').join(next_origin_matrix, od_matrix_ids.next_boarding_id == next_origin_matrix.next_o_boarding_id, 'inner').drop('boarding_id').drop('next_boarding_id')

	user_trips = user_trips.withColumn('o_unixtimestamp',F.unix_timestamp(F.col('o_timestamp'), 'HH:mm:ss')).withColumn('next_o_unixtimestamp',F.unix_timestamp(F.col('next_o_timestamp'), 'HH:mm:ss'))

	user_trips = user_trips.withColumn('leg_duration',F.when(F.col('next_o_unixtimestamp') > F.col('o_unixtimestamp'),((F.col('next_o_unixtimestamp') - F.col('o_unixtimestamp'))/60.0)).otherwise(-1))

	user_trips_data = user_trips.withColumn('o_date',F.from_unixtime(F.unix_timestamp(F.col('o_date'),'yyyy-MM-dd'), 'yyyy-MM-dd')).withColumn('next_o_date',F.from_unixtime(F.unix_timestamp(F.col('next_o_date'),'yyyy-MM-dd'), 'yyyy-MM-dd')).orderBy(['cardNum','o_date','o_timestamp'])

	bus_trip_data = clean_buste_data.orderBy(['route','busCode','tripNum','timestamp']) \
	                            .dropDuplicates(['route','busCode','tripNum','busStopId']) \
	                            .drop('cardNum') \
	                            .withColumn('id',F.monotonically_increasing_id()) \
	                            .withColumn('route', F.col('route').cast(T.IntegerType())) \
	                            .withColumnRenamed('','cardNum')


	cond = [bus_trip_data.route == user_trips_data.o_route, 
	        bus_trip_data.busCode == user_trips_data.o_bus_code, 
	        bus_trip_data.date == user_trips_data.o_date,
	        bus_trip_data.tripNum == user_trips_data.o_tripNum]
	od_trips = bus_trip_data.join(user_trips_data, cond, 'left_outer')

	trips_with_boardings = od_trips

	trips_with_boardings = trips_with_boardings.withColumn('dist',dist(F.col('shapeLat'),F.col('shapeLon'),F.col('next_o_shape_lat'),F.col('next_o_shape_lon')))

	w = Window().partitionBy(['cardNum','date','route','busCode','tripNum']).orderBy('dist')

	od_matrix = trips_with_boardings.filter('timestamp > o_timestamp') \
	                    .withColumn('rn', F.row_number().over(w)) \
	                    .where(F.col('rn') == 1)

	filtered_od_matrix = od_matrix.filter('dist <= 1.0')

	trips_origins = filtered_od_matrix.filter(filtered_od_matrix.cardNum.isNotNull()) \
	                            .select(['o_date','o_route','o_bus_code','o_tripNum','o_stop_id','o_timestamp']) \
	                            .groupBy(['o_date','o_route','o_bus_code','o_tripNum','o_stop_id']) \
	                            .count() \
	                            .withColumnRenamed('count','boarding_cnt') \
	                            .orderBy(['o_date','o_route','o_bus_code','o_tripNum'])

	trips_destinations = filtered_od_matrix.filter(filtered_od_matrix.cardNum.isNotNull()) \
	                            .select(['date','route','busCode','tripNum','busStopId','timestamp']) \
	                            .groupBy(['date','route','busCode','tripNum','busStopId']) \
	                            .count() \
	                            .orderBy(['date','route','busCode','tripNum'])

	trips_destinations = rename_columns(
	                        trips_destinations,
	                        [('date','d_date'),
	                         ('route','d_route'),
	                         ('busCode','d_bus_code'),
	                         ('tripNum','d_tripNum'),
	                         ('busStopId','d_stop_id'),
	                         ('count','alighting_cnt')])

	origin_cond = [bus_trip_data.date == trips_origins.o_date,
	               bus_trip_data.route == trips_origins.o_route, 
	               bus_trip_data.busCode == trips_origins.o_bus_code, 
	               bus_trip_data.tripNum == trips_origins.o_tripNum,
	               bus_trip_data.busStopId == trips_origins.o_stop_id]

	dest_cond = [bus_trip_data.date == trips_destinations.d_date,
	               bus_trip_data.route == trips_destinations.d_route, 
	               bus_trip_data.busCode == trips_destinations.d_bus_code, 
	               bus_trip_data.tripNum == trips_destinations.d_tripNum,
	               bus_trip_data.busStopId == trips_destinations.d_stop_id]

	buste_crowdedness = bus_trip_data.join(trips_origins,origin_cond,'left_outer') \
	                        .join(trips_destinations,dest_cond,'left_outer') \
	                        .drop('o_date','o_route','o_bus_code','o_tripNum','o_stop_id') \
	                        .drop('d_date','d_route','d_bus_code','d_tripNum','d_stop_id') \
	                        .withColumn('boarding_cnt',F.when(F.col('boarding_cnt').isNull(),F.lit(0)).otherwise(F.col('boarding_cnt'))) \
	                        .withColumn('alighting_cnt',F.when(F.col('alighting_cnt').isNull(),F.lit(0)).otherwise(F.col('alighting_cnt')))

	trips_window = Window.partitionBy(['date','route','busCode','tripNum']).orderBy('timestamp')

	buste_crowdedness = buste_crowdedness.withColumn('crowd_bal', F.col('boarding_cnt') - F.col('alighting_cnt')) \
	                        .withColumn('num_pass',F.sum('crowd_bal').over(trips_window)) \
	                        .drop('numPassengers','gps_timestamp','gps_timestamp_in_secs') \
	                        .orderBy(['date','route','busCode','tripNum','timestamp'])


	boarding_data_route_boarding = bus_trip_data.groupby(['route']).count() \
	                                .withColumnRenamed('count','overall_boarding')

	od_matrix_route_boarding = filtered_od_matrix.groupby(['route']).count() \
	                            .withColumnRenamed('count','odmatrix_boarding')

	od_matrix_route_prop = boarding_data_route_boarding.join(od_matrix_route_boarding, 'route','left_outer') \
	                        .withColumn('extrap_factor',F.when(((F.col('odmatrix_boarding') == 0) | (F.col('odmatrix_boarding').isNull())), 0.0) \
	                                    .otherwise(F.col('overall_boarding').cast('float')/F.col('odmatrix_boarding')))

	buste_crowdedness_extrapolated = buste_crowdedness.join(od_matrix_route_prop, 'route', 'left') \
	                                .drop('overall_boarding','odmatrix_boarding') \
	                                .withColumn('ext_num_pass', F.col('num_pass')*F.col('extrap_factor'))

	# buste_crowdedness_extrapolated.write.csv(path=datapath+'/buste_crowdedness',header=True, mode='overwrite')

	return buste_crowdedness_extrapolated

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Error: Wrong parameter specification!"
        print "Your command should be something like:"
        print "spark-submit %s <output-folder-path>" % (sys.argv[0])
        sys.exit(1)

    output_folder = sys.argv[1]

    sc = SparkContext(appName="OD matrix Builder")
    sqlContext = pyspark.SQLContext(sc)
    
    buildODMatrix(output_folder)

    sc.stop()
