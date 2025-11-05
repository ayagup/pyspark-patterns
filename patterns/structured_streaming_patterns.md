### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:39:17 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Structured Streaming**
> *   Windowed Aggregations on Streams
> *   Stateful Stream Processing
> *   Watermarking for Late Data
> *   Joining Streams with Static DataFrames
> *   Joining Streams with Streams
> *   Different Output Modes (Append, Complete, Update)
> *   Idempotent Sinks (`foreachBatch`)
> *   Checkpointing
> ```

#### _Assistant_

> I'll generate comprehensive PySpark Structured Streaming examples for each pattern.
> 
> ## 1. Windowed Aggregations on Streams
> 
> ```python windowed_aggregations.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> 
> spark = SparkSession.builder \
>     .appName("WindowedAggregations") \
>     .getOrCreate()
> 
> # Read streaming data (simulating IoT sensor data)
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 10) \
>     .load() \
>     .withColumn("sensor_id", (rand() * 5).cast("int")) \
>     .withColumn("temperature", (rand() * 40 + 10)) \
>     .withColumnRenamed("timestamp", "event_time")
> 
> # Windowed aggregations
> # Tumbling window (non-overlapping)
> tumbling_window = streaming_df \
>     .groupBy(
>         window("event_time", "1 minute"),
>         "sensor_id"
>     ) \
>     .agg(
>         avg("temperature").alias("avg_temp"),
>         max("temperature").alias("max_temp"),
>         count("*").alias("event_count")
>     )
> 
> # Sliding window (overlapping)
> sliding_window = streaming_df \
>     .groupBy(
>         window("event_time", "2 minutes", "30 seconds"),
>         "sensor_id"
>     ) \
>     .agg(
>         avg("temperature").alias("avg_temp"),
>         stddev("temperature").alias("stddev_temp")
>     )
> 
> # Write output
> query = tumbling_window.writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .option("truncate", False) \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> ## 2. Stateful Stream Processing
> 
> ```python stateful_processing.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("StatefulProcessing") \
>     .getOrCreate()
> 
> # Read streaming data
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 5) \
>     .load() \
>     .withColumn("user_id", (rand() * 100).cast("int")) \
>     .withColumn("action", when(rand() > 0.5, "login").otherwise("logout")) \
>     .withColumnRenamed("timestamp", "event_time")
> 
> # Using mapGroupsWithState for custom stateful processing
> from pyspark.sql.streaming import GroupState, GroupStateTimeout
> 
> # Define state schema
> class SessionState:
>     def __init__(self, start_time, event_count, last_event):
>         self.start_time = start_time
>         self.event_count = event_count
>         self.last_event = last_event
> 
> def update_user_session(user_id, events, state: GroupState):
>     """
>     Track user session with custom state management
>     """
>     if state.hasTimedOut:
>         # Session expired
>         return (user_id, state.get.event_count, "expired")
>     
>     if state.exists:
>         session = state.get
>         session.event_count += len(list(events))
>         for event in events:
>             session.last_event = event.event_time
>     else:
>         events_list = list(events)
>         session = SessionState(
>             start_time=events_list[0].event_time,
>             event_count=len(events_list),
>             last_event=events_list[-1].event_time
>         )
>     
>     state.update(session)
>     state.setTimeoutDuration(60000)  # 60 seconds timeout
>     
>     return (user_id, session.event_count, "active")
> 
> # Simpler approach using groupBy with aggregations
> stateful_aggregation = streaming_df \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy("user_id") \
>     .agg(
>         count("*").alias("total_events"),
>         max("event_time").alias("last_seen"),
>         collect_list("action").alias("actions")
>     )
> 
> query = stateful_aggregation.writeStream \
>     .outputMode("update") \
>     .format("console") \
>     .option("truncate", False) \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> ## 3. Watermarking for Late Data
> 
> ```python watermarking.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> 
> spark = SparkSession.builder \
>     .appName("Watermarking") \
>     .getOrCreate()
> 
> # Read streaming data
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 10) \
>     .load() \
>     .withColumn("sensor_id", (rand() * 3).cast("int")) \
>     .withColumn("value", (rand() * 100)) \
>     .withColumnRenamed("timestamp", "event_time") \
>     .withColumn("event_time", 
>                 expr("event_time - INTERVAL " + 
>                      "CAST(rand() * 120 AS INT) SECONDS"))  # Simulate late data
> 
> # Define watermark - data arriving more than 2 minutes late will be dropped
> watermarked_df = streaming_df \
>     .withWatermark("event_time", "2 minutes")
> 
> # Windowed aggregation with watermark
> aggregated = watermarked_df \
>     .groupBy(
>         window("event_time", "1 minute"),
>         "sensor_id"
>     ) \
>     .agg(
>         avg("value").alias("avg_value"),
>         count("*").alias("count"),
>         min("event_time").alias("min_event_time"),
>         max("event_time").alias("max_event_time")
>     )
> 
> # Multiple watermarks example
> result = aggregated \
>     .select(
>         col("window.start").alias("window_start"),
>         col("window.end").alias("window_end"),
>         "sensor_id",
>         "avg_value",
>         "count",
>         current_timestamp().alias("processing_time")
>     )
> 
> query = result.writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", False) \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> ## 4. Joining Streams with Static DataFrames
> 
> ```python stream_static_join.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("StreamStaticJoin") \
>     .getOrCreate()
> 
> # Create static reference DataFrame (device metadata)
> device_data = [
>     (1, "Temperature Sensor", "Building A", "Floor 1"),
>     (2, "Humidity Sensor", "Building A", "Floor 2"),
>     (3, "Pressure Sensor", "Building B", "Floor 1"),
>     (4, "Motion Sensor", "Building B", "Floor 3"),
>     (5, "Light Sensor", "Building C", "Floor 2")
> ]
> 
> device_schema = StructType([
>     StructField("device_id", IntegerType()),
>     StructField("device_type", StringType()),
>     StructField("building", StringType()),
>     StructField("floor", StringType())
> ])
> 
> static_df = spark.createDataFrame(device_data, device_schema)
> 
> # Read streaming data
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 5) \
>     .load() \
>     .withColumn("device_id", (rand() * 5 + 1).cast("int")) \
>     .withColumn("reading_value", (rand() * 100)) \
>     .withColumnRenamed("timestamp", "event_time")
> 
> # Join stream with static data
> enriched_stream = streaming_df \
>     .join(static_df, "device_id", "left") \
>     .select(
>         "event_time",
>         "device_id",
>         "device_type",
>         "building",
>         "floor",
>         "reading_value"
>     )
> 
> # You can also use broadcast join for better performance
> # enriched_stream = streaming_df.join(broadcast(static_df), "device_id")
> 
> query = enriched_stream.writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", False) \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> ## 5. Joining Streams with Streams
> 
> ```python stream_stream_join.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> 
> spark = SparkSession.builder \
>     .appName("StreamStreamJoin") \
>     .getOrCreate()
> 
> # First stream: User clicks
> clicks_stream = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 10) \
>     .load() \
>     .withColumn("user_id", (rand() * 100).cast("int")) \
>     .withColumn("page_id", (rand() * 20).cast("int")) \
>     .withColumnRenamed("timestamp", "click_time") \
>     .withWatermark("click_time", "5 minutes")
> 
> # Second stream: User impressions
> impressions_stream = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 8) \
>     .load() \
>     .withColumn("user_id", (rand() * 100).cast("int")) \
>     .withColumn("ad_id", (rand() * 50).cast("int")) \
>     .withColumnRenamed("timestamp", "impression_time") \
>     .withWatermark("impression_time", "5 minutes")
> 
> # Inner join with time constraints
> # Match clicks that happen within 10 minutes after an impression
> joined_stream = impressions_stream.alias("i") \
>     .join(
>         clicks_stream.alias("c"),
>         expr("""
>             i.user_id = c.user_id AND
>             c.click_time >= i.impression_time AND
>             c.click_time <= i.impression_time + INTERVAL 10 MINUTES
>         """),
>         "inner"
>     ) \
>     .select(
>         col("i.user_id").alias("user_id"),
>         col("i.ad_id").alias("ad_id"),
>         col("i.impression_time").alias("impression_time"),
>         col("c.page_id").alias("page_id"),
>         col("c.click_time").alias("click_time"),
>         (col("c.click_time").cast("long") - col("i.impression_time").cast("long")).alias("time_to_click_seconds")
>     )
> 
> # Left outer join example
> left_joined = impressions_stream.alias("i") \
>     .join(
>         clicks_stream.alias("c"),
>         expr("""
>             i.user_id = c.user_id AND
>             c.click_time >= i.impression_time AND
>             c.click_time <= i.impression_time + INTERVAL 10 MINUTES
>         """),
>         "leftOuter"
>     )
> 
> query = joined_stream.writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", False) \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> ## 6. Different Output Modes
> 
> ```python output_modes.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> 
> spark = SparkSession.builder \
>     .appName("OutputModes") \
>     .getOrCreate()
> 
> # Create streaming DataFrame
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 5) \
>     .load() \
>     .withColumn("category", (rand() * 3).cast("int")) \
>     .withColumn("amount", (rand() * 1000)) \
>     .withColumnRenamed("timestamp", "event_time")
> 
> # Prepare aggregated data
> aggregated = streaming_df \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy(
>         window("event_time", "1 minute"),
>         "category"
>     ) \
>     .agg(
>         sum("amount").alias("total_amount"),
>         avg("amount").alias("avg_amount"),
>         count("*").alias("count")
>     )
> 
> # 1. APPEND MODE - Only new rows (works with aggregations with watermark)
> append_query = aggregated \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .queryName("append_mode") \
>     .option("truncate", False) \
>     .option("checkpointLocation", "/tmp/checkpoint_append") \
>     .start()
> 
> # 2. COMPLETE MODE - Entire result table (only for aggregations)
> complete_query = streaming_df \
>     .groupBy("category") \
>     .agg(
>         sum("amount").alias("total_amount"),
>         count("*").alias("count")
>     ) \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .queryName("complete_mode") \
>     .option("truncate", False) \
>     .option("checkpointLocation", "/tmp/checkpoint_complete") \
>     .start()
> 
> # 3. UPDATE MODE - Only updated rows
> update_query = aggregated \
>     .writeStream \
>     .outputMode("update") \
>     .format("console") \
>     .queryName("update_mode") \
>     .option("truncate", False) \
>     .option("checkpointLocation", "/tmp/checkpoint_update") \
>     .start()
> 
> # Non-aggregation example (append mode only)
> non_agg_query = streaming_df \
>     .filter(col("amount") > 500) \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .queryName("non_aggregation") \
>     .option("truncate", False) \
>     .option("checkpointLocation", "/tmp/checkpoint_non_agg") \
>     .start()
> 
> # Wait for all queries
> spark.streams.awaitAnyTermination()
> ```
> 
> ## 7. Idempotent Sinks (foreachBatch)
> 
> ```python foreach_batch.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> import logging
> 
> spark = SparkSession.builder \
>     .appName("ForeachBatch") \
>     .getOrCreate()
> 
> # Setup logging
> logging.basicConfig(level=logging.INFO)
> logger = logging.getLogger(__name__)
> 
> # Read streaming data
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 10) \
>     .load() \
>     .withColumn("user_id", (rand() * 100).cast("int")) \
>     .withColumn("transaction_amount", (rand() * 1000)) \
>     .withColumn("transaction_id", expr("uuid()")) \
>     .withColumnRenamed("timestamp", "event_time")
> 
> def process_batch(batch_df, batch_id):
>     """
>     Custom batch processing function with idempotent writes
>     """
>     logger.info(f"Processing batch {batch_id}")
>     
>     # Count records
>     record_count = batch_df.count()
>     logger.info(f"Batch {batch_id} contains {record_count} records")
>     
>     if record_count == 0:
>         logger.info(f"Batch {batch_id} is empty, skipping")
>         return
>     
>     # Multiple writes in a single batch (transactional)
>     try:
>         # Write to multiple sinks
>         # 1. Write to Parquet (with partition)
>         batch_df.write \
>             .mode("append") \
>             .partitionBy("user_id") \
>             .parquet(f"/tmp/transactions/batch_{batch_id}")
>         
>         # 2. Write aggregated data
>         aggregated = batch_df.groupBy("user_id") \
>             .agg(
>                 sum("transaction_amount").alias("total_amount"),
>                 count("*").alias("transaction_count")
>             )
>         
>         aggregated.write \
>             .mode("overwrite") \
>             .parquet(f"/tmp/aggregated/batch_{batch_id}")
>         
>         # 3. Conditional write (high-value transactions)
>         high_value = batch_df.filter(col("transaction_amount") > 500)
>         if high_value.count() > 0:
>             high_value.write \
>                 .mode("append") \
>                 .parquet(f"/tmp/high_value_transactions")
>         
>         # 4. Write to console for debugging
>         logger.info("Sample records from batch:")
>         batch_df.show(5, truncate=False)
>         
>         # 5. Custom processing (e.g., send to external system)
>         # This is where you'd implement idempotency checks
>         # For example, using transaction_id as deduplication key
>         
>         logger.info(f"Successfully processed batch {batch_id}")
>         
>     except Exception as e:
>         logger.error(f"Error processing batch {batch_id}: {str(e)}")
>         raise
> 
> # Alternative: More complex batch processing with external systems
> def process_batch_advanced(batch_df, batch_id):
>     """
>     Advanced batch processing with deduplication and external writes
>     """
>     from datetime import datetime
>     
>     if batch_df.count() == 0:
>         return
>     
>     # Add batch metadata
>     enriched_df = batch_df.withColumn("batch_id", lit(batch_id)) \
>                           .withColumn("processed_time", lit(datetime.now()))
>     
>     # Deduplication based on transaction_id
>     deduplicated = enriched_df.dropDuplicates(["transaction_id"])
>     
>     # Write with exactly-once semantics
>     deduplicated.write \
>         .format("delta")  # Using Delta Lake for ACID transactions \
>         .mode("append") \
>         .option("mergeSchema", "true") \
>         .save("/tmp/delta_transactions")
>     
>     logger.info(f"Batch {batch_id}: Processed {deduplicated.count()} unique transactions")
> 
> # Use foreachBatch
> query = streaming_df.writeStream \
>     .foreachBatch(process_batch) \
>     .option("checkpointLocation", "/tmp/checkpoint_foreach") \
>     .trigger(processingTime="10 seconds") \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> ## 8. Checkpointing
> 
> ```python checkpointing.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> import os
> 
> spark = SparkSession.builder \
>     .appName("Checkpointing") \
>     .config("spark.sql.streaming.checkpointLocation", "/tmp/default_checkpoint") \
>     .getOrCreate()
> 
> # Read streaming data
> streaming_df = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 10) \
>     .load() \
>     .withColumn("sensor_id", (rand() * 5).cast("int")) \
>     .withColumn("value", (rand() * 100)) \
>     .withColumnRenamed("timestamp", "event_time")
> 
> # Checkpoint directory structure:
> # /checkpoint_location/
> #   ├── commits/          (completed batches)
> #   ├── offsets/          (source offsets)
> #   ├── state/            (stateful operation state)
> #   └── metadata          (query metadata)
> 
> # Example 1: Basic checkpointing
> checkpoint_path_1 = "/tmp/checkpoint_basic"
> 
> query1 = streaming_df \
>     .writeStream \
>     .outputMode("append") \
>     .format("parquet") \
>     .option("path", "/tmp/output_basic") \
>     .option("checkpointLocation", checkpoint_path_1) \
>     .start()
> 
> # Example 2: Checkpointing with aggregations (maintains state)
> checkpoint_path_2 = "/tmp/checkpoint_aggregation"
> 
> aggregated = streaming_df \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy(
>         window("event_time", "5 minutes"),
>         "sensor_id"
>     ) \
>     .agg(
>         avg("value").alias("avg_value"),
>         count("*").alias("count")
>     )
> 
> query2 = aggregated.writeStream \
>     .outputMode("append") \
>     .format("parquet") \
>     .option("path", "/tmp/output_aggregation") \
>     .option("checkpointLocation", checkpoint_path_2) \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> # Example 3: Checkpointing with foreachBatch
> checkpoint_path_3 = "/tmp/checkpoint_foreach"
> 
> def write_batch(batch_df, batch_id):
>     """
>     Write batch with checkpoint awareness
>     """
>     print(f"Processing batch {batch_id}")
>     
>     # The checkpoint ensures this batch is processed exactly once
>     # Even if the job fails and restarts
>     batch_df.write \
>         .mode("append") \
>         .parquet(f"/tmp/output_foreach")
>     
>     print(f"Completed batch {batch_id}")
> 
> query3 = streaming_df.writeStream \
>     .foreachBatch(write_batch) \
>     .option("checkpointLocation", checkpoint_path_3) \
>     .start()
> 
> # Example 4: Multiple queries with different checkpoints
> checkpoint_path_4a = "/tmp/checkpoint_filter_high"
> checkpoint_path_4b = "/tmp/checkpoint_filter_low"
> 
> # High value stream
> high_value = streaming_df.filter(col("value") > 50)
> query4a = high_value.writeStream \
>     .outputMode("append") \
>     .format("parquet") \
>     .option("path", "/tmp/output_high_value") \
>     .option("checkpointLocation", checkpoint_path_4a) \
>     .start()
> 
> # Low value stream
> low_value = streaming_df.filter(col("value") <= 50)
> query4b = low_value.writeStream \
>     .outputMode("append") \
>     .format("parquet") \
>     .option("path", "/tmp/output_low_value") \
>     .option("checkpointLocation", checkpoint_path_4b) \
>     .start()
> 
> # Checkpoint recovery example
> def get_or_create_query(checkpoint_location, output_path):
>     """
>     Helper function to recover or create streaming query
>     """
>     if os.path.exists(checkpoint_location):
>         print(f"Recovering from checkpoint: {checkpoint_location}")
>     else:
>         print(f"Creating new checkpoint: {checkpoint_location}")
>     
>     return streaming_df.writeStream \
>         .outputMode("append") \
>         .format("parquet") \
>         .option("path", output_path) \
>         .option("checkpointLocation", checkpoint_location) \
>         .start()
> 
> # Monitor all active streams
> def monitor_streams():
>     """
>     Monitor all active streaming queries
>     """
>     active_streams = spark.streams.active
>     print(f"Active streams: {len(active_streams)}")
>     
>     for stream in active_streams:
>         print(f"Query: {stream.name}, ID: {stream.id}, Status: {stream.status}")
>         
>         # Get recent progress
>         if stream.lastProgress:
>             print(f"  Latest batch: {stream.lastProgress['batchId']}")
>             print(f"  Input rows: {stream.lastProgress['numInputRows']}")
>             print(f"  Processing time: {stream.lastProgress['durationMs']}")
> 
> # Call monitoring
> monitor_streams()
> 
> # Wait for termination
> spark.streams.awaitAnyTermination()
> 
> # Graceful shutdown
> print("Stopping all queries...")
> for query in spark.streams.active:
>     query.stop()
> print("All queries stopped")
> ```
> 
> ## Complete Example: End-to-End Pipeline
> 
> ```python complete_pipeline.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("CompleteStreamingPipeline") \
>     .config("spark.sql.shuffle.partitions", "4") \
>     .getOrCreate()
> 
> # Static reference data
> categories_data = [(1, "Electronics"), (2, "Clothing"), (3, "Food")]
> categories_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
> 
> # Streaming source
> events = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 20) \
>     .load() \
>     .withColumn("event_id", expr("uuid()")) \
>     .withColumn("user_id", (rand() * 1000).cast("int")) \
>     .withColumn("category_id", (rand() * 3 + 1).cast("int")) \
>     .withColumn("amount", (rand() * 500)) \
>     .withColumnRenamed("timestamp", "event_time") \
>     .withColumn("event_time", expr("event_time - INTERVAL CAST(rand() * 60 AS INT) SECONDS"))
> 
> # Apply watermark
> watermarked_events = events.withWatermark("event_time", "5 minutes")
> 
> # Join with static data
> enriched = watermarked_events.join(categories_df, "category_id")
> 
> # Windowed aggregations
> aggregated = enriched \
>     .groupBy(
>         window("event_time", "2 minutes", "1 minute"),
>         "category_name"
>     ) \
>     .agg(
>         sum("amount").alias("total_amount"),
>         avg("amount").alias("avg_amount"),
>         count("*").alias("event_count"),
>         countDistinct("user_id").alias("unique_users")
>     )
> 
> # Write with foreachBatch
> def write_micro_batch(df, batch_id):
>     print(f"Processing batch {batch_id}")
>     
>     # Write to multiple outputs
>     df.write.mode("append").parquet(f"/tmp/complete_output/data")
>     
>     # Write summary
>     df.agg(
>         sum("total_amount").alias("batch_total"),
>         sum("event_count").alias("batch_events")
>     ).write.mode("append").parquet(f"/tmp/complete_output/summary")
> 
> query = aggregated.writeStream \
>     .foreachBatch(write_micro_batch) \
>     .option("checkpointLocation", "/tmp/checkpoint_complete") \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> query.awaitTermination()
> ```
> 
> These examples cover all the major Structured Streaming patterns in PySpark. Each script is self-contained and demonstrates specific concepts with practical use cases. You can run them individually or combine patterns as needed for your application.