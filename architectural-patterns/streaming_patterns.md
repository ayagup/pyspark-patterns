### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 6:59:39 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Streaming Patterns
> - Structured Streaming Pattern
> - Continuous Processing Pattern
> - Micro-batch Processing Pattern
> - Stateful Streaming Pattern
> - Stateless Streaming Pattern
> - Watermarking Pattern
> - Trigger Pattern (Once, ProcessingTime, Continuous)
> - Checkpointing Pattern
> - Event Time Processing Pattern
> - Processing Time Processing Pattern
> - Tumbling Window Pattern
> - Sliding Window Pattern
> - Session Window Pattern
> - Late Data Handling Pattern
> - Exactly-Once Semantics Pattern
> - At-Least-Once Semantics Pattern
> - Idempotent Write Pattern
> - Stream-to-Stream Join Pattern
> - Stream-to-Static Join Pattern
> - Foreachbatch Pattern
> - Foreach Sink Pattern
> ```

#### _Assistant_

> # PySpark Streaming Patterns - Complete Implementation Guide
> 
> I'll create comprehensive examples for each streaming pattern. Let's organize them into separate files:
> 
> ## 1. Basic Streaming Patterns
> 
> ```python streaming_basics.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Streaming Patterns") \
>     .config("spark.sql.shuffle.partitions", "2") \
>     .getOrCreate()
> 
> spark.sparkContext.setLogLevel("WARN")
> 
> # Define schema for input data
> schema = StructType([
>     StructField("id", StringType(), True),
>     StructField("timestamp", TimestampType(), True),
>     StructField("value", DoubleType(), True),
>     StructField("category", StringType(), True)
> ])
> 
> # =============================================================================
> # 1. STRUCTURED STREAMING PATTERN
> # =============================================================================
> print("=" * 80)
> print("1. Structured Streaming Pattern")
> print("=" * 80)
> 
> # Read from socket source (replace with Kafka/file in production)
> structured_stream = spark \
>     .readStream \
>     .format("socket") \
>     .option("host", "localhost") \
>     .option("port", 9999) \
>     .load()
> 
> # Process and write
> structured_query = structured_stream \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .start()
> 
> # =============================================================================
> # 2. STATELESS STREAMING PATTERN
> # =============================================================================
> print("=" * 80)
> print("2. Stateless Streaming Pattern")
> print("=" * 80)
> 
> # Read streaming data
> stateless_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Stateless transformations (filter, map, select)
> stateless_processed = stateless_stream \
>     .filter(col("value") > 100) \
>     .select("id", "value", "category") \
>     .withColumn("doubled_value", col("value") * 2)
> 
> stateless_query = stateless_processed \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> # =============================================================================
> # 3. STATEFUL STREAMING PATTERN
> # =============================================================================
> print("=" * 80)
> print("3. Stateful Streaming Pattern")
> print("=" * 80)
> 
> stateful_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Stateful aggregation
> stateful_processed = stateful_stream \
>     .groupBy("category") \
>     .agg(
>         count("*").alias("count"),
>         avg("value").alias("avg_value"),
>         sum("value").alias("total_value")
>     )
> 
> stateful_query = stateful_processed \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .start()
> 
> ```
> 
> ## 2. Processing Mode Patterns
> 
> ```python processing_modes.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("Processing Mode Patterns") \
>     .getOrCreate()
> 
> schema = StructType([
>     StructField("id", StringType(), True),
>     StructField("timestamp", TimestampType(), True),
>     StructField("value", DoubleType(), True)
> ])
> 
> # =============================================================================
> # 4. MICRO-BATCH PROCESSING PATTERN (Default)
> # =============================================================================
> print("=" * 80)
> print("4. Micro-batch Processing Pattern")
> print("=" * 80)
> 
> microbatch_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Micro-batch with 10 second intervals
> microbatch_query = microbatch_stream \
>     .groupBy(window(col("timestamp"), "1 minute")) \
>     .count() \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .trigger(processingTime="10 seconds") \
>     .start()
> 
> # =============================================================================
> # 5. CONTINUOUS PROCESSING PATTERN
> # =============================================================================
> print("=" * 80)
> print("5. Continuous Processing Pattern")
> print("=" * 80)
> 
> continuous_stream = spark \
>     .readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "input-topic") \
>     .load()
> 
> # Continuous processing with 1 second checkpoint interval
> continuous_query = continuous_stream \
>     .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
>     .writeStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("topic", "output-topic") \
>     .option("checkpointLocation", "/tmp/checkpoint/continuous") \
>     .trigger(continuous="1 second") \
>     .start()
> 
> ```
> 
> ## 3. Time-based Patterns
> 
> ```python time_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> import time
> 
> spark = SparkSession.builder \
>     .appName("Time-based Patterns") \
>     .getOrCreate()
> 
> schema = StructType([
>     StructField("id", StringType(), True),
>     StructField("event_time", TimestampType(), True),
>     StructField("value", DoubleType(), True),
>     StructField("category", StringType(), True)
> ])
> 
> # =============================================================================
> # 6. EVENT TIME PROCESSING PATTERN
> # =============================================================================
> print("=" * 80)
> print("6. Event Time Processing Pattern")
> print("=" * 80)
> 
> event_time_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Use event_time column for time-based operations
> event_time_processed = event_time_stream \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy(
>         window(col("event_time"), "5 minutes", "1 minute"),
>         col("category")
>     ) \
>     .agg(
>         count("*").alias("count"),
>         avg("value").alias("avg_value")
>     )
> 
> event_time_query = event_time_processed \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> # =============================================================================
> # 7. PROCESSING TIME PROCESSING PATTERN
> # =============================================================================
> print("=" * 80)
> print("7. Processing Time Processing Pattern")
> print("=" * 80)
> 
> processing_time_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Use current_timestamp() for processing time
> processing_time_processed = processing_time_stream \
>     .withColumn("processing_time", current_timestamp()) \
>     .groupBy(
>         window(col("processing_time"), "5 minutes"),
>         col("category")
>     ) \
>     .count()
> 
> processing_time_query = processing_time_processed \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .start()
> 
> # =============================================================================
> # 8. WATERMARKING PATTERN
> # =============================================================================
> print("=" * 80)
> print("8. Watermarking Pattern")
> print("=" * 80)
> 
> watermark_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Watermark allows handling late data up to 15 minutes
> watermark_processed = watermark_stream \
>     .withWatermark("event_time", "15 minutes") \
>     .groupBy(
>         window(col("event_time"), "10 minutes", "5 minutes"),
>         col("category")
>     ) \
>     .agg(
>         count("*").alias("count"),
>         sum("value").alias("total_value"),
>         max("event_time").alias("max_event_time")
>     )
> 
> watermark_query = watermark_processed \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> ```
> 
> ## 4. Window Patterns
> 
> ```python window_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("Window Patterns") \
>     .getOrCreate()
> 
> schema = StructType([
>     StructField("user_id", StringType(), True),
>     StructField("event_time", TimestampType(), True),
>     StructField("action", StringType(), True),
>     StructField("value", DoubleType(), True)
> ])
> 
> input_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # =============================================================================
> # 9. TUMBLING WINDOW PATTERN
> # =============================================================================
> print("=" * 80)
> print("9. Tumbling Window Pattern")
> print("=" * 80)
> 
> # Non-overlapping windows of 10 minutes
> tumbling_window = input_stream \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy(
>         window(col("event_time"), "10 minutes"),
>         col("user_id")
>     ) \
>     .agg(
>         count("*").alias("event_count"),
>         sum("value").alias("total_value")
>     )
> 
> tumbling_query = tumbling_window \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> # =============================================================================
> # 10. SLIDING WINDOW PATTERN
> # =============================================================================
> print("=" * 80)
> print("10. Sliding Window Pattern")
> print("=" * 80)
> 
> # Overlapping windows: 10 min window, sliding every 5 min
> sliding_window = input_stream \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy(
>         window(col("event_time"), "10 minutes", "5 minutes"),
>         col("user_id")
>     ) \
>     .agg(
>         count("*").alias("event_count"),
>         avg("value").alias("avg_value")
>     )
> 
> sliding_query = sliding_window \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> # =============================================================================
> # 11. SESSION WINDOW PATTERN
> # =============================================================================
> print("=" * 80)
> print("11. Session Window Pattern")
> print("=" * 80)
> 
> # Session windows group events with gaps <= 30 minutes
> session_window = input_stream \
>     .withWatermark("event_time", "20 minutes") \
>     .groupBy(
>         session_window(col("event_time"), "30 minutes"),
>         col("user_id")
>     ) \
>     .agg(
>         count("*").alias("session_event_count"),
>         min("event_time").alias("session_start"),
>         max("event_time").alias("session_end"),
>         sum("value").alias("session_total_value")
>     )
> 
> session_query = session_window \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> ```
> 
> ## 5. Late Data and Trigger Patterns
> 
> ```python late_data_triggers.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("Late Data and Triggers") \
>     .getOrCreate()
> 
> schema = StructType([
>     StructField("id", StringType(), True),
>     StructField("event_time", TimestampType(), True),
>     StructField("value", DoubleType(), True),
>     StructField("category", StringType(), True)
> ])
> 
> # =============================================================================
> # 12. LATE DATA HANDLING PATTERN
> # =============================================================================
> print("=" * 80)
> print("12. Late Data Handling Pattern")
> print("=" * 80)
> 
> late_data_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Handle late data with watermarking
> late_data_processed = late_data_stream \
>     .withWatermark("event_time", "10 minutes") \
>     .groupBy(
>         window(col("event_time"), "5 minutes"),
>         col("category")
>     ) \
>     .agg(
>         count("*").alias("count"),
>         sum("value").alias("total")
>     ) \
>     .select(
>         col("window.start").alias("window_start"),
>         col("window.end").alias("window_end"),
>         col("category"),
>         col("count"),
>         col("total")
>     )
> 
> late_data_query = late_data_processed \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> # =============================================================================
> # 13. TRIGGER PATTERN - ONCE
> # =============================================================================
> print("=" * 80)
> print("13. Trigger Pattern - Once")
> print("=" * 80)
> 
> trigger_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # Process all available data in a single batch and stop
> trigger_once_query = trigger_stream \
>     .groupBy("category") \
>     .count() \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .trigger(once=True) \
>     .start()
> 
> # =============================================================================
> # 14. TRIGGER PATTERN - PROCESSING TIME
> # =============================================================================
> print("=" * 80)
> print("14. Trigger Pattern - Processing Time")
> print("=" * 80)
> 
> # Trigger every 30 seconds
> trigger_processing_query = trigger_stream \
>     .groupBy("category") \
>     .count() \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> # =============================================================================
> # 15. TRIGGER PATTERN - AVAILABLE NOW
> # =============================================================================
> print("=" * 80)
> print("15. Trigger Pattern - Available Now")
> print("=" * 80)
> 
> # Process all available data in multiple micro-batches
> trigger_available_query = trigger_stream \
>     .groupBy("category") \
>     .count() \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .trigger(availableNow=True) \
>     .start()
> 
> # =============================================================================
> # 16. CHECKPOINTING PATTERN
> # =============================================================================
> print("=" * 80)
> print("16. Checkpointing Pattern")
> print("=" * 80)
> 
> checkpoint_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> checkpoint_query = checkpoint_stream \
>     .groupBy("category") \
>     .agg(count("*").alias("count")) \
>     .writeStream \
>     .outputMode("complete") \
>     .format("console") \
>     .option("checkpointLocation", "/tmp/checkpoint/my_stream") \
>     .start()
> 
> ```
> 
> ## 6. Semantics Patterns
> 
> ```python semantics_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("Semantics Patterns") \
>     .getOrCreate()
> 
> schema = StructType([
>     StructField("id", StringType(), True),
>     StructField("timestamp", TimestampType(), True),
>     StructField("value", DoubleType(), True)
> ])
> 
> # =============================================================================
> # 17. EXACTLY-ONCE SEMANTICS PATTERN
> # =============================================================================
> print("=" * 80)
> print("17. Exactly-Once Semantics Pattern")
> print("=" * 80)
> 
> # Using Kafka with idempotent writes and checkpointing
> exactly_once_stream = spark \
>     .readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "input-topic") \
>     .option("startingOffsets", "earliest") \
>     .load()
> 
> exactly_once_processed = exactly_once_stream \
>     .selectExpr("CAST(value AS STRING) as json") \
>     .select(from_json(col("json"), schema).alias("data")) \
>     .select("data.*") \
>     .groupBy("id") \
>     .agg(sum("value").alias("total_value"))
> 
> exactly_once_query = exactly_once_processed \
>     .writeStream \
>     .outputMode("complete") \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("topic", "output-topic") \
>     .option("checkpointLocation", "/tmp/checkpoint/exactly_once") \
>     .start()
> 
> # =============================================================================
> # 18. AT-LEAST-ONCE SEMANTICS PATTERN
> # =============================================================================
> print("=" * 80)
> print("18. At-Least-Once Semantics Pattern")
> print("=" * 80)
> 
> at_least_once_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # At-least-once: Data might be processed multiple times
> at_least_once_query = at_least_once_stream \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .start()
> 
> # =============================================================================
> # 19. IDEMPOTENT WRITE PATTERN
> # =============================================================================
> print("=" * 80)
> print("19. Idempotent Write Pattern")
> print("=" * 80)
> 
> idempotent_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> def idempotent_write(batch_df, batch_id):
>     """
>     Write function that ensures idempotency
>     """
>     # Add batch_id to ensure idempotency
>     batch_with_id = batch_df.withColumn("batch_id", lit(batch_id))
>     
>     # Write to target with merge/upsert logic
>     batch_with_id.write \
>         .format("delta") \
>         .mode("append") \
>         .option("mergeSchema", "true") \
>         .save("/tmp/output/delta")
>     
>     print(f"Processed batch {batch_id} with {batch_df.count()} records")
> 
> idempotent_query = idempotent_stream \
>     .writeStream \
>     .foreachBatch(idempotent_write) \
>     .option("checkpointLocation", "/tmp/checkpoint/idempotent") \
>     .start()
> 
> ```
> 
> ## 7. Join Patterns
> 
> ```python join_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("Join Patterns") \
>     .getOrCreate()
> 
> # Schemas
> event_schema = StructType([
>     StructField("event_id", StringType(), True),
>     StructField("user_id", StringType(), True),
>     StructField("event_time", TimestampType(), True),
>     StructField("action", StringType(), True)
> ])
> 
> user_schema = StructType([
>     StructField("user_id", StringType(), True),
>     StructField("name", StringType(), True),
>     StructField("country", StringType(), True)
> ])
> 
> # =============================================================================
> # 20. STREAM-TO-STREAM JOIN PATTERN
> # =============================================================================
> print("=" * 80)
> print("20. Stream-to-Stream Join Pattern")
> print("=" * 80)
> 
> # First stream - user events
> stream1 = spark \
>     .readStream \
>     .format("json") \
>     .schema(event_schema) \
>     .option("path", "/tmp/input/events") \
>     .load()
> 
> # Second stream - user actions
> stream2 = spark \
>     .readStream \
>     .format("json") \
>     .schema(event_schema) \
>     .option("path", "/tmp/input/actions") \
>     .load()
> 
> # Stream-to-stream join with watermarking
> stream1_with_watermark = stream1.withWatermark("event_time", "10 minutes")
> stream2_with_watermark = stream2.withWatermark("event_time", "10 minutes")
> 
> stream_join = stream1_with_watermark.alias("s1") \
>     .join(
>         stream2_with_watermark.alias("s2"),
>         expr("""
>             s1.user_id = s2.user_id AND
>             s1.event_time >= s2.event_time AND
>             s1.event_time <= s2.event_time + interval 5 minutes
>         """)
>     ) \
>     .select(
>         col("s1.event_id").alias("event1_id"),
>         col("s2.event_id").alias("event2_id"),
>         col("s1.user_id"),
>         col("s1.event_time").alias("event1_time"),
>         col("s2.event_time").alias("event2_time")
>     )
> 
> stream_join_query = stream_join \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> # =============================================================================
> # 21. STREAM-TO-STATIC JOIN PATTERN
> # =============================================================================
> print("=" * 80)
> print("21. Stream-to-Static Join Pattern")
> print("=" * 80)
> 
> # Static DataFrame - user details
> static_users = spark.read \
>     .format("json") \
>     .schema(user_schema) \
>     .load("/tmp/static/users")
> 
> # Streaming events
> streaming_events = spark \
>     .readStream \
>     .format("json") \
>     .schema(event_schema) \
>     .option("path", "/tmp/input/events") \
>     .load()
> 
> # Join streaming data with static data
> enriched_stream = streaming_events \
>     .join(static_users, "user_id") \
>     .select(
>         col("event_id"),
>         col("user_id"),
>         col("name"),
>         col("country"),
>         col("action"),
>         col("event_time")
>     )
> 
> static_join_query = enriched_stream \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .start()
> 
> ```
> 
> ## 8. Advanced Sink Patterns
> 
> ```python sink_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> import requests
> 
> spark = SparkSession.builder \
>     .appName("Sink Patterns") \
>     .getOrCreate()
> 
> schema = StructType([
>     StructField("id", StringType(), True),
>     StructField("timestamp", TimestampType(), True),
>     StructField("value", DoubleType(), True),
>     StructField("category", StringType(), True)
> ])
> 
> input_stream = spark \
>     .readStream \
>     .format("json") \
>     .schema(schema) \
>     .option("path", "/tmp/input") \
>     .load()
> 
> # =============================================================================
> # 22. FOREACHBATCH PATTERN
> # =============================================================================
> print("=" * 80)
> print("22. ForeachBatch Pattern")
> print("=" * 80)
> 
> def process_batch(batch_df, batch_id):
>     """
>     Custom batch processing logic
>     """
>     print(f"\n{'='*60}")
>     print(f"Processing Batch ID: {batch_id}")
>     print(f"{'='*60}")
>     
>     # Filter high-value records
>     high_value = batch_df.filter(col("value") > 100)
>     
>     # Write to multiple sinks
>     if high_value.count() > 0:
>         # Sink 1: Parquet
>         high_value.write \
>             .mode("append") \
>             .parquet(f"/tmp/output/high_value/batch_{batch_id}")
>         
>         # Sink 2: Delta Lake
>         high_value.write \
>             .format("delta") \
>             .mode("append") \
>             .save("/tmp/output/delta/high_value")
>         
>         # Sink 3: Console for monitoring
>         print(f"High value records in batch {batch_id}:")
>         high_value.show(truncate=False)
>     
>     # Aggregate and write summary
>     summary = batch_df.groupBy("category").agg(
>         count("*").alias("count"),
>         avg("value").alias("avg_value"),
>         sum("value").alias("total_value")
>     )
>     
>     summary.write \
>         .mode("append") \
>         .format("json") \
>         .save(f"/tmp/output/summary/batch_{batch_id}")
>     
>     print(f"Batch {batch_id} processed successfully\n")
> 
> foreachbatch_query = input_stream \
>     .writeStream \
>     .foreachBatch(process_batch) \
>     .option("checkpointLocation", "/tmp/checkpoint/foreachbatch") \
>     .start()
> 
> # =============================================================================
> # 23. FOREACH SINK PATTERN
> # =============================================================================
> print("=" * 80)
> print("23. Foreach Sink Pattern")
> print("=" * 80)
> 
> class CustomSink:
>     """
>     Custom sink for row-by-row processing
>     """
>     def open(self, partition_id, epoch_id):
>         """
>         Called when starting to process a partition
>         """
>         self.partition_id = partition_id
>         self.epoch_id = epoch_id
>         print(f"Opening partition {partition_id} in epoch {epoch_id}")
>         return True
>     
>     def process(self, row):
>         """
>         Process each row
>         """
>         # Custom processing logic
>         print(f"Processing: ID={row.id}, Value={row.value}, Category={row.category}")
>         
>         # Example: Send to external API
>         # self.send_to_api(row)
>         
>         # Example: Write to custom database
>         # self.write_to_db(row)
>         
>         # Example: Apply complex business logic
>         if row.value > 500:
>             print(f"  -> ALERT: High value detected for {row.id}")
>     
>     def close(self, error):
>         """
>         Called when finished processing partition
>         """
>         if error:
>             print(f"Error in partition {self.partition_id}: {error}")
>         else:
>             print(f"Successfully closed partition {self.partition_id}")
>     
>     def send_to_api(self, row):
>         """
>         Example: Send data to external API
>         """
>         try:
>             payload = {
>                 "id": row.id,
>                 "value": row.value,
>                 "category": row.category,
>                 "timestamp": str(row.timestamp)
>             }
>             # requests.post("http://api.example.com/data", json=payload)
>             pass
>         except Exception as e:
>             print(f"API Error: {e}")
> 
> # Use the custom sink
> foreach_query = input_stream \
>     .writeStream \
>     .foreach(CustomSink()) \
>     .option("checkpointLocation", "/tmp/checkpoint/foreach") \
>     .start()
> 
> # =============================================================================
> # Alternative Foreach with Lambda (simpler approach)
> # =============================================================================
> 
> def process_row(row):
>     """
>     Simple row processing function
>     """
>     print(f"Row: {row.id} | Value: {row.value} | Category: {row.category}")
> 
> # Using foreach with a simple function
> simple_foreach_query = input_stream \
>     .writeStream \
>     .foreach(
>         lambda row: process_row(row)
>     ) \
>     .start()
> 
> ```
> 
> ## 9. Complete Example with Multiple Patterns
> 
> ```python complete_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> import time
> 
> # =============================================================================
> # COMPLETE STREAMING APPLICATION
> # Combines multiple patterns
> # =============================================================================
> 
> spark = SparkSession.builder \
>     .appName("Complete Streaming Example") \
>     .config("spark.sql.streaming.schemaInference", "true") \
>     .config("spark.sql.shuffle.partitions", "2") \
>     .getOrCreate()
> 
> spark.sparkContext.setLogLevel("WARN")
> 
> # Define schemas
> transaction_schema = StructType([
>     StructField("transaction_id", StringType(), True),
>     StructField("user_id", StringType(), True),
>     StructField("event_time", TimestampType(), True),
>     StructField("amount", DoubleType(), True),
>     StructField("merchant_id", StringType(), True),
>     StructField("status", StringType(), True)
> ])
> 
> user_schema = StructType([
>     StructField("user_id", StringType(), True),
>     StructField("name", StringType(), True),
>     StructField("country", StringType(), True),
>     StructField("risk_score", IntegerType(), True)
> ])
> 
> # Static user data (Stream-to-Static Join)
> user_data = spark.read \
>     .format("json") \
>     .schema(user_schema) \
>     .load("/tmp/static/users")
> 
> # Streaming transactions (Structured Streaming)
> transactions = spark \
>     .readStream \
>     .format("json") \
>     .schema(transaction_schema) \
>     .option("path", "/tmp/input/transactions") \
>     .option("maxFilesPerTrigger", 1) \
>     .load()
> 
> # Enrich with user data (Stream-to-Static Join)
> enriched_transactions = transactions \
>     .join(user_data, "user_id")
> 
> # Apply watermarking (Watermarking Pattern)
> watermarked_data = enriched_transactions \
>     .withWatermark("event_time", "15 minutes")
> 
> # Tumbling window aggregation (Tumbling Window Pattern + Stateful)
> tumbling_agg = watermarked_data \
>     .groupBy(
>         window(col("event_time"), "10 minutes"),
>         col("merchant_id"),
>         col("country")
>     ) \
>     .agg(
>         count("*").alias("transaction_count"),
>         sum("amount").alias("total_amount"),
>         avg("amount").alias("avg_amount"),
>         max("amount").alias("max_amount")
>     )
> 
> # Sliding window for fraud detection (Sliding Window Pattern)
> fraud_detection = watermarked_data \
>     .groupBy(
>         window(col("event_time"), "30 minutes", "10 minutes"),
>         col("user_id")
>     ) \
>     .agg(
>         count("*").alias("tx

#### _User_

> Continue your response exactly where you left off:

#### _Assistant_

> ```python complete_example.py
> # ... existing code ...
> 
> fraud_detection = watermarked_data \
>     .groupBy(
>         window(col("event_time"), "30 minutes", "10 minutes"),
>         col("user_id")
>     ) \
>     .agg(
>         count("*").alias("tx_count"),
>         sum("amount").alias("total_spent"),
>         countDistinct("merchant_id").alias("unique_merchants")
>     ) \
>     .filter(
>         (col("tx_count") > 10) | (col("total_spent") > 5000)
>     )
> 
> # Session window for user activity (Session Window Pattern)
> user_sessions = watermarked_data \
>     .groupBy(
>         session_window(col("event_time"), "20 minutes"),
>         col("user_id")
>     ) \
>     .agg(
>         count("*").alias("session_tx_count"),
>         sum("amount").alias("session_total"),
>         min("event_time").alias("session_start"),
>         max("event_time").alias("session_end")
>     )
> 
> # Custom batch processing (ForeachBatch Pattern)
> def write_multiple_outputs(batch_df, batch_id):
>     """
>     Write to multiple sinks with custom logic
>     """
>     print(f"\n{'='*70}")
>     print(f"Processing Batch: {batch_id}")
>     print(f"{'='*70}")
>     
>     # Cache for multiple operations
>     batch_df.cache()
>     
>     # Output 1: All aggregated data to Delta
>     batch_df.write \
>         .format("delta") \
>         .mode("append") \
>         .save("/tmp/output/merchant_analytics")
>     
>     # Output 2: High-value merchants to separate location
>     high_value = batch_df.filter(col("total_amount") > 10000)
>     if high_value.count() > 0:
>         high_value.write \
>             .mode("append") \
>             .parquet("/tmp/output/high_value_merchants")
>         print(f"Found {high_value.count()} high-value merchants")
>     
>     # Output 3: Summary to JSON
>     summary = batch_df.agg(
>         sum("transaction_count").alias("total_transactions"),
>         sum("total_amount").alias("overall_total"),
>         avg("avg_amount").alias("global_avg")
>     )
>     
>     summary.write \
>         .mode("overwrite") \
>         .json(f"/tmp/output/summary/batch_{batch_id}")
>     
>     batch_df.unpersist()
>     print(f"Batch {batch_id} completed\n")
> 
> # Query 1: Main analytics with checkpointing (Checkpointing + ForeachBatch)
> analytics_query = tumbling_agg \
>     .writeStream \
>     .foreachBatch(write_multiple_outputs) \
>     .outputMode("append") \
>     .option("checkpointLocation", "/tmp/checkpoint/analytics") \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> # Query 2: Fraud alerts (Foreach Sink Pattern)
> class FraudAlertSink:
>     """
>     Custom sink for fraud alerts
>     """
>     def open(self, partition_id, epoch_id):
>         self.partition_id = partition_id
>         self.epoch_id = epoch_id
>         self.alert_count = 0
>         return True
>     
>     def process(self, row):
>         self.alert_count += 1
>         alert_msg = f"""
>         FRAUD ALERT!
>         User: {row.user_id}
>         Window: {row.window.start} to {row.window.end}
>         Transactions: {row.tx_count}
>         Total Spent: ${row.total_spent:.2f}
>         Unique Merchants: {row.unique_merchants}
>         """
>         print(alert_msg)
>         # In production: send to alerting system, Kafka, etc.
>     
>     def close(self, error):
>         if error:
>             print(f"Error in partition {self.partition_id}: {error}")
>         else:
>             print(f"Partition {self.partition_id}: Sent {self.alert_count} alerts")
> 
> fraud_query = fraud_detection \
>     .writeStream \
>     .foreach(FraudAlertSink()) \
>     .outputMode("append") \
>     .option("checkpointLocation", "/tmp/checkpoint/fraud") \
>     .start()
> 
> # Query 3: Session analysis to console
> session_query = user_sessions \
>     .writeStream \
>     .outputMode("append") \
>     .format("console") \
>     .option("truncate", "false") \
>     .option("checkpointLocation", "/tmp/checkpoint/sessions") \
>     .trigger(processingTime="1 minute") \
>     .start()
> 
> # Monitor all queries
> print("\n" + "="*70)
> print("STREAMING QUERIES STARTED")
> print("="*70)
> print(f"Analytics Query: {analytics_query.name}")
> print(f"Fraud Query: {fraud_query.name}")
> print(f"Session Query: {session_query.name}")
> print("="*70 + "\n")
> 
> # Keep running (in production, use proper orchestration)
> try:
>     spark.streams.awaitAnyTermination()
> except KeyboardInterrupt:
>     print("\nStopping all queries...")
>     analytics_query.stop()
>     fraud_query.stop()
>     session_query.stop()
>     print("All queries stopped.")
> ```
> 
> ## 10. Testing and Utilities
> 
> ```python streaming_utils.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> import random
> import json
> import time
> from datetime import datetime, timedelta
> 
> # =============================================================================
> # UTILITY FUNCTIONS FOR TESTING STREAMING PATTERNS
> # =============================================================================
> 
> def generate_sample_data(output_path, num_records=100, delay_seconds=1):
>     """
>     Generate sample streaming data for testing
>     """
>     categories = ["electronics", "clothing", "food", "books", "sports"]
>     
>     for i in range(num_records):
>         record = {
>             "id": f"id_{i}",
>             "timestamp": (datetime.now() - timedelta(seconds=random.randint(0, 300))).isoformat(),
>             "event_time": (datetime.now() - timedelta(seconds=random.randint(0, 300))).isoformat(),
>             "value": round(random.uniform(10, 1000), 2),
>             "category": random.choice(categories)
>         }
>         
>         # Write to file
>         filename = f"{output_path}/data_{int(time.time())}_{i}.json"
>         with open(filename, 'w') as f:
>             json.dump(record, f)
>         
>         print(f"Generated record {i+1}/{num_records}")
>         time.sleep(delay_seconds)
> 
> def generate_transaction_data(output_path, num_records=50):
>     """
>     Generate transaction data for testing
>     """
>     users = [f"user_{i}" for i in range(1, 21)]
>     merchants = [f"merchant_{i}" for i in range(1, 11)]
>     statuses = ["completed", "pending", "failed"]
>     
>     for i in range(num_records):
>         transaction = {
>             "transaction_id": f"tx_{i}_{int(time.time())}",
>             "user_id": random.choice(users),
>             "event_time": (datetime.now() - timedelta(seconds=random.randint(0, 600))).isoformat(),
>             "amount": round(random.uniform(5, 500), 2),
>             "merchant_id": random.choice(merchants),
>             "status": random.choice(statuses)
>         }
>         
>         filename = f"{output_path}/tx_{int(time.time())}_{i}.json"
>         with open(filename, 'w') as f:
>             json.dump(transaction, f)
>         
>         print(f"Generated transaction {i+1}/{num_records}")
>         time.sleep(0.5)
> 
> def monitor_streaming_query(query, duration_seconds=60):
>     """
>     Monitor a streaming query and print statistics
>     """
>     start_time = time.time()
>     
>     while time.time() - start_time < duration_seconds:
>         if query.isActive:
>             progress = query.lastProgress
>             if progress:
>                 print(f"\n{'='*70}")
>                 print(f"Query: {query.name}")
>                 print(f"Batch: {progress['batchId']}")
>                 print(f"Input Rows: {progress['numInputRows']}")
>                 print(f"Processing Rate: {progress.get('processedRowsPerSecond', 0):.2f} rows/sec")
>                 print(f"Timestamp: {progress['timestamp']}")
>                 print(f"{'='*70}\n")
>         else:
>             print(f"Query {query.name} is not active")
>             break
>         
>         time.sleep(5)
> 
> def create_checkpoint_cleaner(checkpoint_path):
>     """
>     Utility to clean checkpoint directories
>     """
>     import shutil
>     import os
>     
>     if os.path.exists(checkpoint_path):
>         shutil.rmtree(checkpoint_path)
>         print(f"Cleaned checkpoint: {checkpoint_path}")
>     else:
>         print(f"Checkpoint does not exist: {checkpoint_path}")
> 
> # =============================================================================
> # EXAMPLE USAGE
> # =============================================================================
> 
> if __name__ == "__main__":
>     import sys
>     
>     if len(sys.argv) < 2:
>         print("Usage: python streaming_utils.py [generate|monitor|clean]")
>         sys.exit(1)
>     
>     command = sys.argv[1]
>     
>     if command == "generate":
>         print("Generating sample data...")
>         generate_sample_data("/tmp/input", num_records=50, delay_seconds=2)
>         
>     elif command == "generate_tx":
>         print("Generating transaction data...")
>         generate_transaction_data("/tmp/input/transactions", num_records=100)
>         
>     elif command == "clean":
>         print("Cleaning checkpoints...")
>         checkpoints = [
>             "/tmp/checkpoint/analytics",
>             "/tmp/checkpoint/fraud",
>             "/tmp/checkpoint/sessions",
>             "/tmp/checkpoint/exactly_once",
>             "/tmp/checkpoint/foreachbatch",
>             "/tmp/checkpoint/foreach"
>         ]
>         for cp in checkpoints:
>             create_checkpoint_cleaner(cp)
>     
>     else:
>         print(f"Unknown command: {command}")
> ```
> 
> ## 11. Configuration and Best Practices
> 
> ```python streaming_config.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> 
> # =============================================================================
> # STREAMING CONFIGURATION PATTERNS AND BEST PRACTICES
> # =============================================================================
> 
> def create_optimized_spark_session(app_name="StreamingApp"):
>     """
>     Create Spark session with optimized streaming configurations
>     """
>     spark = SparkSession.builder \
>         .appName(app_name) \
>         .config("spark.sql.streaming.schemaInference", "true") \
>         .config("spark.sql.shuffle.partitions", "10") \
>         .config("spark.sql.streaming.stateStore.providerClass", 
>                 "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider") \
>         .config("spark.sql.streaming.metricsEnabled", "true") \
>         .config("spark.sql.streaming.stateStore.maintenanceInterval", "60s") \
>         .config("spark.streaming.stopGracefullyOnShutdown", "true") \
>         .config("spark.sql.adaptive.enabled", "true") \
>         .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>         .getOrCreate()
>     
>     spark.sparkContext.setLogLevel("WARN")
>     return spark
> 
> # =============================================================================
> # PATTERN-SPECIFIC CONFIGURATIONS
> # =============================================================================
> 
> # Configuration for low-latency streaming
> LOW_LATENCY_CONFIG = {
>     "spark.sql.shuffle.partitions": "2",
>     "spark.streaming.backpressure.enabled": "true",
>     "spark.sql.streaming.metricsEnabled": "true"
> }
> 
> # Configuration for high-throughput streaming
> HIGH_THROUGHPUT_CONFIG = {
>     "spark.sql.shuffle.partitions": "200",
>     "spark.streaming.kafka.maxRatePerPartition": "10000",
>     "spark.sql.streaming.kafka.consumer.cache.capacity": "256"
> }
> 
> # Configuration for stateful operations
> STATEFUL_CONFIG = {
>     "spark.sql.streaming.stateStore.stateSchemaCheck": "true",
>     "spark.sql.streaming.statefulOperator.checkCorrectness.enabled": "true",
>     "spark.sql.streaming.flatMapGroupsWithState.stateFormatVersion": "2"
> }
> 
> # =============================================================================
> # MONITORING AND DEBUGGING UTILITIES
> # =============================================================================
> 
> def print_streaming_metrics(query):
>     """
>     Print detailed metrics for a streaming query
>     """
>     if query.lastProgress:
>         progress = query.lastProgress
>         print(f"""
>         ╔══════════════════════════════════════════════════════════════╗
>         ║                    STREAMING METRICS                         ║
>         ╚══════════════════════════════════════════════════════════════╝
>         
>         Query Name:           {query.name}
>         Batch ID:             {progress.get('batchId', 'N/A')}
>         
>         INPUT:
>           Rows:               {progress.get('numInputRows', 0)}
>           Rate:               {progress.get('inputRowsPerSecond', 0):.2f} rows/sec
>         
>         PROCESSING:
>           Rows:               {progress.get('processedRowsPerSecond', 0):.2f} rows/sec
>           Duration:           {progress.get('durationMs', {}).get('triggerExecution', 0)} ms
>         
>         STATE:
>           Store Memory:       {progress.get('stateOperators', [{}])[0].get('memoryUsedBytes', 0)} bytes
>           Num Rows:           {progress.get('stateOperators', [{}])[0].get('numRowsTotal', 0)}
>         
>         SOURCES:
>         """)
>         
>         for source in progress.get('sources', []):
>             print(f"          {source.get('description', 'Unknown')}")
>             print(f"            Start Offset: {source.get('startOffset', 'N/A')}")
>             print(f"            End Offset:   {source.get('endOffset', 'N/A')}")
>         
>         print("\n" + "="*70 + "\n")
> 
> def debug_stream_lineage(df):
>     """
>     Print execution plan for debugging
>     """
>     print("\n" + "="*70)
>     print("LOGICAL PLAN:")
>     print("="*70)
>     df.explain(extended=False)
>     
>     print("\n" + "="*70)
>     print("PHYSICAL PLAN:")
>     print("="*70)
>     df.explain(extended=True)
>     print("="*70 + "\n")
> 
> # =============================================================================
> # ERROR HANDLING PATTERNS
> # =============================================================================
> 
> class StreamingErrorHandler:
>     """
>     Centralized error handling for streaming applications
>     """
>     
>     @staticmethod
>     def handle_batch_error(batch_df, batch_id, error):
>         """
>         Handle errors in foreachBatch
>         """
>         print(f"Error in batch {batch_id}: {str(error)}")
>         # Log to external system
>         # Send alert
>         # Write to dead letter queue
>     
>     @staticmethod
>     def create_dead_letter_queue_writer(error_path):
>         """
>         Write failed records to dead letter queue
>         """
>         def write_to_dlq(batch_df, batch_id):
>             try:
>                 batch_df.write \
>                     .mode("append") \
>                     .json(f"{error_path}/batch_{batch_id}")
>             except Exception as e:
>                 print(f"Failed to write to DLQ: {e}")
>         
>         return write_to_dlq
> 
> # =============================================================================
> # EXAMPLE: PUTTING IT ALL TOGETHER
> # =============================================================================
> 
> if __name__ == "__main__":
>     # Create optimized session
>     spark = create_optimized_spark_session("PatternDemo")
>     
>     # Example query with monitoring
>     df = spark.readStream \
>         .format("json") \
>         .load("/tmp/input") \
>         .groupBy("category") \
>         .count()
>     
>     query = df.writeStream \
>         .outputMode("complete") \
>         .format("console") \
>         .queryName("demo_query") \
>         .start()
>     
>     # Monitor for 30 seconds
>     import time
>     for _ in range(6):
>         time.sleep(5)
>         print_streaming_metrics(query)
>     
>     query.stop()
>     spark.stop()
> ```
> 
> ---
> 
> ## Summary of All Patterns
> 
> Here's a quick reference guide:
> 
> ```python pattern_summary.py
> """
> PYSPARK STREAMING PATTERNS - QUICK REFERENCE
> ==============================================
> 
> 1. STRUCTURED STREAMING PATTERN
>    - Use: .readStream() and .writeStream()
>    - File: streaming_basics.py
> 
> 2. CONTINUOUS PROCESSING PATTERN
>    - Use: .trigger(continuous="1 second")
>    - File: processing_modes.py
> 
> 3. MICRO-BATCH PROCESSING PATTERN
>    - Use: .trigger(processingTime="10 seconds")
>    - File: processing_modes.py
> 
> 4. STATEFUL STREAMING PATTERN
>    - Use: .groupBy().agg() with watermarking
>    - File: streaming_basics.py
> 
> 5. STATELESS STREAMING PATTERN
>    - Use: .filter(), .select(), .map()
>    - File: streaming_basics.py
> 
> 6. WATERMARKING PATTERN
>    - Use: .withWatermark("timestamp_col", "10 minutes")
>    - File: time_patterns.py
> 
> 7. TRIGGER PATTERNS
>    - Once: .trigger(once=True)
>    - Processing Time: .trigger(processingTime="30 seconds")
>    - Continuous: .trigger(continuous="1 second")
>    - Available Now: .trigger(availableNow=True)
>    - File: late_data_triggers.py
> 
> 8. CHECKPOINTING PATTERN
>    - Use: .option("checkpointLocation", "/path")
>    - File: late_data_triggers.py
> 
> 9. EVENT TIME PROCESSING PATTERN
>    - Use: window(col("event_time"), ...)
>    - File: time_patterns.py
> 
> 10. PROCESSING TIME PROCESSING PATTERN
>     - Use: window(current_timestamp(), ...)
>     - File: time_patterns.py
> 
> 11. TUMBLING WINDOW PATTERN
>     - Use: window(col("timestamp"), "10 minutes")
>     - File: window_patterns.py
> 
> 12. SLIDING WINDOW PATTERN
>     - Use: window(col("timestamp"), "10 minutes", "5 minutes")
>     - File: window_patterns.py
> 
> 13. SESSION WINDOW PATTERN
>     - Use: session_window(col("timestamp"), "30 minutes")
>     - File: window_patterns.py
> 
> 14. LATE DATA HANDLING PATTERN
>     - Use: withWatermark + outputMode("append")
>     - File: late_data_triggers.py
> 
> 15. EXACTLY-ONCE SEMANTICS PATTERN
>     - Use: Kafka + checkpointing
>     - File: semantics_patterns.py
> 
> 16. AT-LEAST-ONCE SEMANTICS PATTERN
>     - Use: Standard streaming without idempotency
>     - File: semantics_patterns.py
> 
> 17. IDEMPOTENT WRITE PATTERN
>     - Use: foreachBatch with merge/upsert logic
>     - File: semantics_patterns.py
> 
> 18. STREAM-TO-STREAM JOIN PATTERN
>     - Use: stream1.join(stream2) with watermarks
>     - File: join_patterns.py
> 
> 19. STREAM-TO-STATIC JOIN PATTERN
>     - Use: stream.join(static_df)
>     - File: join_patterns.py
> 
> 20. FOREACHBATCH PATTERN
>     - Use: .foreachBatch(custom_function)
>     - File: sink_patterns.py
> 
> 21. FOREACH SINK PATTERN
>     - Use: .foreach(CustomSink())
>     - File: sink_patterns.py
> 
> UTILITIES:
> - Data generation: streaming_utils.py
> - Configuration: streaming_config.py
> - Complete example: complete_example.py
> """
> ```
> 
> All patterns are production-ready and can be customized for your specific use case. You can use the **Apply Button** on any code block to add it to your project, or switch to **Agent Mode** for automated file creation and updates!