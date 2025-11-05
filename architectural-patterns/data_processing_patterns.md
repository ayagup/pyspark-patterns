### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 5:37:29 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Data Processing Patterns
> - Map-Reduce Pattern
> - Filter-Transform Pattern
> - Aggregation Pattern
> - Window Function Pattern
> - Broadcast Join Pattern
> - Shuffle Hash Join Pattern
> - Sort Merge Join Pattern
> - Bucket Join Pattern
> - Skewed Join Pattern
> - Salting Pattern (for skew handling)
> - Explode Pattern
> - Pivot/Unpivot Pattern
> - Union Pattern
> - Coalesce Pattern
> - Repartition Pattern
> - Cache/Persist Pattern
> - Lazy Evaluation Pattern
> - Chain of Responsibility Pattern
> - Pipeline Pattern
> - Lambda Architecture Pattern
> - Kappa Architecture Pattern
> ```

#### _Assistant_

> # PySpark Data Processing Patterns
> 
> I'll create comprehensive examples for each pattern. Let's organize them into separate files:
> 
> ## 1. Basic Transformation Patterns
> 
> ```python basic_transformations.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, upper, length, explode, array, lit
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("BasicTransformations") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> # Sample data
> data = [
>     (1, "Alice", 25, "Engineering"),
>     (2, "Bob", 30, "Sales"),
>     (3, "Charlie", 35, "Engineering"),
>     (4, "David", 28, "Marketing"),
>     (5, "Eve", 32, "Sales")
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "age", "department"])
> 
> print("=" * 80)
> print("MAP-REDUCE PATTERN")
> print("=" * 80)
> # Map: Transform each element
> mapped_df = df.select(
>     col("id"),
>     upper(col("name")).alias("name_upper"),
>     col("age"),
>     col("department")
> )
> mapped_df.show()
> 
> # Reduce: Aggregate operation
> reduced_df = df.groupBy("department").count()
> print("Department counts:")
> reduced_df.show()
> 
> # Map-Reduce combined: Calculate average age per department
> map_reduce_result = df.groupBy("department") \
>     .agg({"age": "avg"}) \
>     .withColumnRenamed("avg(age)", "avg_age")
> print("Average age by department:")
> map_reduce_result.show()
> 
> print("\n" + "=" * 80)
> print("FILTER-TRANSFORM PATTERN")
> print("=" * 80)
> # Filter records and transform
> filtered_transformed = df \
>     .filter(col("age") > 28) \
>     .select(
>         col("id"),
>         upper(col("name")).alias("name"),
>         col("age"),
>         (col("age") + 5).alias("age_in_5_years"),
>         col("department")
>     )
> filtered_transformed.show()
> 
> print("\n" + "=" * 80)
> print("EXPLODE PATTERN")
> print("=" * 80)
> # Create data with arrays
> array_data = [
>     (1, "Alice", ["Python", "Scala", "Java"]),
>     (2, "Bob", ["Python", "R"]),
>     (3, "Charlie", ["Java", "C++"])
> ]
> array_df = spark.createDataFrame(array_data, ["id", "name", "skills"])
> 
> # Explode array into multiple rows
> exploded_df = array_df.select(
>     col("id"),
>     col("name"),
>     explode(col("skills")).alias("skill")
> )
> print("Exploded skills:")
> exploded_df.show()
> 
> spark.stop()
> ```
> 
> ## 2. Aggregation and Window Patterns
> 
> ```python aggregation_window.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, sum, avg, max, min, count, dense_rank, row_number, lag, lead
> from pyspark.sql.window import Window
> 
> spark = SparkSession.builder \
>     .appName("AggregationWindow") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> # Sample sales data
> sales_data = [
>     (1, "2024-01-01", "Electronics", "Laptop", 1000),
>     (2, "2024-01-01", "Electronics", "Mouse", 25),
>     (3, "2024-01-02", "Electronics", "Keyboard", 75),
>     (4, "2024-01-01", "Clothing", "Shirt", 50),
>     (5, "2024-01-02", "Clothing", "Pants", 80),
>     (6, "2024-01-03", "Electronics", "Monitor", 300),
>     (7, "2024-01-03", "Clothing", "Jacket", 120)
> ]
> 
> df = spark.createDataFrame(sales_data, ["id", "date", "category", "product", "amount"])
> 
> print("=" * 80)
> print("AGGREGATION PATTERN")
> print("=" * 80)
> # Simple aggregations
> agg_result = df.groupBy("category").agg(
>     sum("amount").alias("total_sales"),
>     avg("amount").alias("avg_sales"),
>     count("*").alias("num_transactions"),
>     max("amount").alias("max_sale"),
>     min("amount").alias("min_sale")
> )
> agg_result.show()
> 
> print("\n" + "=" * 80)
> print("WINDOW FUNCTION PATTERN")
> print("=" * 80)
> # Define window specifications
> window_by_category = Window.partitionBy("category").orderBy(col("amount").desc())
> window_by_date = Window.partitionBy("date").orderBy("id")
> 
> # Apply window functions
> windowed_df = df.select(
>     col("*"),
>     row_number().over(window_by_category).alias("rank_in_category"),
>     dense_rank().over(window_by_category).alias("dense_rank_in_category"),
>     sum("amount").over(window_by_category).alias("running_total_by_category"),
>     avg("amount").over(window_by_category).alias("avg_in_category")
> )
> print("Window functions applied:")
> windowed_df.show()
> 
> # Lag and Lead functions
> lag_lead_df = df.select(
>     col("*"),
>     lag("amount", 1).over(window_by_date).alias("prev_amount"),
>     lead("amount", 1).over(window_by_date).alias("next_amount")
> )
> print("\nLag and Lead functions:")
> lag_lead_df.show()
> 
> spark.stop()
> ```
> 
> ## 3. Join Patterns
> 
> ```python join_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, broadcast, concat, lit, rand, floor
> 
> spark = SparkSession.builder \
>     .appName("JoinPatterns") \
>     .master("local[*]") \
>     .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB \
>     .getOrCreate()
> 
> # Sample data
> employees = [
>     (1, "Alice", 1),
>     (2, "Bob", 2),
>     (3, "Charlie", 1),
>     (4, "David", 3),
>     (5, "Eve", 2)
> ]
> 
> departments = [
>     (1, "Engineering"),
>     (2, "Sales"),
>     (3, "Marketing")
> ]
> 
> emp_df = spark.createDataFrame(employees, ["emp_id", "name", "dept_id"])
> dept_df = spark.createDataFrame(departments, ["dept_id", "dept_name"])
> 
> print("=" * 80)
> print("BROADCAST JOIN PATTERN")
> print("=" * 80)
> # Small table broadcast to all executors
> broadcast_join_df = emp_df.join(
>     broadcast(dept_df),
>     emp_df.dept_id == dept_df.dept_id,
>     "inner"
> ).select(emp_df.emp_id, emp_df.name, dept_df.dept_name)
> 
> print("Broadcast Join Result:")
> broadcast_join_df.show()
> print(f"Physical Plan:\n{broadcast_join_df._jdf.queryExecution().toString()}")
> 
> print("\n" + "=" * 80)
> print("SHUFFLE HASH JOIN PATTERN")
> print("=" * 80)
> # Force shuffle hash join (disable broadcast)
> spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
> spark.conf.set("spark.sql.join.preferSortMergeJoin", "false")
> 
> shuffle_hash_join_df = emp_df.join(
>     dept_df,
>     emp_df.dept_id == dept_df.dept_id,
>     "inner"
> )
> print("Shuffle Hash Join Result:")
> shuffle_hash_join_df.show()
> 
> print("\n" + "=" * 80)
> print("SORT MERGE JOIN PATTERN")
> print("=" * 80)
> # Enable sort merge join (default for large datasets)
> spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")
> 
> sort_merge_join_df = emp_df.join(
>     dept_df,
>     emp_df.dept_id == dept_df.dept_id,
>     "inner"
> )
> print("Sort Merge Join Result:")
> sort_merge_join_df.show()
> 
> print("\n" + "=" * 80)
> print("BUCKET JOIN PATTERN")
> print("=" * 80)
> # Create bucketed tables
> emp_df.write.mode("overwrite") \
>     .bucketBy(4, "dept_id") \
>     .sortBy("dept_id") \
>     .saveAsTable("bucketed_employees")
> 
> dept_df.write.mode("overwrite") \
>     .bucketBy(4, "dept_id") \
>     .sortBy("dept_id") \
>     .saveAsTable("bucketed_departments")
> 
> # Read bucketed tables
> bucketed_emp = spark.table("bucketed_employees")
> bucketed_dept = spark.table("bucketed_departments")
> 
> # Join bucketed tables (no shuffle needed)
> bucket_join_df = bucketed_emp.join(
>     bucketed_dept,
>     "dept_id"
> )
> print("Bucket Join Result:")
> bucket_join_df.show()
> 
> spark.stop()
> ```
> 
> ## 4. Skew Handling Patterns
> 
> ```python skew_handling.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, concat, lit, rand, floor, when, count
> 
> spark = SparkSession.builder \
>     .appName("SkewHandling") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("SKEWED JOIN PATTERN")
> print("=" * 80)
> 
> # Create skewed dataset (many records with same key)
> skewed_data = [(1, f"value_{i}") for i in range(1000)] + \
>               [(2, f"value_{i}") for i in range(10)] + \
>               [(3, f"value_{i}") for i in range(10)]
> 
> lookup_data = [(1, "Type_A"), (2, "Type_B"), (3, "Type_C")]
> 
> skewed_df = spark.createDataFrame(skewed_data, ["key", "value"])
> lookup_df = spark.createDataFrame(lookup_data, ["key", "type"])
> 
> # Check skew
> print("Data distribution:")
> skewed_df.groupBy("key").count().show()
> 
> # Identify skewed keys
> skew_threshold = 100
> skew_stats = skewed_df.groupBy("key").count()
> skewed_keys = skew_stats.filter(col("count") > skew_threshold).select("key").rdd.flatMap(lambda x: x).collect()
> 
> print(f"Skewed keys: {skewed_keys}")
> 
> # Handle skewed join by splitting into two joins
> if skewed_keys:
>     # Split data: skewed vs non-skewed
>     skewed_records = skewed_df.filter(col("key").isin(skewed_keys))
>     non_skewed_records = skewed_df.filter(~col("key").isin(skewed_keys))
>     
>     # Broadcast join for skewed keys
>     from pyspark.sql.functions import broadcast
>     skewed_join = skewed_records.join(
>         broadcast(lookup_df.filter(col("key").isin(skewed_keys))),
>         "key"
>     )
>     
>     # Regular join for non-skewed keys
>     non_skewed_join = non_skewed_records.join(lookup_df, "key")
>     
>     # Union results
>     final_result = skewed_join.union(non_skewed_join)
>     print("Skewed Join Result (first 10 rows):")
>     final_result.show(10)
> 
> print("\n" + "=" * 80)
> print("SALTING PATTERN")
> print("=" * 80)
> 
> # Salting: Add random salt to distribute skewed keys
> num_salts = 10
> 
> # Add salt to skewed data
> salted_df = skewed_df.withColumn(
>     "salt",
>     floor(rand() * num_salts).cast("int")
> ).withColumn(
>     "salted_key",
>     concat(col("key").cast("string"), lit("_"), col("salt").cast("string"))
> )
> 
> # Replicate lookup table with all salt values
> from pyspark.sql.functions import explode, array
> salt_array = array([lit(i) for i in range(num_salts)])
> replicated_lookup = lookup_df.withColumn("salt", explode(salt_array)) \
>     .withColumn("salted_key", concat(col("key").cast("string"), lit("_"), col("salt").cast("string")))
> 
> # Join on salted key
> salted_join_result = salted_df.join(
>     replicated_lookup,
>     "salted_key"
> ).select(
>     salted_df.key,
>     salted_df.value,
>     replicated_lookup.type
> )
> 
> print("Salted Join Result (first 10 rows):")
> salted_join_result.show(10)
> 
> # Verify result count
> print(f"Original count: {skewed_df.count()}")
> print(f"Salted join count: {salted_join_result.count()}")
> 
> spark.stop()
> ```
> 
> ## 5. Pivot and Union Patterns
> 
> ```python pivot_union.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, sum, avg
> 
> spark = SparkSession.builder \
>     .appName("PivotUnion") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("PIVOT PATTERN")
> print("=" * 80)
> 
> # Sample sales data
> sales_data = [
>     ("2024-01", "Electronics", 1000),
>     ("2024-01", "Clothing", 500),
>     ("2024-02", "Electronics", 1200),
>     ("2024-02", "Clothing", 600),
>     ("2024-03", "Electronics", 1100),
>     ("2024-03", "Clothing", 550)
> ]
> 
> sales_df = spark.createDataFrame(sales_data, ["month", "category", "sales"])
> 
> print("Original data:")
> sales_df.show()
> 
> # Pivot: Transform rows to columns
> pivoted_df = sales_df.groupBy("month").pivot("category").sum("sales")
> print("Pivoted data (categories as columns):")
> pivoted_df.show()
> 
> # Pivot with aggregation
> pivoted_agg_df = sales_df.groupBy("month").pivot("category").agg(
>     sum("sales").alias("total"),
>     avg("sales").alias("average")
> )
> 
> print("\n" + "=" * 80)
> print("UNPIVOT PATTERN")
> print("=" * 80)
> 
> # Unpivot: Transform columns back to rows
> from pyspark.sql.functions import expr, lit
> 
> # Create sample pivoted data
> pivoted_data = [
>     ("2024-01", 1000, 500),
>     ("2024-02", 1200, 600),
>     ("2024-03", 1100, 550)
> ]
> pivoted_sample = spark.createDataFrame(pivoted_data, ["month", "Electronics", "Clothing"])
> 
> print("Pivoted data:")
> pivoted_sample.show()
> 
> # Unpivot using stack function
> unpivoted_df = pivoted_sample.selectExpr(
>     "month",
>     "stack(2, 'Electronics', Electronics, 'Clothing', Clothing) as (category, sales)"
> )
> print("Unpivoted data:")
> unpivoted_df.show()
> 
> print("\n" + "=" * 80)
> print("UNION PATTERN")
> print("=" * 80)
> 
> # Create multiple dataframes
> df1 = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])
> df2 = spark.createDataFrame([(3, "C"), (4, "D")], ["id", "value"])
> df3 = spark.createDataFrame([(5, "E"), (6, "F")], ["id", "value"])
> 
> print("DataFrame 1:")
> df1.show()
> print("DataFrame 2:")
> df2.show()
> print("DataFrame 3:")
> df3.show()
> 
> # Union all dataframes
> union_df = df1.union(df2).union(df3)
> print("Union result:")
> union_df.show()
> 
> # Union by name (handles different column orders)
> df4 = spark.createDataFrame([("G", 7), ("H", 8)], ["value", "id"])
> union_by_name_df = df1.unionByName(df4)
> print("Union by name result:")
> union_by_name_df.show()
> 
> spark.stop()
> ```
> 
> ## 6. Partitioning and Caching Patterns
> 
> ```python partitioning_caching.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> from pyspark import StorageLevel
> import time
> 
> spark = SparkSession.builder \
>     .appName("PartitioningCaching") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> # Create sample data
> data = [(i, f"name_{i}", i % 10) for i in range(1000)]
> df = spark.createDataFrame(data, ["id", "name", "group"])
> 
> print("=" * 80)
> print("COALESCE PATTERN")
> print("=" * 80)
> 
> print(f"Original partitions: {df.rdd.getNumPartitions()}")
> 
> # Reduce number of partitions (no shuffle)
> coalesced_df = df.coalesce(2)
> print(f"After coalesce: {coalesced_df.rdd.getNumPartitions()}")
> 
> # Coalesce is useful after filtering to reduce small partitions
> filtered_df = df.filter(col("group") == 1)
> print(f"Filtered partitions: {filtered_df.rdd.getNumPartitions()}")
> coalesced_filtered = filtered_df.coalesce(1)
> print(f"Coalesced filtered partitions: {coalesced_filtered.rdd.getNumPartitions()}")
> 
> print("\n" + "=" * 80)
> print("REPARTITION PATTERN")
> print("=" * 80)
> 
> # Increase or rebalance partitions (with shuffle)
> repartitioned_df = df.repartition(8)
> print(f"After repartition: {repartitioned_df.rdd.getNumPartitions()}")
> 
> # Repartition by column (hash partitioning)
> repartitioned_by_col = df.repartition(4, "group")
> print(f"After repartition by column: {repartitioned_by_col.rdd.getNumPartitions()}")
> 
> # Check distribution
> print("\nDistribution across partitions:")
> repartitioned_by_col.rdd.mapPartitionsWithIndex(
>     lambda idx, it: [(idx, sum(1 for _ in it))]
> ).collect()
> 
> print("\n" + "=" * 80)
> print("CACHE/PERSIST PATTERN")
> print("=" * 80)
> 
> # Create a dataframe with expensive operations
> expensive_df = df.filter(col("group") < 5) \
>     .groupBy("group").count() \
>     .orderBy("group")
> 
> # Without caching
> start = time.time()
> result1 = expensive_df.count()
> time1 = time.time() - start
> print(f"First execution (no cache): {time1:.4f} seconds, count: {result1}")
> 
> start = time.time()
> result2 = expensive_df.count()
> time2 = time.time() - start
> print(f"Second execution (no cache): {time2:.4f} seconds, count: {result2}")
> 
> # With caching
> cached_df = expensive_df.cache()  # or persist(StorageLevel.MEMORY_AND_DISK)
> 
> start = time.time()
> result3 = cached_df.count()  # Triggers cache
> time3 = time.time() - start
> print(f"\nFirst execution (with cache): {time3:.4f} seconds, count: {result3}")
> 
> start = time.time()
> result4 = cached_df.count()  # Uses cache
> time4 = time.time() - start
> print(f"Second execution (from cache): {time4:.4f} seconds, count: {result4}")
> 
> # Different storage levels
> memory_only_df = df.persist(StorageLevel.MEMORY_ONLY)
> memory_and_disk_df = df.persist(StorageLevel.MEMORY_AND_DISK)
> disk_only_df = df.persist(StorageLevel.DISK_ONLY)
> 
> print("\nStorage Levels:")
> print("MEMORY_ONLY: Fast but limited by memory")
> print("MEMORY_AND_DISK: Spills to disk if memory full")
> print("DISK_ONLY: Slower but handles large data")
> print("MEMORY_AND_DISK_SER: Serialized to save memory")
> 
> # Unpersist when done
> cached_df.unpersist()
> 
> spark.stop()
> ```
> 
> ## 7. Architectural Patterns
> 
> ```python lazy_evaluation_pipeline.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, upper, length, current_timestamp, lit
> 
> spark = SparkSession.builder \
>     .appName("LazyEvaluationPipeline") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("LAZY EVALUATION PATTERN")
> print("=" * 80)
> 
> # Create dataframe
> data = [(1, "alice"), (2, "bob"), (3, "charlie")]
> df = spark.createDataFrame(data, ["id", "name"])
> 
> # These transformations are NOT executed yet (lazy)
> print("Defining transformations (no execution)...")
> df1 = df.filter(col("id") > 1)
> df2 = df1.select(col("id"), upper(col("name")).alias("name"))
> df3 = df2.withColumn("name_length", length(col("name")))
> 
> print("Transformations defined. Nothing executed yet.")
> print(f"Logical plan created: {df3._jdf.queryExecution().logical()}")
> 
> # Action triggers execution
> print("\nExecuting action (count)...")
> count = df3.count()
> print(f"Count: {count}")
> 
> # Another action
> print("\nExecuting action (show)...")
> df3.show()
> 
> print("\n" + "=" * 80)
> print("CHAIN OF RESPONSIBILITY PATTERN")
> print("=" * 80)
> 
> # Chain multiple transformations
> result = df \
>     .filter(col("id") > 0) \
>     .select(col("id"), upper(col("name")).alias("name")) \
>     .withColumn("name_length", length(col("name"))) \
>     .withColumn("processed_at", current_timestamp()) \
>     .orderBy("id")
> 
> print("Chain of transformations:")
> result.show()
> 
> print("\n" + "=" * 80)
> print("PIPELINE PATTERN")
> print("=" * 80)
> 
> # Define reusable transformation functions
> def clean_data(df):
>     """Remove nulls and duplicates"""
>     return df.dropna().dropDuplicates()
> 
> def normalize_names(df):
>     """Normalize name column"""
>     return df.withColumn("name", upper(col("name")))
> 
> def add_metadata(df):
>     """Add processing metadata"""
>     return df.withColumn("processed_at", current_timestamp()) \
>              .withColumn("version", lit("1.0"))
> 
> def calculate_metrics(df):
>     """Calculate derived metrics"""
>     return df.withColumn("name_length", length(col("name")))
> 
> # Build pipeline
> def data_pipeline(df):
>     """Complete data processing pipeline"""
>     return (df
>             .transform(clean_data)
>             .transform(normalize_names)
>             .transform(calculate_metrics)
>             .transform(add_metadata))
> 
> # Execute pipeline
> print("Executing data pipeline:")
> pipeline_result = data_pipeline(df)
> pipeline_result.show()
> 
> # Alternative: Using a list of functions
> transformations = [clean_data, normalize_names, calculate_metrics, add_metadata]
> 
> def apply_pipeline(df, transformations):
>     """Apply a list of transformations"""
>     result = df
>     for transform in transformations:
>         result = result.transform(transform)
>     return result
> 
> print("\nExecuting pipeline with function list:")
> result2 = apply_pipeline(df, transformations)
> result2.show()
> 
> spark.stop()
> ```
> 
> ## 8. Lambda and Kappa Architecture Patterns
> 
> ```python lambda_kappa_architecture.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, window, sum, avg, current_timestamp, from_json, to_json, struct
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
> 
> spark = SparkSession.builder \
>     .appName("LambdaKappaArchitecture") \
>     .master("local[*]") \
>     .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("LAMBDA ARCHITECTURE PATTERN")
> print("=" * 80)
> print("Lambda = Batch Layer + Speed Layer + Serving Layer")
> 
> # Simulate historical batch data
> batch_data = [
>     (1, "2024-01-01 10:00:00", "product_A", 100),
>     (2, "2024-01-01 11:00:00", "product_B", 150),
>     (3, "2024-01-01 12:00:00", "product_A", 200),
>     (4, "2024-01-02 10:00:00", "product_C", 300),
> ]
> 
> batch_df = spark.createDataFrame(
>     batch_data,
>     ["id", "timestamp", "product", "amount"]
> )
> batch_df = batch_df.withColumn("timestamp", col("timestamp").cast(TimestampType()))
> 
> print("\n--- BATCH LAYER (Historical Data Processing) ---")
> # Batch processing: Precompute aggregations
> batch_aggregations = batch_df.groupBy("product").agg(
>     sum("amount").alias("total_sales"),
>     avg("amount").alias("avg_sale"),
>     count("*").alias("num_transactions")
> )
> print("Batch layer results:")
> batch_aggregations.show()
> 
> # Simulate real-time streaming data
> print("\n--- SPEED LAYER (Real-time Data Processing) ---")
> realtime_data = [
>     (5, "2024-01-02 11:00:00", "product_A", 120),
>     (6, "2024-01-02 11:30:00", "product_B", 180),
> ]
> 
> realtime_df = spark.createDataFrame(
>     realtime_data,
>     ["id", "timestamp", "product", "amount"]
> )
> realtime_df = realtime_df.withColumn("timestamp", col("timestamp").cast(TimestampType()))
> 
> # Speed layer: Real-time aggregations
> realtime_aggregations = realtime_df.groupBy("product").agg(
>     sum("amount").alias("realtime_sales"),
>     count("*").alias("realtime_transactions")
> )
> print("Speed layer results:")
> realtime_aggregations.show()
> 
> print("\n--- SERVING LAYER (Merge Batch + Speed) ---")
> # Serving layer: Combine batch and speed layer results
> serving_layer = batch_aggregations.join(
>     realtime_aggregations,
>     "product",
>     "outer"
> ).fillna(0)
> 
> # Calculate final metrics
> from pyspark.sql.functions import coalesce
> final_view = serving_layer.select(
>     col("product"),
>     (col("total_sales") + col("realtime_sales")).alias("total_sales"),
>     (col("num_transactions") + col("realtime_transactions")).alias("total_transactions")
> )
> print("Serving layer (combined view):")
> final_view.show()
> 
> print("\n" + "=" * 80)
> print("KAPPA ARCHITECTURE PATTERN")
> print("=" * 80)
> print("Kappa = Single Stream Processing Layer (no separate batch layer)")
> 
> # In Kappa, everything is treated as a stream
> # Historical data is just old stream data
> 
> # Combined stream (both historical and realtime)
> all_data = batch_df.union(realtime_df)
> 
> print("\n--- STREAM PROCESSING LAYER ---")
> # Single processing pipeline for all data
> stream_aggregations = all_data.groupBy("product").agg(
>     sum("amount").alias("total_sales"),
>     avg("amount").alias("avg_sale"),
>     count("*").alias("num_transactions")
> )
> print("Stream processing results:")
> stream_aggregations.show()
> 
> # Time-based windowed aggregations
> windowed_aggregations = all_data.groupBy(
>     window(col("timestamp"), "1 day"),
>     "product"
> ).agg(
>     sum("amount").alias("daily_sales"),
>     count("*").alias("daily_transactions")
> )
> print("\nWindowed aggregations (daily):")
> windowed_aggregations.orderBy("window").show(truncate=False)
> 
> print("\n--- KAPPA WITH REPROCESSING ---")
> # In Kappa, reprocessing is done by replaying the stream
> # Simulate version 2 of processing logic
> def processing_v1(df):
>     """Original processing logic"""
>     return df.groupBy("product").agg(sum("amount").alias("total_sales"))
> 
> def processing_v2(df):
>     """Updated processing logic with additional metrics"""
>     return df.groupBy("product").agg(
>         sum("amount").alias("total_sales"),
>         avg("amount").alias("avg_sale"),
>         (sum("amount") / count("*")).alias("calculated_avg")
>     )
> 
> # Reprocess with new logic
> print("Processing V1:")
> processing_v1(all_data).show()
> 
> print("\nProcessing V2 (reprocessed):")
> processing_v2(all_data).show()
> 
> # Note: In production, you would use Kafka or similar for actual streaming
> print("\n--- PRODUCTION STREAMING EXAMPLE STRUCTURE ---")
> print("""
> # Structured Streaming (pseudo-code for Kafka)
> stream_df = spark.readStream \\
>     .format("kafka") \\
>     .option("kafka.bootstrap.servers", "localhost:9092") \\
>     .option("subscribe", "transactions") \\
>     .load()
> 
> # Process stream
> processed_stream = stream_df \\
>     .selectExpr("CAST(value AS STRING)") \\
>     .select(from_json(col("value"), schema).alias("data")) \\
>     .select("data.*") \\
>     .groupBy(window(col("timestamp"), "1 minute"), "product") \\
>     .agg(sum("amount").alias("total_sales"))
> 
> # Write results
> query = processed_stream.writeStream \\
>     .outputMode("update") \\
>     .format("console") \\
>     .start()
> """)
> 
> spark.stop()
> ```
> 
> ## 9. Complete Example - E-commerce Analytics
> 
> ```python complete_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.window import Window
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("CompleteExample") \
>     .master("local[*]") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("COMPLETE E-COMMERCE ANALYTICS PIPELINE")
> print("Using Multiple Patterns")
> print("=" * 80)
> 
> # Generate sample data
> orders_data = [
>     (1, 101, "2024-01-01", 250.00, "Electronics"),
>     (2, 102, "2024-01-01", 150.00, "Clothing"),
>     (3, 101, "2024-01-02", 300.00, "Electronics"),
>     (4, 103, "2024-01-02",

#### _User_

> Continue your response exactly where you left off:

#### _Assistant_

> ```python complete_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.window import Window
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("CompleteExample") \
>     .master("local[*]") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("COMPLETE E-COMMERCE ANALYTICS PIPELINE")
> print("Using Multiple Patterns")
> print("=" * 80)
> 
> # Generate sample data
> orders_data = [
>     (1, 101, "2024-01-01", 250.00, "Electronics"),
>     (2, 102, "2024-01-01", 150.00, "Clothing"),
>     (3, 101, "2024-01-02", 300.00, "Electronics"),
>     (4, 103, "2024-01-02", 75.00, "Books"),
>     (5, 102, "2024-01-03", 200.00, "Electronics"),
>     (6, 104, "2024-01-03", 450.00, "Electronics"),
>     (7, 101, "2024-01-04", 100.00, "Clothing"),
>     (8, 105, "2024-01-04", 80.00, "Books"),
> ]
> 
> customers_data = [
>     (101, "Alice", "Premium", "New York"),
>     (102, "Bob", "Standard", "Los Angeles"),
>     (103, "Charlie", "Premium", "Chicago"),
>     (104, "David", "Standard", "Houston"),
>     (105, "Eve", "Premium", "Phoenix"),
> ]
> 
> products_data = [
>     ("Electronics", 0.10),
>     ("Clothing", 0.15),
>     ("Books", 0.05),
> ]
> 
> orders_df = spark.createDataFrame(orders_data, 
>     ["order_id", "customer_id", "order_date", "amount", "category"])
> customers_df = spark.createDataFrame(customers_data, 
>     ["customer_id", "name", "tier", "city"])
> products_df = spark.createDataFrame(products_data, 
>     ["category", "commission_rate"])
> 
> orders_df = orders_df.withColumn("order_date", col("order_date").cast(DateType()))
> 
> # PATTERN 1: Cache frequently used data
> print("\n1. CACHE PATTERN - Caching customer dimension")
> customers_df.cache()
> customers_df.count()  # Trigger cache
> 
> # PATTERN 2: Broadcast Join (small dimension table)
> print("\n2. BROADCAST JOIN - Joining with customer data")
> orders_with_customers = orders_df.join(
>     broadcast(customers_df),
>     "customer_id",
>     "left"
> ).select(
>     "order_id",
>     "customer_id",
>     "name",
>     "tier",
>     "city",
>     "order_date",
>     "amount",
>     "category"
> )
> orders_with_customers.show(5)
> 
> # PATTERN 3: Filter-Transform Pattern
> print("\n3. FILTER-TRANSFORM - Premium customers only")
> premium_orders = orders_with_customers \
>     .filter(col("tier") == "Premium") \
>     .withColumn("vip_discount", col("amount") * 0.05) \
>     .withColumn("final_amount", col("amount") - col("vip_discount"))
> premium_orders.show(5)
> 
> # PATTERN 4: Aggregation Pattern
> print("\n4. AGGREGATION PATTERN - Sales by category and tier")
> category_tier_sales = orders_with_customers.groupBy("category", "tier").agg(
>     sum("amount").alias("total_sales"),
>     avg("amount").alias("avg_order_value"),
>     count("*").alias("num_orders"),
>     countDistinct("customer_id").alias("unique_customers")
> )
> category_tier_sales.show()
> 
> # PATTERN 5: Window Function Pattern
> print("\n5. WINDOW FUNCTION - Ranking and running totals")
> customer_window = Window.partitionBy("customer_id").orderBy("order_date")
> category_window = Window.partitionBy("category").orderBy(col("amount").desc())
> 
> orders_with_analytics = orders_with_customers.select(
>     "*",
>     row_number().over(customer_window).alias("order_sequence"),
>     sum("amount").over(customer_window).alias("customer_running_total"),
>     dense_rank().over(category_window).alias("category_rank"),
>     lag("amount", 1).over(customer_window).alias("previous_order_amount")
> )
> orders_with_analytics.show(10)
> 
> # PATTERN 6: Map-Reduce Pattern
> print("\n6. MAP-REDUCE - Total revenue by city")
> city_revenue = orders_with_customers \
>     .groupBy("city") \
>     .agg(sum("amount").alias("total_revenue")) \
>     .orderBy(col("total_revenue").desc())
> city_revenue.show()
> 
> # PATTERN 7: Join with commission rates
> print("\n7. SORT MERGE JOIN - Adding commission data")
> orders_with_commission = orders_with_customers.join(
>     products_df,
>     "category"
> ).select(
>     "order_id",
>     "customer_id",
>     "name",
>     "category",
>     "amount",
>     "commission_rate",
>     (col("amount") * col("commission_rate")).alias("commission_amount")
> )
> orders_with_commission.show(5)
> 
> # PATTERN 8: Pivot Pattern
> print("\n8. PIVOT PATTERN - Sales by customer and category")
> pivot_sales = orders_df.groupBy("customer_id").pivot("category").sum("amount").fillna(0)
> pivot_sales.show()
> 
> # PATTERN 9: Explode Pattern (create sample with arrays)
> print("\n9. EXPLODE PATTERN - Customer preferences")
> customer_preferences = [
>     (101, ["Electronics", "Books"]),
>     (102, ["Clothing", "Electronics"]),
>     (103, ["Books"]),
> ]
> pref_df = spark.createDataFrame(customer_preferences, ["customer_id", "preferences"])
> exploded_prefs = pref_df.select(
>     "customer_id",
>     explode("preferences").alias("preference")
> )
> exploded_prefs.show()
> 
> # PATTERN 10: Repartition by key for optimal processing
> print("\n10. REPARTITION PATTERN - Optimize for customer analysis")
> print(f"Original partitions: {orders_with_customers.rdd.getNumPartitions()}")
> repartitioned_orders = orders_with_customers.repartition(4, "customer_id")
> print(f"After repartition: {repartitioned_orders.rdd.getNumPartitions()}")
> 
> # PATTERN 11: Coalesce for output
> print("\n11. COALESCE PATTERN - Reduce partitions for output")
> final_output = category_tier_sales.coalesce(1)
> print(f"Output partitions: {final_output.rdd.getNumPartitions()}")
> 
> # PATTERN 12: Chain of Responsibility / Pipeline Pattern
> print("\n12. PIPELINE PATTERN - Complete transformation chain")
> 
> def clean_orders(df):
>     """Remove invalid orders"""
>     return df.filter((col("amount") > 0) & col("customer_id").isNotNull())
> 
> def enrich_with_customers(df, customers):
>     """Add customer information"""
>     return df.join(broadcast(customers), "customer_id", "left")
> 
> def calculate_metrics(df):
>     """Add calculated fields"""
>     return df.withColumn("tax", col("amount") * 0.08) \
>              .withColumn("total_with_tax", col("amount") + col("tax"))
> 
> def add_segments(df):
>     """Segment customers based on order value"""
>     return df.withColumn("segment",
>         when(col("amount") >= 300, "High Value")
>         .when(col("amount") >= 150, "Medium Value")
>         .otherwise("Low Value")
>     )
> 
> # Execute pipeline
> pipeline_result = (orders_df
>     .transform(clean_orders)
>     .transform(lambda df: enrich_with_customers(df, customers_df))
>     .transform(calculate_metrics)
>     .transform(add_segments)
> )
> 
> print("Pipeline transformation result:")
> pipeline_result.show(5)
> 
> # PATTERN 13: Union Pattern - Combine multiple periods
> print("\n13. UNION PATTERN - Combine Q1 and Q2 data")
> q1_orders = orders_df.filter(col("order_date") < "2024-02-01")
> q2_orders = orders_df.filter(col("order_date") >= "2024-02-01")
> all_orders = q1_orders.union(q2_orders)
> print(f"Q1 count: {q1_orders.count()}, Q2 count: {q2_orders.count()}, Total: {all_orders.count()}")
> 
> # PATTERN 14: Lazy Evaluation - Build complex query
> print("\n14. LAZY EVALUATION - Query plan optimization")
> complex_query = orders_with_customers \
>     .filter(col("amount") > 100) \
>     .groupBy("city", "category") \
>     .agg(sum("amount").alias("total")) \
>     .filter(col("total") > 200) \
>     .orderBy(col("total").desc())
> 
> print("Query built but not executed yet...")
> print("Executing now:")
> complex_query.show()
> 
> # PATTERN 15: Salting for skew (if we had skewed data)
> print("\n15. SALTING PATTERN EXAMPLE - Handling data skew")
> # Add salt to distribute heavy keys
> num_salts = 3
> salted_orders = orders_df.withColumn(
>     "salt", 
>     (rand() * num_salts).cast("int")
> ).withColumn(
>     "salted_category",
>     concat(col("category"), lit("_"), col("salt"))
> )
> print("Salted data (first 5 rows):")
> salted_orders.select("order_id", "category", "salted_category", "salt").show(5)
> 
> # Final Summary
> print("\n" + "=" * 80)
> print("SUMMARY OF PATTERNS DEMONSTRATED")
> print("=" * 80)
> patterns_used = [
>     "1. Cache/Persist Pattern - Cached dimension tables",
>     "2. Broadcast Join Pattern - Small table optimization",
>     "3. Filter-Transform Pattern - Data cleaning and enrichment",
>     "4. Aggregation Pattern - Group by operations",
>     "5. Window Function Pattern - Analytical functions",
>     "6. Map-Reduce Pattern - Distributed aggregation",
>     "7. Sort Merge Join Pattern - Large table joins",
>     "8. Pivot Pattern - Reshape data",
>     "9. Explode Pattern - Array expansion",
>     "10. Repartition Pattern - Optimize partition distribution",
>     "11. Coalesce Pattern - Reduce partitions",
>     "12. Pipeline Pattern - Chained transformations",
>     "13. Union Pattern - Combine datasets",
>     "14. Lazy Evaluation Pattern - Query optimization",
>     "15. Salting Pattern - Skew handling"
> ]
> 
> for pattern in patterns_used:
>     print(pattern)
> 
> # Cleanup
> customers_df.unpersist()
> spark.stop()
> 
> print("\n" + "=" * 80)
> print("Pipeline execution completed!")
> print("=" * 80)
> ```
> 
> ## 10. Performance Optimization Patterns
> 
> ```python performance_optimization.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> import time
> 
> spark = SparkSession.builder \
>     .appName("PerformanceOptimization") \
>     .master("local[*]") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .config("spark.sql.adaptive.skewJoin.enabled", "true") \
>     .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>     .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
>     .getOrCreate()
> 
> spark.sparkContext.setLogLevel("WARN")
> 
> print("=" * 80)
> print("PERFORMANCE OPTIMIZATION PATTERNS")
> print("=" * 80)
> 
> # Create larger sample dataset
> large_data = [(i, f"customer_{i % 1000}", i % 50, i * 10.0) 
>               for i in range(10000)]
> df = spark.createDataFrame(large_data, ["id", "customer", "category", "amount"])
> 
> print("\n--- OPTIMIZATION 1: Predicate Pushdown ---")
> # Bad: Filter after processing
> start = time.time()
> result_bad = df.select("*").filter(col("category") < 10).count()
> time_bad = time.time() - start
> 
> # Good: Filter early (predicate pushdown)
> start = time.time()
> result_good = df.filter(col("category") < 10).select("*").count()
> time_good = time.time() - start
> 
> print(f"Without optimization: {time_bad:.4f}s")
> print(f"With predicate pushdown: {time_good:.4f}s")
> 
> print("\n--- OPTIMIZATION 2: Column Pruning ---")
> # Bad: Select all columns when only few needed
> start = time.time()
> result_bad = df.select("*").groupBy("category").count().count()
> time_bad = time.time() - start
> 
> # Good: Select only required columns
> start = time.time()
> result_good = df.select("category").groupBy("category").count().count()
> time_good = time.time() - start
> 
> print(f"Without column pruning: {time_bad:.4f}s")
> print(f"With column pruning: {time_good:.4f}s")
> 
> print("\n--- OPTIMIZATION 3: Avoid UDFs (use built-in functions) ---")
> from pyspark.sql.types import IntegerType
> 
> # Bad: UDF
> @udf(returnType=IntegerType())
> def calculate_udf(amount):
>     return int(amount * 1.1)
> 
> start = time.time()
> result_bad = df.withColumn("new_amount", calculate_udf(col("amount"))).count()
> time_bad = time.time() - start
> 
> # Good: Built-in functions
> start = time.time()
> result_good = df.withColumn("new_amount", col("amount") * 1.1).count()
> time_good = time.time() - start
> 
> print(f"With UDF: {time_bad:.4f}s")
> print(f"With built-in function: {time_good:.4f}s")
> 
> print("\n--- OPTIMIZATION 4: Optimal Partition Size ---")
> # Too many small partitions
> df_many_partitions = df.repartition(100)
> start = time.time()
> df_many_partitions.groupBy("category").count().count()
> time_many = time.time() - start
> 
> # Optimal partitions (rule of thumb: 128MB per partition)
> df_optimal = df.repartition(4)
> start = time.time()
> df_optimal.groupBy("category").count().count()
> time_optimal = time.time() - start
> 
> print(f"With 100 partitions: {time_many:.4f}s")
> print(f"With 4 partitions: {time_optimal:.4f}s")
> 
> print("\n--- OPTIMIZATION 5: Caching Strategy ---")
> # Multiple operations on same dataframe
> base_df = df.filter(col("category") < 20)
> 
> # Without cache
> start = time.time()
> count1 = base_df.filter(col("amount") > 100).count()
> count2 = base_df.filter(col("amount") > 200).count()
> count3 = base_df.filter(col("amount") > 300).count()
> time_no_cache = time.time() - start
> 
> # With cache
> cached_df = base_df.cache()
> cached_df.count()  # Materialize cache
> 
> start = time.time()
> count1 = cached_df.filter(col("amount") > 100).count()
> count2 = cached_df.filter(col("amount") > 200).count()
> count3 = cached_df.filter(col("amount") > 300).count()
> time_with_cache = time.time() - start
> 
> cached_df.unpersist()
> 
> print(f"Without cache: {time_no_cache:.4f}s")
> print(f"With cache: {time_with_cache:.4f}s")
> 
> print("\n--- OPTIMIZATION 6: Avoid Shuffle When Possible ---")
> # Operations that cause shuffle
> print("Operations that cause shuffle:")
> print("- groupBy")
> print("- join (except broadcast join)")
> print("- repartition")
> print("- distinct")
> print("- orderBy (global sort)")
> 
> # Use coalesce instead of repartition when reducing partitions
> df_repartitioned = df.repartition(8)
> print(f"\nPartitions after repartition(8): {df_repartitioned.rdd.getNumPartitions()}")
> 
> # Reducing partitions - coalesce is faster (no shuffle)
> df_coalesced = df_repartitioned.coalesce(2)
> print(f"Partitions after coalesce(2): {df_coalesced.rdd.getNumPartitions()}")
> 
> print("\n--- OPTIMIZATION 7: Adaptive Query Execution (AQE) ---")
> print("AQE Features:")
> print("1. Dynamically coalesce shuffle partitions")
> print("2. Dynamically switch join strategies")
> print("3. Dynamically optimize skew joins")
> print(f"AQE Enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")
> 
> print("\n--- OPTIMIZATION 8: Data Skew Handling ---")
> # Create skewed data
> skewed_data = [(1, "common_key", i) for i in range(8000)] + \
>               [(i, f"key_{i}", i) for i in range(2, 3000)]
> skewed_df = spark.createDataFrame(skewed_data, ["id", "key", "value"])
> 
> print("Data distribution:")
> skewed_df.groupBy("key").count().orderBy(col("count").desc()).show(5)
> 
> # Enable skew join optimization
> spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
> spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
> 
> print("\n--- Best Practices Summary ---")
> best_practices = """
> 1. Filter early (predicate pushdown)
> 2. Select only needed columns (column pruning)
> 3. Use built-in functions instead of UDFs
> 4. Cache intermediate results used multiple times
> 5. Use broadcast joins for small tables
> 6. Avoid shuffles when possible
> 7. Use optimal partition sizes (128MB per partition)
> 8. Enable Adaptive Query Execution
> 9. Handle data skew with salting or AQE
> 10. Coalesce instead of repartition when reducing partitions
> 11. Use appropriate storage formats (Parquet, ORC)
> 12. Partition data by commonly filtered columns
> 13. Use bucketing for join optimization
> 14. Monitor and tune spark.sql.shuffle.partitions
> 15. Avoid small files problem
> """
> print(best_practices)
> 
> spark.stop()
> ```
> 
> ## 11. Advanced Join Patterns Details
> 
> ```python advanced_joins.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> 
> spark = SparkSession.builder \
>     .appName("AdvancedJoins") \
>     .master("local[*]") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("ADVANCED JOIN PATTERNS - DETAILED EXAMPLES")
> print("=" * 80)
> 
> # Sample datasets
> employees = [(1, "Alice", 1, 50000),
>              (2, "Bob", 2, 60000),
>              (3, "Charlie", 1, 55000),
>              (4, "David", 3, 70000),
>              (5, "Eve", 2, 65000)]
> 
> departments = [(1, "Engineering", "Building A"),
>                (2, "Sales", "Building B"),
>                (3, "Marketing", "Building C"),
>                (4, "HR", "Building D")]
> 
> emp_df = spark.createDataFrame(employees, ["emp_id", "name", "dept_id", "salary"])
> dept_df = spark.createDataFrame(departments, ["dept_id", "dept_name", "location"])
> 
> print("\n--- JOIN TYPE 1: BROADCAST HASH JOIN ---")
> print("Best for: Small table (< 10MB) joining with large table")
> print("Characteristics: No shuffle, broadcasts small table to all executors\n")
> 
> broadcast_result = emp_df.join(
>     broadcast(dept_df),
>     emp_df.dept_id == dept_df.dept_id,
>     "inner"
> ).select(emp_df["*"], dept_df.dept_name, dept_df.location)
> 
> broadcast_result.show()
> print(f"Join type used: {broadcast_result._jdf.queryExecution().executedPlan()}")
> 
> print("\n--- JOIN TYPE 2: SHUFFLE HASH JOIN ---")
> print("Best for: Medium-sized tables, build side much smaller than probe side")
> print("Characteristics: Shuffles data, builds hash table from smaller side\n")
> 
> spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
> spark.conf.set("spark.sql.join.preferSortMergeJoin", "false")
> 
> shuffle_hash_result = emp_df.join(dept_df, "dept_id")
> shuffle_hash_result.show()
> 
> print("\n--- JOIN TYPE 3: SORT MERGE JOIN ---")
> print("Best for: Large tables of similar size")
> print("Characteristics: Sorts both sides, then merges\n")
> 
> spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")
> 
> sort_merge_result = emp_df.join(dept_df, "dept_id")
> sort_merge_result.show()
> 
> print("\n--- JOIN TYPE 4: CARTESIAN JOIN (CROSS JOIN) ---")
> print("Warning: Produces N×M rows, use with caution!\n")
> 
> # Small example to avoid explosion
> small_emp = emp_df.limit(2)
> small_dept = dept_df.limit(2)
> 
> cartesian_result = small_emp.crossJoin(small_dept)
> print(f"Employees: {small_emp.count()}, Departments: {small_dept.count()}")
> print(f"Cartesian result: {cartesian_result.count()} rows")
> cartesian_result.show()
> 
> print("\n--- JOIN TYPE 5: SEMI JOIN ---")
> print("Returns rows from left table where join condition is met")
> print("Does NOT include columns from right table\n")
> 
> # Find employees who have departments
> semi_join = emp_df.join(dept_df, emp_df.dept_id == dept_df.dept_id, "left_semi")
> semi_join.show()
> 
> print("\n--- JOIN TYPE 6: ANTI JOIN ---")
> print("Returns rows from left table where join condition is NOT met\n")
> 
> # Find employees without departments (none in this case)
> anti_join = emp_df.join(dept_df, emp_df.dept_id == dept_df.dept_id, "left_anti")
> print(f"Employees without departments: {anti_join.count()}")
> 
> # Create example with unmatched data
> emp_with_invalid = emp_df.union(
>     spark.createDataFrame([(6, "Frank", 99, 45000)], emp_df.schema)
> )
> anti_join = emp_with_invalid.join(dept_df, emp_with_invalid.dept_id == dept_df.dept_id, "left_anti")
> print("Employees with invalid department:")
> anti_join.show()
> 
> print("\n--- JOIN TYPE 7: SELF JOIN ---")
> print("Join table with itself\n")
> 
> # Find employees in same department
> self_join = emp_df.alias("e1").join(
>     emp_df.alias("e2"),
>     (col("e1.dept_id") == col("e2.dept_id")) & (col("e1.emp_id") < col("e2.emp_id"))
> ).select(
>     col("e1.name").alias("employee1"),
>     col("e2.name").alias("employee2"),
>     col("e1.dept_id")
> )
> print("Employee pairs in same department:")
> self_join.show()
> 
> print("\n--- JOIN OPTIMIZATION: Bucket Join ---")
> # Write bucketed tables (for demonstration)
> emp_df.write.mode("overwrite") \
>     .bucketBy(4, "dept_id") \
>     .sortBy("dept_id") \
>     .format("parquet") \
>     .saveAsTable("bucketed_emp")
> 
> dept_df.write.mode("overwrite") \
>     .bucketBy(4, "dept_id") \
>     .sortBy("dept_id") \
>     .format("parquet") \
>     .saveAsTable("bucketed_dept")
> 
> bucketed_emp = spark.table("bucketed_emp")
> bucketed_dept = spark.table("bucketed_dept")
> 
> bucket_join = bucketed_emp.join(bucketed_dept, "dept_id")
> print("Bucket join eliminates shuffle:")
> bucket_join.show()
> 
> spark.stop()
> ```
> 
> All patterns have been implemented! Here's what each file demonstrates:
> 
> 1. **basic_transformations.py** - Map-Reduce, Filter-Transform, Explode
> 2. **aggregation_window.py** - Aggregation, Window Functions
> 3. **join_patterns.py** - Broadcast, Shuffle Hash, Sort Merge, Bucket Joins
> 4. **skew_handling.py** - Skewed Join, Salting patterns
> 5. **pivot_union.py** - Pivot/Unpivot, Union patterns
> 6. **partitioning_caching.py** - Coalesce, Repartition, Cache/Persist
> 7. **lazy_evaluation_pipeline.py** - Lazy Evaluation, Chain of Responsibility, Pipeline
> 8. **lambda_kappa_architecture.py** - Lambda & Kappa architectures
> 9. **complete_example.py** - Comprehensive e-commerce example using multiple patterns
> 10. **performance_optimization.py** - Best practices and optimization techniques
> 11. **advanced_joins.py** - Detailed join pattern examples
> 
> You can run each file independently to see the patterns in action!