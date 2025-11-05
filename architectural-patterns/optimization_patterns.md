### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 7:08:58 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Optimization Patterns
> - Catalyst Optimizer Pattern
> - Adaptive Query Execution (AQE) Pattern
> - Dynamic Partition Pruning Pattern
> - Predicate Pushdown Pattern
> - Projection Pushdown Pattern
> - Column Pruning Pattern
> - Constant Folding Pattern
> - Cost-Based Optimization Pattern
> - Broadcast Variable Pattern
> - Accumulator Pattern
> - Caching Strategy Pattern
> - Persistence Level Pattern
> - Tungsten Optimization Pattern
> - Whole-Stage Code Generation Pattern
> - Vectorized Execution Pattern
> ```

#### _Assistant_

> I'll create comprehensive PySpark examples for each optimization pattern:
> 
> ```python optimization_patterns/catalyst_optimizer.py
> """
> Catalyst Optimizer Pattern
> - Demonstrates how Catalyst automatically optimizes logical plans
> - Shows explain() to view optimization stages
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, lit
> 
> spark = SparkSession.builder \
>     .appName("CatalystOptimizerPattern") \
>     .getOrCreate()
> 
> # Create sample data
> df = spark.createDataFrame([
>     (1, "Alice", 100),
>     (2, "Bob", 200),
>     (3, "Charlie", 150),
>     (4, "David", 300)
> ], ["id", "name", "amount"])
> 
> # Catalyst will optimize this query by:
> # 1. Combining filters
> # 2. Pushing filters down
> # 3. Eliminating redundant operations
> result = df.filter(col("amount") > 50) \
>     .filter(col("amount") < 250) \
>     .select("name", "amount") \
>     .filter(col("id") > 0)  # Redundant filter will be optimized
> 
> # View the optimized logical and physical plans
> print("=== Parsed Logical Plan ===")
> result.explain(extended=True)
> 
> # Execute the query
> result.show()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/aqe_pattern.py
> """
> Adaptive Query Execution (AQE) Pattern
> - Dynamically optimizes query execution at runtime
> - Adjusts shuffle partitions, join strategies, and handles skew
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> 
> spark = SparkSession.builder \
>     .appName("AQEPattern") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>     .config("spark.sql.adaptive.skewJoin.enabled", "true") \
>     .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
>     .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
>     .getOrCreate()
> 
> # Create sample datasets
> large_df = spark.range(0, 1000000).withColumnRenamed("id", "user_id")
> small_df = spark.range(0, 100).withColumnRenamed("id", "product_id")
> 
> # AQE will:
> # 1. Automatically convert sort-merge join to broadcast join if applicable
> # 2. Coalesce shuffle partitions after filtering
> # 3. Handle skewed data in joins
> result = large_df.filter(col("user_id") < 10000) \
>     .join(small_df, large_df.user_id == small_df.product_id, "inner")
> 
> print("=== AQE Optimized Plan ===")
> result.explain(mode="formatted")
> 
> result.show(10)
> 
> # Check final number of partitions (AQE may have coalesced them)
> print(f"Final partitions: {result.rdd.getNumPartitions()}")
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/dynamic_partition_pruning.py
> """
> Dynamic Partition Pruning Pattern
> - Reduces data scan by pruning partitions at runtime based on join conditions
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import os
> import shutil
> 
> spark = SparkSession.builder \
>     .appName("DynamicPartitionPruning") \
>     .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true") \
>     .getOrCreate()
> 
> # Create partitioned fact table
> fact_data = [(i, f"2024-{(i % 12) + 1:02d}-01", i * 100) 
>              for i in range(1, 1001)]
> fact_df = spark.createDataFrame(fact_data, ["id", "date", "amount"])
> 
> # Write as partitioned table
> output_path = "/tmp/partitioned_sales"
> if os.path.exists(output_path):
>     shutil.rmtree(output_path)
> 
> fact_df.write.partitionBy("date").parquet(output_path)
> 
> # Read partitioned data
> sales_df = spark.read.parquet(output_path)
> 
> # Create dimension table (small)
> dim_data = [(f"2024-01-01",), (f"2024-02-01",), (f"2024-03-01",)]
> dim_df = spark.createDataFrame(dim_data, ["filter_date"])
> 
> # DPP will prune partitions based on the join with dimension table
> # Only partitions matching filter_date will be scanned
> result = sales_df.join(dim_df, sales_df.date == dim_df.filter_date)
> 
> print("=== Dynamic Partition Pruning Plan ===")
> result.explain(mode="formatted")
> 
> result.show()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/predicate_pushdown.py
> """
> Predicate Pushdown Pattern
> - Pushes filters down to data source to reduce data transfer
> - Works with Parquet, ORC, JDBC, and other sources
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import shutil
> 
> spark = SparkSession.builder \
>     .appName("PredicatePushdown") \
>     .getOrCreate()
> 
> # Create and save sample data
> data = [(i, f"user_{i}", i * 10) for i in range(1, 10001)]
> df = spark.createDataFrame(data, ["id", "name", "score"])
> 
> output_path = "/tmp/predicate_pushdown_data"
> if os.path.exists(output_path):
>     shutil.rmtree(output_path)
> 
> df.write.parquet(output_path)
> 
> # Read with predicate pushdown
> # Filter is pushed to Parquet reader, reducing I/O
> filtered_df = spark.read.parquet(output_path) \
>     .filter(col("score") > 5000) \
>     .filter(col("id") < 800)
> 
> print("=== Predicate Pushdown to Parquet ===")
> filtered_df.explain(mode="formatted")
> 
> # The Physical Plan will show "PushedFilters" in the scan operation
> filtered_df.show(10)
> 
> # Example with JDBC (uncomment if you have a database)
> # jdbc_df = spark.read \
> #     .format("jdbc") \
> #     .option("url", "jdbc:postgresql://localhost:5432/mydb") \
> #     .option("dbtable", "users") \
> #     .option("user", "user") \
> #     .option("password", "pass") \
> #     .load() \
> #     .filter(col("age") > 25)  # This filter is pushed to the database
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/projection_pushdown.py
> """
> Projection Pushdown Pattern
> - Only reads required columns from data source
> - Reduces I/O and memory usage
> """
> from pyspark.sql import SparkSession
> import shutil
> 
> spark = SparkSession.builder \
>     .appName("ProjectionPushdown") \
>     .getOrCreate()
> 
> # Create dataset with many columns
> data = [(i, f"name_{i}", i * 10, f"email_{i}@test.com", 
>          f"address_{i}", f"phone_{i}", i % 100) 
>         for i in range(1, 10001)]
> 
> df = spark.createDataFrame(data, 
>     ["id", "name", "score", "email", "address", "phone", "category"])
> 
> output_path = "/tmp/projection_pushdown_data"
> if os.path.exists(output_path):
>     shutil.rmtree(output_path)
> 
> df.write.parquet(output_path)
> 
> # Only select needed columns - projection is pushed to Parquet reader
> # Parquet will only read the "name" and "score" columns from disk
> result = spark.read.parquet(output_path) \
>     .select("name", "score")  # Only these columns are read from storage
> 
> print("=== Projection Pushdown ===")
> result.explain(mode="formatted")
> 
> # Physical plan will show only required columns in scan
> result.show(10)
> 
> # Contrast: Reading all columns then selecting
> inefficient = spark.read.parquet(output_path)
> inefficient_select = inefficient.select("name", "score")
> 
> print("\n=== Without early projection (less efficient) ===")
> inefficient_select.explain(mode="formatted")
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/column_pruning.py
> """
> Column Pruning Pattern
> - Eliminates unused columns early in the query plan
> - Reduces memory and processing overhead
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, sum as _sum
> 
> spark = SparkSession.builder \
>     .appName("ColumnPruning") \
>     .getOrCreate()
> 
> # Create wide dataset
> data = [(i, f"name_{i}", i * 10, i * 20, i * 30, i * 40, i % 10) 
>         for i in range(1, 1001)]
> 
> df = spark.createDataFrame(data, 
>     ["id", "name", "col1", "col2", "col3", "col4", "category"])
> 
> # Catalyst will prune unused columns throughout the plan
> # Only columns used in final output and intermediate operations are kept
> result = df.filter(col("category") < 5) \
>     .groupBy("category") \
>     .agg(_sum("col1").alias("total")) \
>     .select("category", "total")
> 
> # Columns: name, col2, col3, col4 are pruned early
> print("=== Column Pruning ===")
> result.explain(mode="formatted")
> 
> result.show()
> 
> # Anti-pattern: Using select * keeps all columns
> anti_pattern = df.select("*") \
>     .filter(col("category") < 5) \
>     .groupBy("category") \
>     .agg(_sum("col1").alias("total"))
> 
> print("\n=== Without Column Pruning (Anti-pattern) ===")
> anti_pattern.explain(mode="formatted")
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/constant_folding.py
> """
> Constant Folding Pattern
> - Evaluates constant expressions at compile time
> - Reduces runtime computation
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, lit
> 
> spark = SparkSession.builder \
>     .appName("ConstantFolding") \
>     .getOrCreate()
> 
> df = spark.createDataFrame([
>     (1, 100),
>     (2, 200),
>     (3, 300)
> ], ["id", "amount"])
> 
> # Constant expressions are evaluated at optimization time
> # (10 * 5 + 20) is computed once, not for each row
> result = df.select(
>     col("id"),
>     col("amount"),
>     (col("amount") * (lit(10) * lit(5) + lit(20))).alias("calculated"),
>     (lit(100) / lit(4)).alias("constant_ratio")
> )
> 
> print("=== Constant Folding Optimization ===")
> result.explain(mode="formatted")
> # You'll see constants pre-computed in the plan
> 
> result.show()
> 
> # More complex example
> result2 = df.filter(col("amount") > (lit(50) * lit(2))) \
>     .select(
>         col("id"),
>         (col("amount") + (lit(10) + lit(20) + lit(30))).alias("total")
>     )
> 
> print("\n=== Complex Constant Folding ===")
> result2.explain(mode="formatted")
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/cost_based_optimization.py
> """
> Cost-Based Optimization (CBO) Pattern
> - Uses statistics to choose optimal join strategies and query plans
> - Requires collecting table statistics
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> 
> spark = SparkSession.builder \
>     .appName("CostBasedOptimization") \
>     .config("spark.sql.cbo.enabled", "true") \
>     .config("spark.sql.cbo.joinReorder.enabled", "true") \
>     .config("spark.sql.statistics.histogram.enabled", "true") \
>     .getOrCreate()
> 
> # Create tables
> large_table = spark.range(0, 100000).withColumnRenamed("id", "user_id")
> medium_table = spark.range(0, 10000).withColumnRenamed("id", "product_id")
> small_table = spark.range(0, 100).withColumnRenamed("id", "category_id")
> 
> # Register as temp views
> large_table.createOrReplaceTempView("users")
> medium_table.createOrReplaceTempView("products")
> small_table.createOrReplaceTempView("categories")
> 
> # Analyze tables to collect statistics (crucial for CBO)
> spark.sql("ANALYZE TABLE users COMPUTE STATISTICS FOR ALL COLUMNS")
> spark.sql("ANALYZE TABLE products COMPUTE STATISTICS FOR ALL COLUMNS")
> spark.sql("ANALYZE TABLE categories COMPUTE STATISTICS FOR ALL COLUMNS")
> 
> # CBO will use statistics to:
> # 1. Determine optimal join order
> # 2. Choose broadcast vs. shuffle join
> # 3. Estimate cardinality
> query = """
>     SELECT u.user_id, p.product_id, c.category_id
>     FROM users u
>     JOIN products p ON u.user_id = p.product_id
>     JOIN categories c ON p.product_id = c.category_id
>     WHERE u.user_id < 1000
> """
> 
> result = spark.sql(query)
> 
> print("=== Cost-Based Optimization Plan ===")
> result.explain(mode="cost")
> 
> result.show(10)
> 
> # View statistics
> spark.sql("DESCRIBE EXTENDED users").show(truncate=False)
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/broadcast_variable.py
> """
> Broadcast Variable Pattern
> - Efficiently distributes read-only data to all executors
> - Avoids shuffling small datasets in joins and lookups
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import broadcast, col
> 
> spark = SparkSession.builder \
>     .appName("BroadcastVariable") \
>     .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
>     .getOrCreate()
> 
> # Large fact table
> large_df = spark.range(0, 1000000).withColumnRenamed("id", "user_id")
> 
> # Small dimension table (< 10MB)
> small_df = spark.range(0, 100).withColumnRenamed("id", "category_id") \
>     .selectExpr("category_id", "concat('Category_', category_id) as category_name")
> 
> # Method 1: Explicit broadcast hint
> broadcast_join = large_df.join(
>     broadcast(small_df),
>     large_df.user_id == small_df.category_id,
>     "left"
> )
> 
> print("=== Broadcast Join Plan ===")
> broadcast_join.explain(mode="formatted")
> 
> broadcast_join.show(10)
> 
> # Method 2: Using broadcast variables for lookups
> lookup_dict = {i: f"Category_{i}" for i in range(100)}
> broadcast_dict = spark.sparkContext.broadcast(lookup_dict)
> 
> # Use in UDF
> from pyspark.sql.functions import udf
> from pyspark.sql.types import StringType
> 
> @udf(returnType=StringType())
> def lookup_category(user_id):
>     return broadcast_dict.value.get(user_id % 100, "Unknown")
> 
> result = large_df.withColumn("category", lookup_category(col("user_id")))
> 
> result.show(10)
> 
> # Clean up
> broadcast_dict.unpersist()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/accumulator_pattern.py
> """
> Accumulator Pattern
> - Aggregates values across executors for monitoring and debugging
> - Used for counters, metrics, and custom aggregations
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, udf
> from pyspark.sql.types import IntegerType
> 
> spark = SparkSession.builder \
>     .appName("AccumulatorPattern") \
>     .getOrCreate()
> 
> # Create accumulators
> processed_count = spark.sparkContext.accumulator(0)
> error_count = spark.sparkContext.accumulator(0)
> large_value_count = spark.sparkContext.accumulator(0)
> 
> # Custom accumulator for complex aggregation
> class SetAccumulator:
>     def __init__(self):
>         self.value = set()
>     
>     def add(self, item):
>         self.value.add(item)
>     
>     def merge(self, other):
>         self.value.update(other.value)
> 
> # Create sample data
> data = [(i, i * 10 if i % 10 != 0 else None) for i in range(1, 1001)]
> df = spark.createDataFrame(data, ["id", "value"])
> 
> # Define UDF that uses accumulators
> @udf(returnType=IntegerType())
> def process_value(value):
>     processed_count.add(1)
>     
>     if value is None:
>         error_count.add(1)
>         return -1
>     
>     if value > 500:
>         large_value_count.add(1)
>     
>     return value * 2
> 
> # Apply transformation
> result = df.withColumn("processed", process_value(col("value")))
> 
> # Trigger action to execute
> result.count()
> 
> print(f"Total processed: {processed_count.value}")
> print(f"Errors encountered: {error_count.value}")
> print(f"Large values: {large_value_count.value}")
> 
> # Use accumulator for filtering metrics
> valid_records = spark.sparkContext.accumulator(0)
> invalid_records = spark.sparkContext.accumulator(0)
> 
> def validate_and_count(row):
>     if row.value is not None and row.value > 0:
>         valid_records.add(1)
>         return True
>     else:
>         invalid_records.add(1)
>         return False
> 
> filtered_rdd = df.rdd.filter(validate_and_count)
> filtered_rdd.count()
> 
> print(f"\nValid records: {valid_records.value}")
> print(f"Invalid records: {invalid_records.value}")
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/caching_strategy.py
> """
> Caching Strategy Pattern
> - Caches frequently accessed DataFrames in memory
> - Improves performance for iterative algorithms
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import time
> 
> spark = SparkSession.builder \
>     .appName("CachingStrategy") \
>     .config("spark.memory.fraction", "0.8") \
>     .getOrCreate()
> 
> # Create base DataFrame
> base_df = spark.range(0, 1000000).withColumnRenamed("id", "user_id")
> 
> # Strategy 1: Cache for multiple transformations
> cached_df = base_df.filter(col("user_id") < 500000).cache()
> 
> start = time.time()
> count1 = cached_df.count()
> end1 = time.time()
> print(f"First count (caching): {end1 - start:.2f}s")
> 
> start = time.time()
> count2 = cached_df.filter(col("user_id") > 100000).count()
> end2 = time.time()
> print(f"Second operation (cached): {end2 - start:.2f}s")
> 
> # Strategy 2: Cache at checkpoints in complex pipelines
> df1 = spark.range(0, 100000)
> df2 = spark.range(50000, 150000)
> 
> # Cache the join result if used multiple times
> joined = df1.join(df2, df1.id == df2.id).cache()
> 
> result1 = joined.filter(col("id") < 75000).count()
> result2 = joined.filter(col("id") > 60000).count()
> 
> print(f"Cached join results: {result1}, {result2}")
> 
> # Strategy 3: Selective caching based on usage patterns
> # Cache only if DataFrame is reused 2+ times
> df = spark.range(0, 10000)
> 
> # Don't cache - used once
> one_time_use = df.filter(col("id") < 5000)
> one_time_use.show(5)
> 
> # Cache - used multiple times
> multi_use = df.filter(col("id") < 8000).cache()
> multi_use.count()
> multi_use.filter(col("id") > 4000).count()
> multi_use.show(5)
> 
> # Check what's cached
> print("\n=== Cached Tables ===")
> spark.catalog.listTables()
> 
> # Unpersist when done
> cached_df.unpersist()
> joined.unpersist()
> multi_use.unpersist()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/persistence_level.py
> """
> Persistence Level Pattern
> - Chooses appropriate storage level based on use case
> - Balances memory, disk, and serialization
> """
> from pyspark.sql import SparkSession
> from pyspark import StorageLevel
> import time
> 
> spark = SparkSession.builder \
>     .appName("PersistenceLevel") \
>     .config("spark.memory.fraction", "0.6") \
>     .getOrCreate()
> 
> df = spark.range(0, 1000000)
> 
> # Level 1: MEMORY_ONLY (default cache)
> # Fast but may cause recomputation if memory is insufficient
> memory_only = df.filter("id < 500000").persist(StorageLevel.MEMORY_ONLY)
> memory_only.count()
> print("MEMORY_ONLY: Fast access, risk of recomputation")
> 
> # Level 2: MEMORY_AND_DISK
> # Spills to disk if memory is full
> memory_and_disk = df.filter("id < 600000").persist(StorageLevel.MEMORY_AND_DISK)
> memory_and_disk.count()
> print("MEMORY_AND_DISK: Balanced approach")
> 
> # Level 3: MEMORY_ONLY_SER
> # Serialized storage - saves memory but adds CPU overhead
> memory_ser = df.filter("id < 700000").persist(StorageLevel.MEMORY_ONLY_SER)
> memory_ser.count()
> print("MEMORY_ONLY_SER: Space efficient, slower access")
> 
> # Level 4: DISK_ONLY
> # For large datasets that don't fit in memory
> disk_only = df.filter("id < 800000").persist(StorageLevel.DISK_ONLY)
> disk_only.count()
> print("DISK_ONLY: Slower but handles large data")
> 
> # Level 5: OFF_HEAP
> # Uses off-heap memory, avoids GC overhead
> # Requires: spark.memory.offHeap.enabled=true
> off_heap = df.filter("id < 900000").persist(StorageLevel.OFF_HEAP)
> off_heap.count()
> print("OFF_HEAP: Avoids GC, requires configuration")
> 
> # Level 6: MEMORY_AND_DISK_SER
> # Serialized with disk spillover - good for large iterative workloads
> memory_disk_ser = df.filter("id < 950000").persist(StorageLevel.MEMORY_AND_DISK_SER)
> memory_disk_ser.count()
> print("MEMORY_AND_DISK_SER: Best for large iterative jobs")
> 
> # Strategy selection guide
> def choose_persistence_level(data_size_gb, reuse_count, memory_available_gb):
>     """
>     Helper to choose persistence level
>     """
>     if data_size_gb < memory_available_gb * 0.3:
>         if reuse_count > 5:
>             return StorageLevel.MEMORY_ONLY
>         else:
>             return StorageLevel.MEMORY_AND_DISK
>     elif data_size_gb < memory_available_gb:
>         return StorageLevel.MEMORY_AND_DISK_SER
>     else:
>         return StorageLevel.DISK_ONLY
> 
> # Example usage
> level = choose_persistence_level(
>     data_size_gb=10,
>     reuse_count=3,
>     memory_available_gb=8
> )
> print(f"\nRecommended level: {level}")
> 
> # Cleanup
> for df_cached in [memory_only, memory_and_disk, memory_ser, 
>                    disk_only, off_heap, memory_disk_ser]:
>     df_cached.unpersist()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/tungsten_optimization.py
> """
> Tungsten Optimization Pattern
> - Demonstrates Tungsten's memory management and binary processing
> - Shows efficiency of Tungsten's execution engine
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, sum as _sum, avg, count
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType
> 
> spark = SparkSession.builder \
>     .appName("TungstenOptimization") \
>     .config("spark.sql.tungsten.enabled", "true") \
>     .config("spark.sql.codegen.wholeStage", "true") \
>     .config("spark.sql.codegen.factoryMode", "CODEGEN_ONLY") \
>     .getOrCreate()
> 
> # Tungsten benefits:
> # 1. Binary in-memory format (no serialization overhead)
> # 2. Cache-aware computation
> # 3. Whole-stage code generation
> 
> # Create structured data - Tungsten optimizes this
> schema = StructType([
>     StructField("id", IntegerType(), False),
>     StructField("category", StringType(), False),
>     StructField("value", IntegerType(), False)
> ])
> 
> data = [(i, f"cat_{i % 10}", i * 10) for i in range(1, 1000001)]
> df = spark.createDataFrame(data, schema)
> 
> # Tungsten optimizes:
> # - Binary format reduces memory by 2-5x vs Java objects
> # - Off-heap memory management
> # - Cache-friendly memory layout
> 
> result = df.groupBy("category") \
>     .agg(
>         _sum("value").alias("total"),
>         avg("value").alias("average"),
>         count("*").alias("count")
>     ) \
>     .filter(col("total") > 1000000)
> 
> print("=== Tungsten Optimized Plan ===")
> result.explain(mode="formatted")
> 
> result.show()
> 
> # Compare memory usage
> print(f"\nDataFrame partitions: {df.rdd.getNumPartitions()}")
> print("Tungsten uses binary format for efficient memory usage")
> 
> # Tungsten Sort - more efficient than traditional sorting
> sorted_df = df.orderBy("value")
> print("\n=== Tungsten Sort ===")
> sorted_df.explain()
> 
> # Tungsten Aggregation - uses hash-based aggregation
> agg_result = df.groupBy("category", "id").agg(_sum("value"))
> print("\n=== Tungsten Hash Aggregation ===")
> agg_result.explain()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/whole_stage_codegen.py
> """
> Whole-Stage Code Generation Pattern
> - Combines multiple operators into a single Java function
> - Eliminates virtual function calls and improves CPU efficiency
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, sum as _sum, when
> 
> spark = SparkSession.builder \
>     .appName("WholeStageCodegen") \
>     .config("spark.sql.codegen.wholeStage", "true") \
>     .config("spark.sql.codegen.maxFields", "200") \
>     .getOrCreate()
> 
> # Create dataset
> df = spark.range(0, 10000000)
> 
> # Whole-stage codegen combines these operations into single generated code
> # Without codegen: filter -> map -> filter (multiple function calls)
> # With codegen: single compiled function doing all operations
> result = df.filter(col("id") % 2 == 0) \
>     .withColumn("doubled", col("id") * 2) \
>     .filter(col("doubled") < 1000000) \
>     .withColumn("category", 
>         when(col("doubled") < 100000, "small")
>         .when(col("doubled") < 500000, "medium")
>         .otherwise("large")
>     )
> 
> print("=== Whole-Stage Code Generation Plan ===")
> result.explain(mode="formatted")
> # Look for "*WholeStageCodegen" in the plan
> 
> # The generated code is visible with explain(extended=True)
> print("\n=== Generated Code ===")
> result.explain(extended=True)
> 
> result.show(10)
> 
> # Complex pipeline - all combined into one code block
> complex_result = df.select(
>     col("id"),
>     (col("id") * 2).alias("double"),
>     (col("id") * 3).alias("triple")
> ).filter(col("double") > 100) \
>     .filter(col("triple") < 1000000) \
>     .select(
>         col("id"),
>         (col("double") + col("triple")).alias("sum")
>     )
> 
> print("\n=== Complex Whole-Stage Codegen ===")
> complex_result.explain()
> 
> # Operations that break whole-stage codegen:
> # 1. UDFs (use built-in functions instead)
> # 2. Repartitioning
> # 3. Some complex nested operations
> 
> # Example of codegen breakpoint
> from pyspark.sql.functions import udf
> from pyspark.sql.types import IntegerType
> 
> @udf(returnType=IntegerType())
> def custom_func(x):
>     return x * 10
> 
> # This breaks whole-stage codegen
> broken_codegen = df.withColumn("custom", custom_func(col("id")))
> print("\n=== Broken Codegen (UDF) ===")
> broken_codegen.explain()
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/vectorized_execution.py
> """
> Vectorized Execution Pattern
> - Processes data in batches (columnar format)
> - Leverages SIMD instructions and Arrow format
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, pandas_udf
> from pyspark.sql.types import IntegerType
> import pandas as pd
> import numpy as np
> 
> spark = SparkSession.builder \
>     .appName("VectorizedExecution") \
>     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
>     .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
>     .getOrCreate()
> 
> # Create dataset
> df = spark.range(0, 1000000)
> 
> # Vectorized UDF using Pandas (much faster than row-by-row)
> @pandas_udf(IntegerType())
> def vectorized_multiply(series: pd.Series) -> pd.Series:
>     # Operates on entire column at once
>     return series * 10
> 
> result = df.withColumn("multiplied", vectorized_multiply(col("id")))
> 
> print("=== Vectorized Execution ===")
> result.show(10)
> 
> # Vectorized aggregation
> @pandas_udf(IntegerType())
> def vectorized_aggregate(series: pd.Series) -> int:
>     return int(np.sum(series))
> 
> # Use with groupBy
> grouped_df = df.withColumn("category", col("id") % 100)
> agg_result = grouped_df.groupBy("category") \
>     .agg(vectorized_aggregate(col("id")).alias("total"))
> 
> agg_result.show(10)
> 
> # Arrow-based conversion (vectorized)
> # Convert to Pandas DataFrame efficiently using Arrow
> pandas_df = df.limit(10000).toPandas()
> print(f"\nConverted to Pandas (Arrow): {len(pandas_df)} rows")
> 
> # Convert from Pandas efficiently
> spark_df = spark.createDataFrame(pandas_df)
> print(f"Converted from Pandas (Arrow): {spark_df.count()} rows")
> 
> # Vectorized window functions
> from pyspark.sql.window import Window
> from pyspark.sql.functions import row_number
> 
> window_spec = Window.partitionBy("category").orderBy("id")
> windowed = grouped_df.withColumn("row_num", row_number().over(window_spec))
> 
> print("\n=== Vectorized Window Function ===")
> windowed.show(10)
> 
> # Performance comparison example
> print("\n=== Performance Benefits ===")
> print("Vectorized operations benefit from:")
> print("1. Columnar processing (better cache utilization)")
> print("2. Arrow format (zero-copy between Spark and Pandas)")
> print("3. SIMD instructions (parallel processing)")
> print("4.

#### _User_

> continue exactly where you left off

#### _Assistant_

> ```python optimization_patterns/vectorized_execution.py
> # ... existing code ...
> 
> # Performance comparison example
> print("\n=== Performance Benefits ===")
> print("Vectorized operations benefit from:")
> print("1. Columnar processing (better cache utilization)")
> print("2. Arrow format (zero-copy between Spark and Pandas)")
> print("3. SIMD instructions (parallel processing)")
> print("4. Reduced Python overhead (batch processing)")
> 
> # Complex vectorized transformation
> @pandas_udf(IntegerType())
> def complex_vectorized_calc(id_series: pd.Series) -> pd.Series:
>     # NumPy vectorized operations are extremely fast
>     result = np.where(
>         id_series < 500000,
>         id_series * 2,
>         np.where(
>             id_series < 750000,
>             id_series * 3,
>             id_series * 4
>         )
>     )
>     return pd.Series(result)
> 
> complex_result = df.withColumn("calculated", complex_vectorized_calc(col("id")))
> complex_result.show(10)
> 
> # Vectorized filter (using Arrow)
> # More efficient than row-by-row filtering
> vectorized_filter = df.filter(col("id") > 500000).filter(col("id") < 600000)
> print(f"\nVectorized filter result: {vectorized_filter.count()} rows")
> 
> spark.stop()
> ```
> 
> ```python optimization_patterns/demo_runner.py
> """
> Demo Runner - Execute all optimization patterns
> Run this to see all patterns in action
> """
> from pyspark.sql import SparkSession
> import subprocess
> import os
> 
> def run_pattern(script_name):
>     """Execute a pattern script"""
>     print(f"\n{'='*60}")
>     print(f"Running: {script_name}")
>     print(f"{'='*60}\n")
>     
>     try:
>         result = subprocess.run(
>             ["python", script_name],
>             capture_output=True,
>             text=True,
>             timeout=60
>         )
>         print(result.stdout)
>         if result.stderr:
>             print("STDERR:", result.stderr)
>         return result.returncode == 0
>     except subprocess.TimeoutExpired:
>         print(f"TIMEOUT: {script_name} took too long")
>         return False
>     except Exception as e:
>         print(f"ERROR running {script_name}: {e}")
>         return False
> 
> def main():
>     patterns = [
>         "catalyst_optimizer.py",
>         "aqe_pattern.py",
>         "dynamic_partition_pruning.py",
>         "predicate_pushdown.py",
>         "projection_pushdown.py",
>         "column_pruning.py",
>         "constant_folding.py",
>         "cost_based_optimization.py",
>         "broadcast_variable.py",
>         "accumulator_pattern.py",
>         "caching_strategy.py",
>         "persistence_level.py",
>         "tungsten_optimization.py",
>         "whole_stage_codegen.py",
>         "vectorized_execution.py"
>     ]
>     
>     print("PySpark Optimization Patterns Demo")
>     print("=" * 60)
>     
>     results = {}
>     for pattern in patterns:
>         script_path = os.path.join("optimization_patterns", pattern)
>         if os.path.exists(script_path):
>             results[pattern] = run_pattern(script_path)
>         else:
>             print(f"Script not found: {script_path}")
>             results[pattern] = False
>     
>     # Summary
>     print("\n" + "=" * 60)
>     print("SUMMARY")
>     print("=" * 60)
>     for pattern, success in results.items():
>         status = "✓ PASSED" if success else "✗ FAILED"
>         print(f"{status}: {pattern}")
>     
>     passed = sum(1 for v in results.values() if v)
>     total = len(results)
>     print(f"\nTotal: {passed}/{total} patterns executed successfully")
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> ```python optimization_patterns/best_practices_guide.py
> """
> Best Practices Guide - When to use each optimization pattern
> """
> 
> OPTIMIZATION_GUIDE = """
> ╔══════════════════════════════════════════════════════════════════════════════╗
> ║                    PYSPARK OPTIMIZATION PATTERNS GUIDE                        ║
> ╚══════════════════════════════════════════════════════════════════════════════╝
> 
> 1. CATALYST OPTIMIZER PATTERN
>    When: Always enabled by default
>    Use: Understanding query optimization, debugging performance
>    Tips: Use explain() to verify optimizations are applied
> 
> 2. ADAPTIVE QUERY EXECUTION (AQE) PATTERN
>    When: Spark 3.0+, queries with joins and aggregations
>    Use: Dynamic optimization at runtime
>    Config: spark.sql.adaptive.enabled=true
>    Benefits: Auto-optimizes shuffle partitions, join strategies, skew handling
> 
> 3. DYNAMIC PARTITION PRUNING PATTERN
>    When: Star schema joins with partitioned fact tables
>    Use: Reduce partition scanning based on dimension filters
>    Config: spark.sql.optimizer.dynamicPartitionPruning.enabled=true
>    Impact: Can reduce I/O by 90%+ in partitioned datasets
> 
> 4. PREDICATE PUSHDOWN PATTERN
>    When: Reading from Parquet, ORC, JDBC, or other sources
>    Use: Filter data at the source
>    Tips: Place filters early in query, use partition columns
>    Impact: Reduces data transfer and memory usage
> 
> 5. PROJECTION PUSHDOWN PATTERN
>    When: Working with wide tables (many columns)
>    Use: Read only necessary columns from storage
>    Tips: Select columns early, avoid SELECT *
>    Impact: Reduces I/O by 50-90% for wide tables
> 
> 6. COLUMN PRUNING PATTERN
>    When: Complex transformations with intermediate columns
>    Use: Automatically removes unused columns
>    Tips: Let Catalyst handle it, but verify with explain()
>    Impact: Reduces memory footprint
> 
> 7. CONSTANT FOLDING PATTERN
>    When: Queries with constant expressions
>    Use: Automatic - Catalyst evaluates constants at compile time
>    Tips: Use lit() for constants, avoid UDFs for simple math
>    Impact: Reduces CPU overhead
> 
> 8. COST-BASED OPTIMIZATION PATTERN
>    When: Complex multi-table joins
>    Use: Collect statistics for better join ordering
>    Config: spark.sql.cbo.enabled=true
>    Commands: ANALYZE TABLE ... COMPUTE STATISTICS
>    Impact: Can improve join performance by 2-10x
> 
> 9. BROADCAST VARIABLE PATTERN
>    When: Joining large table with small table (<10MB)
>    Use: Avoid shuffle by broadcasting small table
>    Tips: Use broadcast() hint or set autoBroadcastJoinThreshold
>    Impact: Eliminates shuffle, 5-20x faster joins
> 
> 10. ACCUMULATOR PATTERN
>     When: Monitoring, debugging, custom metrics
>     Use: Track counters across distributed operations
>     Tips: Don't use for business logic, only monitoring
>     Impact: Zero overhead, great for observability
> 
> 11. CACHING STRATEGY PATTERN
>     When: DataFrame is reused 2+ times
>     Use: Cache intermediate results in iterative algorithms
>     Tips: Cache after expensive operations, unpersist when done
>     Impact: 10-100x speedup for reused data
> 
> 12. PERSISTENCE LEVEL PATTERN
>     When: Choosing how to cache data
>     Use: 
>       - MEMORY_ONLY: Fast, fits in memory
>       - MEMORY_AND_DISK: Safe default
>       - MEMORY_AND_DISK_SER: Large datasets
>       - DISK_ONLY: Very large datasets
>     Impact: Memory vs. speed tradeoff
> 
> 13. TUNGSTEN OPTIMIZATION PATTERN
>     When: Always enabled in Spark 2.0+
>     Use: Automatic binary format and memory management
>     Benefits: 2-5x memory efficiency, faster execution
>     Config: Always on, no action needed
> 
> 14. WHOLE-STAGE CODE GENERATION PATTERN
>     When: Chains of operations (filter, map, aggregate)
>     Use: Automatic - combines operators into single compiled function
>     Config: spark.sql.codegen.wholeStage=true
>     Tips: Avoid UDFs which break codegen
>     Impact: 2-10x speedup for compatible operations
> 
> 15. VECTORIZED EXECUTION PATTERN
>     When: Using Pandas UDFs, large datasets
>     Use: Process batches of rows with Arrow format
>     Config: spark.sql.execution.arrow.pyspark.enabled=true
>     Tips: Use pandas_udf instead of regular UDFs
>     Impact: 3-100x faster than row-by-row UDFs
> 
> ╔══════════════════════════════════════════════════════════════════════════════╗
> ║                          OPTIMIZATION CHECKLIST                               ║
> ╚══════════════════════════════════════════════════════════════════════════════╝
> 
> □ Enable AQE for Spark 3.0+
> □ Collect statistics with ANALYZE TABLE for CBO
> □ Partition large tables by commonly filtered columns
> □ Use Parquet or ORC for columnar storage
> □ Place filters as early as possible
> □ Use broadcast() for small table joins
> □ Cache DataFrames used 2+ times
> □ Replace UDFs with built-in functions when possible
> □ Use pandas_udf for Python transformations
> □ Monitor with explain() and Spark UI
> □ Unpersist cached DataFrames when done
> □ Use appropriate persistence levels
> □ Avoid collect() on large datasets
> □ Repartition intelligently (not too many, not too few)
> □ Use coalesce() instead of repartition() when reducing partitions
> 
> ╔══════════════════════════════════════════════════════════════════════════════╗
> ║                          COMMON ANTI-PATTERNS                                 ║
> ╚══════════════════════════════════════════════════════════════════════════════╝
> 
> ✗ Using collect() on large DataFrames
> ✗ Using Python UDFs instead of built-in functions
> ✗ Not caching reused DataFrames
> ✗ Caching everything (wastes memory)
> ✗ Using SELECT * with wide tables
> ✗ Not partitioning large tables
> ✗ Too many small files (use coalesce/repartition)
> ✗ Shuffle-heavy operations without optimization
> ✗ Not enabling AQE in Spark 3.0+
> ✗ Ignoring skewed data in joins
> 
> ╔══════════════════════════════════════════════════════════════════════════════╗
> ║                       PERFORMANCE TROUBLESHOOTING                             ║
> ╚══════════════════════════════════════════════════════════════════════════════╝
> 
> Issue: Slow joins
> → Check join keys distribution (skew)
> → Use broadcast for small tables
> → Enable AQE skew join optimization
> → Collect statistics for CBO
> 
> Issue: High memory usage
> → Use appropriate persistence levels
> → Unpersist unused cached data
> → Reduce partition size
> → Use column pruning/projection pushdown
> 
> Issue: Many small tasks
> → Coalesce partitions
> → Increase partition size
> → Use AQE to auto-coalesce
> 
> Issue: Data skew
> → Enable AQE skew join handling
> → Salt join keys
> → Use broadcast join if one side is small
> 
> Issue: Slow Python UDFs
> → Replace with built-in functions
> → Use pandas_udf for vectorized processing
> → Use Arrow for Pandas conversion
> """
> 
> def print_guide():
>     print(OPTIMIZATION_GUIDE)
> 
> if __name__ == "__main__":
>     print_guide()
> ```
> 
> ```python optimization_patterns/performance_comparison.py
> """
> Performance Comparison - Benchmark different optimization patterns
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, broadcast, pandas_udf, udf
> from pyspark.sql.types import IntegerType
> import pandas as pd
> import time
> 
> spark = SparkSession.builder \
>     .appName("PerformanceComparison") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
>     .getOrCreate()
> 
> def benchmark(name, func):
>     """Utility to benchmark operations"""
>     start = time.time()
>     result = func()
>     end = time.time()
>     duration = end - start
>     print(f"{name}: {duration:.3f}s")
>     return result, duration
> 
> print("=" * 70)
> print("PERFORMANCE COMPARISON - OPTIMIZATION PATTERNS")
> print("=" * 70)
> 
> # Create test datasets
> large_df = spark.range(0, 1000000)
> small_df = spark.range(0, 100)
> 
> # 1. JOIN STRATEGIES
> print("\n1. JOIN STRATEGIES")
> print("-" * 70)
> 
> def shuffle_join():
>     return large_df.join(small_df, large_df.id % 100 == small_df.id).count()
> 
> def broadcast_join():
>     return large_df.join(broadcast(small_df), large_df.id % 100 == small_df.id).count()
> 
> _, shuffle_time = benchmark("  Shuffle Join", shuffle_join)
> _, broadcast_time = benchmark("  Broadcast Join", broadcast_join)
> print(f"  Speedup: {shuffle_time / broadcast_time:.2f}x")
> 
> # 2. CACHING STRATEGIES
> print("\n2. CACHING STRATEGIES")
> print("-" * 70)
> 
> filtered_df = large_df.filter(col("id") < 500000)
> 
> def without_cache():
>     count1 = filtered_df.filter(col("id") > 100000).count()
>     count2 = filtered_df.filter(col("id") > 200000).count()
>     return count1, count2
> 
> def with_cache():
>     cached = filtered_df.cache()
>     count1 = cached.filter(col("id") > 100000).count()
>     count2 = cached.filter(col("id") > 200000).count()
>     cached.unpersist()
>     return count1, count2
> 
> _, no_cache_time = benchmark("  Without Cache", without_cache)
> _, cache_time = benchmark("  With Cache", with_cache)
> print(f"  Speedup: {no_cache_time / cache_time:.2f}x")
> 
> # 3. UDF vs PANDAS_UDF
> print("\n3. UDF vs PANDAS_UDF vs BUILT-IN")
> print("-" * 70)
> 
> test_df = spark.range(0, 100000)
> 
> # Regular UDF
> @udf(returnType=IntegerType())
> def regular_udf(x):
>     return x * 2
> 
> # Pandas UDF
> @pandas_udf(IntegerType())
> def pandas_udf_func(series: pd.Series) -> pd.Series:
>     return series * 2
> 
> def use_regular_udf():
>     return test_df.withColumn("result", regular_udf(col("id"))).count()
> 
> def use_pandas_udf():
>     return test_df.withColumn("result", pandas_udf_func(col("id"))).count()
> 
> def use_builtin():
>     return test_df.withColumn("result", col("id") * 2).count()
> 
> _, regular_time = benchmark("  Regular UDF", use_regular_udf)
> _, pandas_time = benchmark("  Pandas UDF", use_pandas_udf)
> _, builtin_time = benchmark("  Built-in Function", use_builtin)
> 
> print(f"  Pandas UDF vs Regular UDF: {regular_time / pandas_time:.2f}x faster")
> print(f"  Built-in vs Regular UDF: {regular_time / builtin_time:.2f}x faster")
> 
> # 4. COLUMN PRUNING
> print("\n4. COLUMN PRUNING")
> print("-" * 70)
> 
> wide_df = spark.range(0, 100000).selectExpr(
>     "id",
>     "id * 2 as col1",
>     "id * 3 as col2",
>     "id * 4 as col3",
>     "id * 5 as col4",
>     "id * 6 as col5"
> )
> 
> def select_all():
>     return wide_df.select("*").filter(col("id") < 50000).count()
> 
> def select_needed():
>     return wide_df.select("id", "col1").filter(col("id") < 50000).count()
> 
> _, select_all_time = benchmark("  SELECT *", select_all)
> _, select_needed_time = benchmark("  SELECT specific", select_needed)
> print(f"  Speedup: {select_all_time / select_needed_time:.2f}x")
> 
> # 5. PARTITIONING
> print("\n5. PARTITIONING STRATEGIES")
> print("-" * 70)
> 
> df_many_partitions = spark.range(0, 100000).repartition(1000)
> df_optimal_partitions = spark.range(0, 100000).repartition(10)
> 
> def many_partitions():
>     return df_many_partitions.groupBy("id").count().count()
> 
> def optimal_partitions():
>     return df_optimal_partitions.groupBy("id").count().count()
> 
> _, many_part_time = benchmark("  1000 partitions", many_partitions)
> _, optimal_part_time = benchmark("  10 partitions", optimal_partitions)
> print(f"  Speedup: {many_part_time / optimal_part_time:.2f}x")
> 
> print("\n" + "=" * 70)
> print("SUMMARY")
> print("=" * 70)
> print("Key Takeaways:")
> print("  • Broadcast joins are significantly faster for small tables")
> print("  • Caching reused DataFrames provides major speedups")
> print("  • Pandas UDFs are much faster than regular UDFs")
> print("  • Built-in functions are fastest - avoid UDFs when possible")
> print("  • Column pruning reduces I/O and memory")
> print("  • Optimal partitioning is critical for performance")
> print("=" * 70)
> 
> spark.stop()
> ```
> 
> These patterns provide a comprehensive guide to PySpark optimization! Each script demonstrates:
> 
> - **What**: The optimization pattern
> - **When**: Best use cases
> - **How**: Implementation examples
> - **Why**: Performance benefits
> 
> You can run individual patterns or use the `demo_runner.py` to execute them all. The `best_practices_guide.py` provides a reference guide, and `performance_comparison.py` shows measurable performance differences.