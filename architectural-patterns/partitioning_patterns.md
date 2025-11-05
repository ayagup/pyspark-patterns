### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 7:01:21 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Partitioning Patterns
> - Hash Partitioning Pattern
> - Range Partitioning Pattern
> - Custom Partitioning Pattern
> - Dynamic Partitioning Pattern
> - Static Partitioning Pattern
> - Time-based Partitioning Pattern
> - Date Partitioning Pattern (Year/Month/Day)
> - Bucketing Pattern
> - Co-partitioning Pattern
> - Partition Pruning Pattern
> - Partition Discovery Pattern
> - Multi-level Partitioning Pattern
> ```

#### _Assistant_

> # PySpark Partitioning Patterns
> 
> I'll create comprehensive examples for each partitioning pattern:
> 
> ```python partitioning_patterns.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import *
> from pyspark.sql.types import *
> from pyspark.sql.window import Window
> from datetime import datetime, timedelta
> import random
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Partitioning Patterns") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .config("spark.sql.shuffle.partitions", "10") \
>     .getOrCreate()
> 
> # Sample data generation
> def generate_sample_data():
>     data = []
>     categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Sports']
>     regions = ['North', 'South', 'East', 'West']
>     
>     base_date = datetime(2024, 1, 1)
>     for i in range(1000):
>         sale_date = base_date + timedelta(days=random.randint(0, 364))
>         data.append({
>             'order_id': i,
>             'customer_id': random.randint(1, 100),
>             'product': f'Product_{random.randint(1, 50)}',
>             'category': random.choice(categories),
>             'region': random.choice(regions),
>             'amount': round(random.uniform(10, 1000), 2),
>             'quantity': random.randint(1, 10),
>             'sale_date': sale_date,
>             'year': sale_date.year,
>             'month': sale_date.month,
>             'day': sale_date.day
>         })
>     
>     schema = StructType([
>         StructField("order_id", IntegerType(), False),
>         StructField("customer_id", IntegerType(), False),
>         StructField("product", StringType(), False),
>         StructField("category", StringType(), False),
>         StructField("region", StringType(), False),
>         StructField("amount", DoubleType(), False),
>         StructField("quantity", IntegerType(), False),
>         StructField("sale_date", TimestampType(), False),
>         StructField("year", IntegerType(), False),
>         StructField("month", IntegerType(), False),
>         StructField("day", IntegerType(), False)
>     ])
>     
>     return spark.createDataFrame(data, schema)
> 
> df = generate_sample_data()
> print("Sample Data Generated:")
> df.show(5)
> 
> # ============================================================================
> # 1. Hash Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("1. HASH PARTITIONING PATTERN")
> print("="*80)
> 
> # Repartition data using hash partitioning on customer_id
> hash_partitioned_df = df.repartition(8, "customer_id")
> 
> print(f"Number of partitions: {hash_partitioned_df.rdd.getNumPartitions()}")
> print("Partition distribution:")
> hash_partitioned_df.withColumn("partition_id", spark_partition_id()).groupBy("partition_id").count().show()
> 
> # Write with hash partitioning
> hash_partitioned_df.write \
>     .mode("overwrite") \
>     .parquet("output/hash_partitioned")
> 
> # ============================================================================
> # 2. Range Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("2. RANGE PARTITIONING PATTERN")
> print("="*80)
> 
> # Range partition by amount (useful for sorted operations)
> range_partitioned_df = df.repartitionByRange(8, "amount")
> 
> print(f"Number of partitions: {range_partitioned_df.rdd.getNumPartitions()}")
> print("Partition distribution by amount ranges:")
> range_partitioned_df.withColumn("partition_id", spark_partition_id()) \
>     .groupBy("partition_id") \
>     .agg(
>         min("amount").alias("min_amount"),
>         max("amount").alias("max_amount"),
>         count("*").alias("count")
>     ).orderBy("partition_id").show()
> 
> # Write with range partitioning
> range_partitioned_df.write \
>     .mode("overwrite") \
>     .parquet("output/range_partitioned")
> 
> # ============================================================================
> # 3. Custom Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("3. CUSTOM PARTITIONING PATTERN")
> print("="*80)
> 
> # Custom partitioner using UDF
> from pyspark import SparkContext
> 
> # Add custom partition column based on business logic
> custom_partitioned_df = df.withColumn(
>     "custom_partition",
>     when(col("amount") < 100, "low")
>     .when((col("amount") >= 100) & (col("amount") < 500), "medium")
>     .otherwise("high")
> )
> 
> # Repartition based on custom column
> custom_partitioned_df = custom_partitioned_df.repartition(3, "custom_partition")
> 
> print("Custom partition distribution:")
> custom_partitioned_df.groupBy("custom_partition").count().show()
> 
> custom_partitioned_df.write \
>     .mode("overwrite") \
>     .parquet("output/custom_partitioned")
> 
> # ============================================================================
> # 4. Dynamic Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("4. DYNAMIC PARTITIONING PATTERN")
> print("="*80)
> 
> # Dynamic partitioning - partitions are created based on data values
> # Commonly used when writing to Hive or file systems
> 
> spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
> 
> df.write \
>     .mode("overwrite") \
>     .partitionBy("category", "region") \
>     .parquet("output/dynamic_partitioned")
> 
> print("Dynamic partitions created based on category and region values")
> print("Partition structure will have directories like: category=Electronics/region=North/")
> 
> # Read back and show partition info
> dynamic_read = spark.read.parquet("output/dynamic_partitioned")
> print("\nPartition columns:")
> print(dynamic_read.select("category", "region").distinct().orderBy("category", "region").show())
> 
> # ============================================================================
> # 5. Static Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("5. STATIC PARTITIONING PATTERN")
> print("="*80)
> 
> # Static partitioning - explicitly specify partition values
> # Write specific partition
> electronics_north = df.filter((col("category") == "Electronics") & (col("region") == "North"))
> 
> electronics_north.write \
>     .mode("overwrite") \
>     .parquet("output/static_partitioned/category=Electronics/region=North")
> 
> print("Static partition created for category=Electronics and region=North")
> print(f"Records written: {electronics_north.count()}")
> 
> # ============================================================================
> # 6. Time-based Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("6. TIME-BASED PARTITIONING PATTERN")
> print("="*80)
> 
> # Partition by hour, useful for streaming or time-series data
> time_based_df = df.withColumn("hour", hour(col("sale_date"))) \
>                   .withColumn("date", to_date(col("sale_date")))
> 
> time_based_df.write \
>     .mode("overwrite") \
>     .partitionBy("date", "hour") \
>     .parquet("output/time_based_partitioned")
> 
> print("Time-based partitioning by date and hour:")
> time_based_df.groupBy("date", "hour").count().orderBy("date", "hour").show(10)
> 
> # ============================================================================
> # 7. Date Partitioning Pattern (Year/Month/Day)
> # ============================================================================
> print("\n" + "="*80)
> print("7. DATE PARTITIONING PATTERN (YEAR/MONTH/DAY)")
> print("="*80)
> 
> # Multi-level date partitioning
> date_partitioned_df = df.select(
>     "*",
>     year(col("sale_date")).alias("year"),
>     month(col("sale_date")).alias("month"),
>     dayofmonth(col("sale_date")).alias("day")
> )
> 
> date_partitioned_df.write \
>     .mode("overwrite") \
>     .partitionBy("year", "month", "day") \
>     .parquet("output/date_partitioned")
> 
> print("Date partitioning hierarchy created (year/month/day):")
> date_partitioned_df.groupBy("year", "month", "day").count().orderBy("year", "month", "day").show(10)
> 
> # ============================================================================
> # 8. Bucketing Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("8. BUCKETING PATTERN")
> print("="*80)
> 
> # Bucketing - similar to hash partitioning but for tables
> # Creates fixed number of buckets based on hash of column values
> 
> df.write \
>     .mode("overwrite") \
>     .bucketBy(10, "customer_id") \
>     .sortBy("order_id") \
>     .saveAsTable("bucketed_sales")
> 
> print("Bucketed table created with 10 buckets on customer_id")
> print("Bucketing distribution:")
> bucketed_df = spark.table("bucketed_sales")
> bucketed_df.groupBy("customer_id").count().show(10)
> 
> # ============================================================================
> # 9. Co-partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("9. CO-PARTITIONING PATTERN")
> print("="*80)
> 
> # Create two DataFrames that will be joined
> customers_data = [(i, f"Customer_{i}", random.choice(['North', 'South', 'East', 'West'])) 
>                   for i in range(1, 101)]
> customers_df = spark.createDataFrame(customers_data, ["customer_id", "customer_name", "region"])
> 
> # Co-partition both DataFrames on the join key
> orders_copart = df.repartition(8, "customer_id")
> customers_copart = customers_df.repartition(8, "customer_id")
> 
> print("Both DataFrames partitioned on customer_id for efficient join:")
> print(f"Orders partitions: {orders_copart.rdd.getNumPartitions()}")
> print(f"Customers partitions: {customers_copart.rdd.getNumPartitions()}")
> 
> # This join will be more efficient (no shuffle needed)
> joined_df = orders_copart.join(customers_copart, "customer_id")
> print(f"Joined records: {joined_df.count()}")
> 
> # ============================================================================
> # 10. Partition Pruning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("10. PARTITION PRUNING PATTERN")
> print("="*80)
> 
> # Write partitioned data
> df.write \
>     .mode("overwrite") \
>     .partitionBy("category", "region") \
>     .parquet("output/pruning_example")
> 
> # Read with partition pruning - only reads relevant partitions
> pruned_df = spark.read.parquet("output/pruning_example") \
>     .filter((col("category") == "Electronics") & (col("region") == "North"))
> 
> print("Partition pruning - only Electronics/North partition is read:")
> print(f"Filtered records: {pruned_df.count()}")
> print("\nExplain plan shows partition filters:")
> pruned_df.explain()
> 
> # ============================================================================
> # 11. Partition Discovery Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("11. PARTITION DISCOVERY PATTERN")
> print("="*80)
> 
> # Write partitioned data without schema
> df.select("order_id", "product", "amount", "category", "region").write \
>     .mode("overwrite") \
>     .partitionBy("category", "region") \
>     .parquet("output/discovery_example")
> 
> # Spark automatically discovers partitions
> discovered_df = spark.read.parquet("output/discovery_example")
> 
> print("Partition columns automatically discovered:")
> print(discovered_df.printSchema())
> print("\nDiscovered partition values:")
> discovered_df.select("category", "region").distinct().orderBy("category", "region").show()
> 
> # Manual partition discovery
> spark.sql("MSCK REPAIR TABLE discovered_sales")  # For Hive tables
> 
> # ============================================================================
> # 12. Multi-level Partitioning Pattern
> # ============================================================================
> print("\n" + "="*80)
> print("12. MULTI-LEVEL PARTITIONING PATTERN")
> print("="*80)
> 
> # Complex multi-level partitioning combining different strategies
> multi_level_df = df.withColumn(
>     "amount_tier",
>     when(col("amount") < 100, "tier1")
>     .when((col("amount") >= 100) & (col("amount") < 500), "tier2")
>     .otherwise("tier3")
> )
> 
> multi_level_df.write \
>     .mode("overwrite") \
>     .partitionBy("year", "month", "category", "region", "amount_tier") \
>     .parquet("output/multi_level_partitioned")
> 
> print("Multi-level partitioning hierarchy:")
> print("year -> month -> category -> region -> amount_tier")
> print("\nPartition distribution:")
> multi_level_df.groupBy("year", "month", "category", "region", "amount_tier") \
>     .count() \
>     .orderBy("year", "month", "category", "region", "amount_tier") \
>     .show(20)
> 
> # Read specific partition path
> specific_partition = spark.read.parquet(
>     "output/multi_level_partitioned/year=2024/month=1/category=Electronics"
> )
> print(f"\nRecords in specific partition path: {specific_partition.count()}")
> 
> # ============================================================================
> # BONUS: Partition Management and Optimization
> # ============================================================================
> print("\n" + "="*80)
> print("PARTITION MANAGEMENT AND OPTIMIZATION")
> print("="*80)
> 
> # Coalesce - reduce partitions without full shuffle
> coalesced_df = df.coalesce(4)
> print(f"Coalesced partitions: {coalesced_df.rdd.getNumPartitions()}")
> 
> # Check partition sizes
> def analyze_partitions(df, name):
>     print(f"\n{name} - Partition Analysis:")
>     partition_counts = df.withColumn("partition_id", spark_partition_id()) \
>         .groupBy("partition_id").count() \
>         .orderBy("partition_id")
>     partition_counts.show()
>     
>     stats = partition_counts.select(
>         min("count").alias("min_rows"),
>         max("count").alias("max_rows"),
>         avg("count").alias("avg_rows")
>     )
>     stats.show()
> 
> analyze_partitions(df, "Original DataFrame")
> analyze_partitions(hash_partitioned_df, "Hash Partitioned")
> analyze_partitions(range_partitioned_df, "Range Partitioned")
> 
> # Repartition optimization for write operations
> optimized_write_df = df.repartition(col("category"), col("region"))
> print("\nOptimized for writing partitioned data:")
> print(f"Partitions: {optimized_write_df.rdd.getNumPartitions()}")
> 
> # Clean up
> spark.catalog.dropTempView("bucketed_sales") if spark.catalog._jcatalog.tableExists("bucketed_sales") else None
> 
> print("\n" + "="*80)
> print("ALL PARTITIONING PATTERNS COMPLETED!")
> print("="*80)
> 
> spark.stop()
> ```
> 
> ```python partitioning_best_practices.py
> # Best Practices and Common Use Cases for Partitioning Patterns
> 
> """
> PARTITIONING PATTERN SELECTION GUIDE:
> 
> 1. HASH PARTITIONING
>    - Use when: Need even data distribution for parallel processing
>    - Example: Distributing user events by user_id for aggregations
>    - Benefits: Balanced partitions, good for joins on partition key
> 
> 2. RANGE PARTITIONING
>    - Use when: Data has natural ordering, need sorted output
>    - Example: Time-series data, numerical ranges
>    - Benefits: Efficient range queries, sorted partitions
> 
> 3. CUSTOM PARTITIONING
>    - Use when: Business logic requires specific grouping
>    - Example: Segmenting customers by value tier
>    - Benefits: Domain-specific optimization
> 
> 4. DYNAMIC PARTITIONING
>    - Use when: Writing to storage with unknown partition values
>    - Example: ETL jobs writing daily data
>    - Benefits: Automatic partition creation, flexible
> 
> 5. STATIC PARTITIONING
>    - Use when: Partition values are known and fixed
>    - Example: Processing specific date ranges
>    - Benefits: Predictable, efficient for targeted writes
> 
> 6. TIME-BASED PARTITIONING
>    - Use when: Processing streaming or time-series data
>    - Example: Log aggregation, sensor data
>    - Benefits: Efficient time-range queries, data retention
> 
> 7. DATE PARTITIONING
>    - Use when: Historical data analysis, data lake organization
>    - Example: Sales analytics, event tracking
>    - Benefits: Standard pattern, easy querying
> 
> 8. BUCKETING
>    - Use when: Repeated joins on same key, exact partition control
>    - Example: Fact-dimension joins in data warehouse
>    - Benefits: Join optimization, consistent hashing
> 
> 9. CO-PARTITIONING
>    - Use when: Multiple DataFrames joined frequently
>    - Example: Orders-Customers join in analytics pipeline
>    - Benefits: Eliminates shuffle in joins
> 
> 10. PARTITION PRUNING
>     - Use when: Querying large partitioned datasets
>     - Example: Querying specific dates from historical data
>     - Benefits: Reduced I/O, faster queries
> 
> 11. PARTITION DISCOVERY
>     - Use when: Reading externally partitioned data
>     - Example: Reading data from data lake
>     - Benefits: Automatic schema detection
> 
> 12. MULTI-LEVEL PARTITIONING
>     - Use when: Complex querying patterns, large datasets
>     - Example: Global sales data (region/country/state/city)
>     - Benefits: Flexible querying, optimized for various access patterns
> 
> OPTIMIZATION TIPS:
> - Aim for 128MB - 1GB partition size
> - Avoid too many small files (< 10MB)
> - Use coalesce for reducing partitions (no shuffle)
> - Use repartition for increasing partitions (with shuffle)
> - Choose partition columns with good cardinality
> - Avoid over-partitioning (too many directories)
> - Consider query patterns when designing partitions
> - Use partition pruning filters in queries
> - Monitor skew in partition distributions
> """
> ```
> 
> These examples cover all 12 partitioning patterns with practical implementations. Each pattern includes:
> - Clear use case explanation
> - Working code example
> - Output demonstration
> - Performance considerations
> 
> You can run the main script with:
> ```bash
> spark-submit partitioning_patterns.py
> ```
> 
> The patterns progress from basic (hash/range) to advanced (multi-level), helping you understand when and how to use each technique in real-world PySpark applications.