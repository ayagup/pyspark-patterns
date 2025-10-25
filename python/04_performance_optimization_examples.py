"""
PySpark Performance Optimization Patterns - Example Implementations
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, rand, floor, concat, lit
from pyspark import StorageLevel

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("PerformancePatterns") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()


# ============================================
# 4.1 Broadcast Join Pattern
# ============================================
def broadcast_join_example():
    """Demonstrate broadcast join for small tables"""
    print("\n=== Broadcast Join Pattern ===")
    
    # Large dataset
    large_df = spark.range(1, 100000).selectExpr(
        "id as order_id",
        "cast(rand() * 1000 as int) as customer_id",
        "cast(rand() * 1000 as int) as amount"
    )
    
    # Small lookup dataset
    small_df = spark.range(1, 101).selectExpr(
        "id as customer_id",
        "concat('Customer_', id) as customer_name"
    )
    
    print(f"Large DF count: {large_df.count()}")
    print(f"Small DF count: {small_df.count()}")
    
    # Regular join (may shuffle both tables)
    print("\nRegular Join:")
    result_regular = large_df.join(small_df, "customer_id")
    print(f"Result count: {result_regular.count()}")
    
    # Broadcast join (small table broadcast to all nodes)
    print("\nBroadcast Join:")
    result_broadcast = large_df.join(broadcast(small_df), "customer_id")
    print(f"Result count: {result_broadcast.count()}")
    result_broadcast.show(5)


# ============================================
# 4.2 Partitioning Pattern
# ============================================
def partitioning_example():
    """Demonstrate different partitioning strategies"""
    print("\n=== Partitioning Pattern ===")
    
    # Create sample data
    df = spark.range(1, 10001).selectExpr(
        "id",
        "cast(rand() * 10 as int) as category",
        "cast(rand() * 1000 as int) as value"
    )
    
    print(f"Original partitions: {df.rdd.getNumPartitions()}")
    
    # Repartition for parallelism
    df_repartitioned = df.repartition(8)
    print(f"After repartition(8): {df_repartitioned.rdd.getNumPartitions()}")
    
    # Repartition by column for better operations
    df_by_column = df.repartition("category")
    print(f"After repartition by 'category': {df_by_column.rdd.getNumPartitions()}")
    
    # Coalesce to reduce partitions (no shuffle)
    df_coalesced = df_repartitioned.coalesce(4)
    print(f"After coalesce(4): {df_coalesced.rdd.getNumPartitions()}")
    
    # Show distribution
    print("\nCategory distribution:")
    df.groupBy("category").count().orderBy("category").show()


# ============================================
# 4.3 Caching Pattern
# ============================================
def caching_example():
    """Demonstrate caching for frequently accessed data"""
    print("\n=== Caching Pattern ===")
    
    # Create sample data
    df = spark.range(1, 100001).selectExpr(
        "id",
        "cast(rand() * 100 as int) as value"
    )
    
    # Without caching - each operation reads from source
    print("Without caching:")
    import time
    
    start = time.time()
    count1 = df.filter(col("value") > 50).count()
    count2 = df.filter(col("value") <= 50).count()
    sum_val = df.agg({"value": "sum"}).collect()[0][0]
    time_without_cache = time.time() - start
    
    print(f"Count > 50: {count1}")
    print(f"Count <= 50: {count2}")
    print(f"Sum: {sum_val}")
    print(f"Time: {time_without_cache:.3f}s")
    
    # With caching
    print("\nWith caching:")
    df_cached = df.cache()
    
    start = time.time()
    count1 = df_cached.filter(col("value") > 50).count()
    count2 = df_cached.filter(col("value") <= 50).count()
    sum_val = df_cached.agg({"value": "sum"}).collect()[0][0]
    time_with_cache = time.time() - start
    
    print(f"Count > 50: {count1}")
    print(f"Count <= 50: {count2}")
    print(f"Sum: {sum_val}")
    print(f"Time: {time_with_cache:.3f}s")
    
    # Unpersist when done
    df_cached.unpersist()
    
    print(f"\nSpeedup: {time_without_cache / time_with_cache:.2f}x")


# ============================================
# 4.4 Salting Pattern for Data Skew
# ============================================
def salting_example():
    """Demonstrate salting to handle data skew"""
    print("\n=== Salting Pattern ===")
    
    # Create skewed dataset (key 1 appears much more frequently)
    skewed_data = [(1, f"value_{i}") for i in range(1000)] + \
                  [(i, f"value_{i}") for i in range(2, 101)]
    
    large_df = spark.createDataFrame(skewed_data, ["key", "value"])
    
    # Small dimension table
    small_df = spark.createDataFrame(
        [(i, f"dimension_{i}") for i in range(1, 101)],
        ["key", "dimension"]
    )
    
    print("Key distribution in large dataset:")
    large_df.groupBy("key").count().orderBy(col("count").desc()).show(5)
    
    # Add salt to skewed keys
    num_salts = 10
    df_salted = large_df.withColumn("salt", floor(rand() * num_salts)) \
        .withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))
    
    # Replicate small table with salt
    small_salted = small_df.crossJoin(
        spark.range(num_salts).withColumnRenamed("id", "salt")
    ).withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))
    
    # Join on salted key
    result = df_salted.join(small_salted, "salted_key") \
        .drop("salt", "salted_key")
    
    print(f"\nJoin result count: {result.count()}")
    result.show(5)


# ============================================
# 4.5 Predicate Pushdown Pattern
# ============================================
def predicate_pushdown_example():
    """Demonstrate predicate pushdown optimization"""
    print("\n=== Predicate Pushdown Pattern ===")
    
    # Create partitioned data
    df = spark.range(1, 10001).selectExpr(
        "id",
        "cast(id / 1000 as int) as year",
        "cast((id % 1000) / 100 as int) as month",
        "cast(rand() * 1000 as int) as value"
    )
    
    # Write partitioned data
    df.write.mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("/tmp/partitioned_perf")
    
    # Read with partition pruning (predicate pushdown)
    print("Reading with partition filter:")
    df_filtered = spark.read.parquet("/tmp/partitioned_perf") \
        .filter((col("year") == 0) & (col("month") == 5))
    
    print(f"Filtered count: {df_filtered.count()}")
    df_filtered.show(5)
    
    # Column pruning - select only needed columns
    print("\nWith column pruning:")
    df_columns = spark.read.parquet("/tmp/partitioned_perf") \
        .select("id", "value") \
        .filter(col("value") > 500)
    
    print(f"Selected count: {df_columns.count()}")
    df_columns.show(5)


# ============================================
# 4.6 Adaptive Query Execution (AQE)
# ============================================
def aqe_example():
    """Demonstrate AQE configuration"""
    print("\n=== Adaptive Query Execution Pattern ===")
    
    # Enable AQE
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    
    print("AQE Configuration:")
    print(f"AQE Enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")
    print(f"Coalesce Partitions: {spark.conf.get('spark.sql.adaptive.coalescePartitions.enabled')}")
    print(f"Skew Join: {spark.conf.get('spark.sql.adaptive.skewJoin.enabled')}")
    
    # Create datasets
    df1 = spark.range(1, 100001).selectExpr("id", "cast(rand() * 1000 as int) as key")
    df2 = spark.range(1, 1001).selectExpr("id as key", "concat('value_', id) as value")
    
    # Join with AQE
    result = df1.join(df2, "key")
    
    print(f"\nJoin result count: {result.count()}")
    print("\nExecution Plan (with AQE):")
    result.explain("formatted")


# ============================================
# 4.7 Bucketing Pattern
# ============================================
def bucketing_example():
    """Demonstrate bucketing for optimized joins"""
    print("\n=== Bucketing Pattern ===")
    
    # Create sample data
    df1 = spark.range(1, 10001).selectExpr(
        "id",
        "cast(rand() * 100 as int) as customer_id",
        "cast(rand() * 1000 as int) as amount"
    )
    
    df2 = spark.range(1, 101).selectExpr(
        "id as customer_id",
        "concat('Customer_', id) as name"
    )
    
    # Write with bucketing
    print("Writing bucketed tables...")
    df1.write.mode("overwrite") \
        .bucketBy(10, "customer_id") \
        .sortBy("customer_id") \
        .saveAsTable("orders_bucketed")
    
    df2.write.mode("overwrite") \
        .bucketBy(10, "customer_id") \
        .sortBy("customer_id") \
        .saveAsTable("customers_bucketed")
    
    # Read bucketed tables
    orders_bucketed = spark.table("orders_bucketed")
    customers_bucketed = spark.table("customers_bucketed")
    
    # Join bucketed tables (no shuffle needed!)
    result = orders_bucketed.join(customers_bucketed, "customer_id")
    
    print(f"Join result count: {result.count()}")
    result.show(5)
    
    print("\nExecution Plan:")
    result.explain()


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("PySpark Performance Optimization Patterns Examples")
    print("=" * 60)
    
    broadcast_join_example()
    partitioning_example()
    caching_example()
    salting_example()
    predicate_pushdown_example()
    aqe_example()
    bucketing_example()
    
    spark.stop()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
