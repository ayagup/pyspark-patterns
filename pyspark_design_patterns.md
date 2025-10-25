# Comprehensive PySpark Design Patterns

## Table of Contents
1. [Data Ingestion Patterns](#data-ingestion-patterns)
2. [Data Transformation Patterns](#data-transformation-patterns)
3. [Data Quality Patterns](#data-quality-patterns)
4. [Performance Optimization Patterns](#performance-optimization-patterns)
5. [Data Storage Patterns](#data-storage-patterns)
6. [Error Handling Patterns](#error-handling-patterns)
7. [Testing Patterns](#testing-patterns)
8. [Streaming Patterns](#streaming-patterns)
9. [Machine Learning Patterns](#machine-learning-patterns)
10. [Architecture Patterns](#architecture-patterns)

---

## 1. Data Ingestion Patterns

### 1.1 Schema Evolution Pattern
Handle changing schemas gracefully over time.
```python
# Allow schema evolution with mergeSchema
df = spark.read.option("mergeSchema", "true").parquet("path/to/data")

# Use schema inference with sampling
df = spark.read.option("samplingRatio", "0.1").json("path/to/json")
```

### 1.2 Incremental Load Pattern
Load only new or changed data since last load.
```python
from pyspark.sql.functions import col

# Read with date partition filtering
df = spark.read.parquet("path/to/data").filter(col("date") > last_load_date)

# High watermark pattern
max_id = spark.read.table("target_table").agg({"id": "max"}).collect()[0][0]
df_incremental = spark.read.jdbc(...).filter(col("id") > max_id)
```

### 1.3 Multiformat Reader Pattern
Abstract data source formats behind a unified interface.
```python
class DataReader:
    def __init__(self, spark):
        self.spark = spark
    
    def read(self, format_type, path, **options):
        readers = {
            'parquet': lambda: self.spark.read.parquet(path),
            'csv': lambda: self.spark.read.option("header", "true").csv(path),
            'json': lambda: self.spark.read.json(path),
            'delta': lambda: self.spark.read.format("delta").load(path)
        }
        return readers.get(format_type, lambda: None)()
```

### 1.4 Data Source Abstraction Pattern
Decouple business logic from data source specifics.
```python
from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    def read(self, spark):
        pass

class S3DataSource(DataSource):
    def __init__(self, bucket, key):
        self.bucket = bucket
        self.key = key
    
    def read(self, spark):
        return spark.read.parquet(f"s3://{self.bucket}/{self.key}")

class JDBCDataSource(DataSource):
    def __init__(self, url, table, properties):
        self.url = url
        self.table = table
        self.properties = properties
    
    def read(self, spark):
        return spark.read.jdbc(self.url, self.table, properties=self.properties)
```

### 1.5 Partition Pruning Pattern
Read only necessary partitions to reduce I/O.
```python
# Partition by date for efficient querying
df = spark.read.parquet("path/to/data") \
    .filter((col("year") == 2024) & (col("month") == 10))

# Dynamic partition pruning
df = spark.read.parquet("path/to/data") \
    .join(broadcast(dimension_df), "key") \
    .filter(col("date") >= "2024-01-01")
```

---

## 2. Data Transformation Patterns

### 2.1 Pipeline Pattern
Chain transformations in a reusable, composable way.
```python
from pyspark.sql import DataFrame

class TransformationPipeline:
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, func):
        self.transformations.append(func)
        return self
    
    def execute(self, df: DataFrame) -> DataFrame:
        for transform in self.transformations:
            df = transform(df)
        return df

# Usage
pipeline = TransformationPipeline()
pipeline.add_transformation(lambda df: df.filter(col("age") > 18))
pipeline.add_transformation(lambda df: df.withColumn("name", upper(col("name"))))
result = pipeline.execute(input_df)
```

### 2.2 Transformation Registry Pattern
Register and manage reusable transformations.
```python
class TransformationRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

@TransformationRegistry.register("clean_names")
def clean_names(df):
    return df.withColumn("name", trim(upper(col("name"))))

# Usage
df_cleaned = TransformationRegistry.get("clean_names")(df)
```

### 2.3 UDF Pattern with Error Handling
Create safe User Defined Functions.
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

@udf(returnType=IntegerType())
def safe_divide(numerator, denominator):
    try:
        if denominator == 0:
            return None
        return int(numerator / denominator)
    except:
        return None

# Using pandas UDF for better performance
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(IntegerType())
def safe_divide_pandas(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, pd.NA)).astype('Int64')
```

### 2.4 Window Function Pattern
Perform calculations across row windows.
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead

# Ranking pattern
window_spec = Window.partitionBy("department").orderBy(col("salary").desc())
df_ranked = df.withColumn("rank", rank().over(window_spec))

# Running total pattern
window_spec = Window.partitionBy("customer_id").orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
df_running = df.withColumn("running_total", sum("amount").over(window_spec))

# Moving average pattern
window_spec = Window.partitionBy("product_id").orderBy("date") \
    .rowsBetween(-6, 0)  # 7-day window
df_ma = df.withColumn("moving_avg", avg("sales").over(window_spec))

# Previous/Next value pattern
window_spec = Window.partitionBy("user_id").orderBy("timestamp")
df_with_prev = df.withColumn("previous_action", lag("action", 1).over(window_spec))
```

### 2.5 Slowly Changing Dimension (SCD) Patterns

#### SCD Type 1 (Overwrite)
```python
def scd_type1_merge(target_table, source_df, key_cols):
    """Overwrite existing records with new values"""
    from delta.tables import DeltaTable
    
    delta_table = DeltaTable.forName(spark, target_table)
    
    merge_condition = " AND ".join([f"target.{col} = source.{col}" for col in key_cols])
    
    delta_table.alias("target").merge(
        source_df.alias("source"),
        merge_condition
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
```

#### SCD Type 2 (Historical Tracking)
```python
from pyspark.sql.functions import current_timestamp, lit

def scd_type2_merge(target_table, source_df, key_cols):
    """Maintain history with effective dates"""
    from delta.tables import DeltaTable
    
    delta_table = DeltaTable.forName(spark, target_table)
    merge_condition = " AND ".join([f"target.{col} = source.{col}" for col in key_cols])
    merge_condition += " AND target.is_current = true"
    
    # Close existing records
    delta_table.alias("target").merge(
        source_df.alias("source"),
        merge_condition
    ).whenMatchedUpdate(
        set={
            "is_current": lit(False),
            "end_date": current_timestamp()
        }
    ).execute()
    
    # Insert new records
    new_records = source_df.withColumn("start_date", current_timestamp()) \
        .withColumn("end_date", lit(None)) \
        .withColumn("is_current", lit(True))
    new_records.write.format("delta").mode("append").saveAsTable(target_table)
```

### 2.6 Explode and Collect Pattern
Work with nested/array data structures.
```python
from pyspark.sql.functions import explode, collect_list, collect_set, struct

# Explode arrays
df_exploded = df.select("id", explode("items").alias("item"))

# Collect back to arrays
df_collected = df.groupBy("id").agg(
    collect_list("item").alias("items"),
    collect_set("category").alias("unique_categories")
)

# Struct aggregation
df_structured = df.groupBy("customer_id").agg(
    collect_list(struct("order_id", "amount", "date")).alias("orders")
)
```

### 2.7 Pivot and Unpivot Pattern
Reshape data between wide and long formats.
```python
# Pivot (long to wide)
df_pivot = df.groupBy("customer_id").pivot("product_category").agg(
    sum("amount").alias("total"),
    count("*").alias("count")
)

# Unpivot (wide to long)
from pyspark.sql.functions import expr, array, lit

# Stack function for unpivot
df_unpivot = df.selectExpr(
    "customer_id",
    "stack(3, 'cat1', cat1, 'cat2', cat2, 'cat3', cat3) as (category, amount)"
)
```

### 2.8 Deduplication Pattern
Remove duplicate records efficiently.
```python
# Simple deduplication
df_deduped = df.dropDuplicates(["id"])

# Deduplication with preference (keep latest)
from pyspark.sql.window import Window

window_spec = Window.partitionBy("id").orderBy(col("timestamp").desc())
df_deduped = df.withColumn("rn", row_number().over(window_spec)) \
    .filter(col("rn") == 1) \
    .drop("rn")

# Approximate deduplication for large datasets
df_deduped = df.dropDuplicates(["id"]).repartition("id")
```

### 2.9 Data Enrichment Pattern
Join and enhance data from multiple sources.
```python
# Broadcast join for small lookup tables
from pyspark.sql.functions import broadcast

df_enriched = df.join(broadcast(lookup_df), "key", "left")

# Multiple enrichment layers
def enrich_data(df):
    df = df.join(broadcast(country_lookup), "country_code", "left")
    df = df.join(broadcast(product_lookup), "product_id", "left")
    df = df.join(customer_df, "customer_id", "left")
    return df

# Coalesce for default values
df_enriched = df.join(lookup_df, "key", "left") \
    .withColumn("category", coalesce(col("lookup_category"), lit("UNKNOWN")))
```

---

## 3. Data Quality Patterns

### 3.1 Validation Pattern
Validate data quality rules.
```python
from pyspark.sql.functions import when, col, sum as _sum

class DataValidator:
    def __init__(self, df):
        self.df = df
        self.validations = []
    
    def add_rule(self, name, condition):
        self.validations.append((name, condition))
        return self
    
    def validate(self):
        results = {}
        for name, condition in self.validations:
            invalid_count = self.df.filter(~condition).count()
            total_count = self.df.count()
            results[name] = {
                'invalid_count': invalid_count,
                'valid_percentage': ((total_count - invalid_count) / total_count) * 100
            }
        return results

# Usage
validator = DataValidator(df)
validator.add_rule("age_positive", col("age") > 0)
validator.add_rule("email_not_null", col("email").isNotNull())
results = validator.validate()
```

### 3.2 Null Handling Pattern
Systematic approach to handling missing values.
```python
from pyspark.sql.functions import when, col, lit, mean, last

# Strategy pattern for null handling
class NullHandlingStrategy:
    def handle(self, df, column):
        raise NotImplementedError

class DropNullStrategy(NullHandlingStrategy):
    def handle(self, df, column):
        return df.filter(col(column).isNotNull())

class FillDefaultStrategy(NullHandlingStrategy):
    def __init__(self, default_value):
        self.default_value = default_value
    
    def handle(self, df, column):
        return df.withColumn(column, coalesce(col(column), lit(self.default_value)))

class FillMeanStrategy(NullHandlingStrategy):
    def handle(self, df, column):
        mean_value = df.select(mean(col(column))).collect()[0][0]
        return df.fillna({column: mean_value})

class ForwardFillStrategy(NullHandlingStrategy):
    def __init__(self, partition_cols, order_col):
        self.partition_cols = partition_cols
        self.order_col = order_col
    
    def handle(self, df, column):
        window = Window.partitionBy(self.partition_cols) \
            .orderBy(self.order_col) \
            .rowsBetween(Window.unboundedPreceding, 0)
        return df.withColumn(column, last(col(column), ignorenulls=True).over(window))
```

### 3.3 Outlier Detection Pattern
Identify and handle outliers.
```python
from pyspark.sql.functions import percentile_approx

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
    Q1, Q3 = quantiles[0], quantiles[1]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df.withColumn(
        f"{column}_is_outlier",
        when((col(column) < lower_bound) | (col(column) > upper_bound), True)
        .otherwise(False)
    )

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    stats = df.select(mean(col(column)), stddev(col(column))).collect()[0]
    mean_val, stddev_val = stats[0], stats[1]
    
    return df.withColumn(
        f"{column}_zscore",
        abs((col(column) - lit(mean_val)) / lit(stddev_val))
    ).withColumn(
        f"{column}_is_outlier",
        col(f"{column}_zscore") > threshold
    )
```

### 3.4 Data Profiling Pattern
Analyze and understand data characteristics.
```python
def profile_dataframe(df):
    """Generate comprehensive data profile"""
    profile = {
        'row_count': df.count(),
        'column_count': len(df.columns),
        'columns': {}
    }
    
    for col_name in df.columns:
        col_stats = df.select(
            count(col(col_name)).alias('count'),
            count(when(col(col_name).isNull(), True)).alias('null_count'),
            countDistinct(col(col_name)).alias('distinct_count')
        ).collect()[0]
        
        profile['columns'][col_name] = {
            'count': col_stats['count'],
            'null_count': col_stats['null_count'],
            'null_percentage': (col_stats['null_count'] / profile['row_count']) * 100,
            'distinct_count': col_stats['distinct_count'],
            'cardinality': col_stats['distinct_count'] / profile['row_count']
        }
    
    return profile
```

### 3.5 Schema Validation Pattern
Enforce and validate schemas.
```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

class SchemaValidator:
    def __init__(self, expected_schema):
        self.expected_schema = expected_schema
    
    def validate(self, df):
        """Validate DataFrame against expected schema"""
        errors = []
        
        # Check column names
        expected_cols = set(self.expected_schema.fieldNames())
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            errors.append(f"Extra columns: {extra_cols}")
        
        # Check data types
        for field in self.expected_schema.fields:
            if field.name in df.columns:
                actual_type = df.schema[field.name].dataType
                if actual_type != field.dataType:
                    errors.append(
                        f"Column {field.name}: expected {field.dataType}, got {actual_type}"
                    )
        
        return {'valid': len(errors) == 0, 'errors': errors}

# Usage
expected_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("age", IntegerType(), True)
])

validator = SchemaValidator(expected_schema)
result = validator.validate(df)
```

---

## 4. Performance Optimization Patterns

### 4.1 Broadcast Join Pattern
Optimize joins with small tables.
```python
from pyspark.sql.functions import broadcast

# Explicit broadcast
df_result = large_df.join(broadcast(small_df), "key")

# Auto-broadcast threshold configuration
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10485760)  # 10MB

# Disable broadcast for specific join
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
df_result = large_df.join(small_df, "key")
```

### 4.2 Partitioning Pattern
Optimize data distribution.
```python
# Repartition for parallelism
df_repartitioned = df.repartition(200)

# Repartition by column for better joins
df_repartitioned = df.repartition("customer_id")

# Coalesce to reduce partitions (no shuffle)
df_coalesced = df.coalesce(10)

# Adaptive partitioning
df_adaptive = df.repartition(col("date"), col("region"))

# Partition writing
df.write.partitionBy("year", "month").parquet("path/to/data")
```

### 4.3 Caching Pattern
Cache frequently accessed data.
```python
# Cache in memory
df.cache()
# or
df.persist()

# Cache with storage level
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# Unpersist when done
df.unpersist()

# Selective caching strategy
def process_with_caching(df):
    # Cache after expensive operations
    df_filtered = df.filter(expensive_condition).cache()
    
    # Multiple operations use cached data
    result1 = df_filtered.agg(...)
    result2 = df_filtered.groupBy(...)
    
    # Unpersist when done
    df_filtered.unpersist()
    
    return result1, result2
```

### 4.4 Bucketing Pattern
Optimize joins and aggregations through bucketing.
```python
# Write with bucketing
df.write.bucketBy(100, "customer_id") \
    .sortBy("order_date") \
    .saveAsTable("orders_bucketed")

# Both tables bucketed on join key = shuffle-free join
df_orders = spark.table("orders_bucketed")
df_customers = spark.table("customers_bucketed")
df_joined = df_orders.join(df_customers, "customer_id")
```

### 4.5 Predicate Pushdown Pattern
Filter data at source.
```python
# Partition pruning
df = spark.read.parquet("path/to/data") \
    .filter((col("year") == 2024) & (col("month") == 10))

# Column pruning - select only needed columns
df = spark.read.parquet("path/to/data") \
    .select("id", "name", "amount") \
    .filter(col("amount") > 1000)

# JDBC predicate pushdown
df = spark.read.jdbc(
    url=jdbc_url,
    table="large_table",
    predicates=["date >= '2024-01-01' AND date < '2024-02-01'"],
    properties=properties
)
```

### 4.6 Salting Pattern
Handle data skew.
```python
from pyspark.sql.functions import concat, lit, rand, floor

# Add salt to skewed key
num_salts = 10
df_salted = df.withColumn("salt", floor(rand() * num_salts))
df_salted = df_salted.withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))

# Join with replicated dimension
dimension_salted = dimension_df.crossJoin(
    spark.range(num_salts).withColumnRenamed("id", "salt")
).withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))

result = df_salted.join(dimension_salted, "salted_key")

# Remove salt after join
result = result.drop("salt", "salted_key")
```

### 4.7 Columnar Storage Pattern
Optimize for analytical queries.
```python
# Use Parquet for column-oriented storage
df.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet("path/to/parquet")

# Use ORC for Hive compatibility
df.write.mode("overwrite") \
    .option("compression", "zlib") \
    .orc("path/to/orc")

# Delta Lake for ACID transactions
df.write.format("delta") \
    .mode("overwrite") \
    .save("path/to/delta")
```

### 4.8 Adaptive Query Execution (AQE) Pattern
Leverage runtime optimization.
```python
# Enable AQE
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Configure AQE thresholds
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1")
```

### 4.9 DataFrame vs SQL Pattern
Choose the right API for performance.
```python
# DataFrame API - better for programmatic access
df_result = df.filter(col("age") > 18) \
    .groupBy("department") \
    .agg(avg("salary").alias("avg_salary"))

# SQL API - better for complex queries and optimization
df.createOrReplaceTempView("employees")
df_result = spark.sql("""
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    WHERE age > 18
    GROUP BY department
""")

# Catalyst optimizer handles both similarly
```

---

## 5. Data Storage Patterns

### 5.1 Delta Lake Pattern
ACID transactions and time travel.
```python
# Write Delta table
df.write.format("delta") \
    .mode("overwrite") \
    .save("/path/to/delta-table")

# Append to Delta table
df.write.format("delta") \
    .mode("append") \
    .save("/path/to/delta-table")

# Merge (Upsert) pattern
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/path/to/delta-table")

delta_table.alias("target").merge(
    source_df.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "value": "source.value",
    "updated_at": "source.updated_at"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "value": "source.value",
    "updated_at": "source.updated_at"
}).execute()

# Time travel
df_historical = spark.read.format("delta") \
    .option("versionAsOf", 5) \
    .load("/path/to/delta-table")

df_timestamp = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-01") \
    .load("/path/to/delta-table")

# Optimize and vacuum
delta_table.optimize().executeCompaction()
delta_table.vacuum(168)  # Retain 7 days
```

### 5.2 Partitioned Storage Pattern
Organize data by partition keys.
```python
# Write with partitioning
df.write.partitionBy("year", "month", "day") \
    .mode("overwrite") \
    .parquet("/path/to/data")

# Dynamic partition overwrite
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
df.write.mode("overwrite") \
    .partitionBy("date") \
    .parquet("/path/to/data")

# Partition discovery
df = spark.read.parquet("/path/to/data")  # Automatically discovers partitions
```

### 5.3 Table Format Pattern
Choose appropriate table format.
```python
# Managed table
df.write.saveAsTable("database.table_name")

# External table
df.write.option("path", "/path/to/data") \
    .saveAsTable("database.table_name")

# Temporary view
df.createOrReplaceTempView("temp_view")

# Global temporary view
df.createOrReplaceGlobalTempView("global_temp_view")
# Access via: global_temp.global_temp_view
```

### 5.4 Compaction Pattern
Optimize file sizes.
```python
# Small file problem - compact files
df = spark.read.parquet("/path/to/many/small/files")
df.coalesce(10).write.mode("overwrite").parquet("/path/to/output")

# Repartition before write to control file size
df.repartition(100).write.parquet("/path/to/output")

# Delta optimize
from delta.tables import DeltaTable
delta_table = DeltaTable.forPath(spark, "/path/to/delta")
delta_table.optimize().executeCompaction()

# Z-order optimization for multi-dimensional queries
delta_table.optimize().executeZOrderBy("country", "product_type")
```

### 5.5 Data Lake Pattern
Organize data in layers (Bronze/Silver/Gold).
```python
# Bronze layer - raw data
df_raw.write.format("delta") \
    .mode("append") \
    .save("/data/bronze/source_name")

# Silver layer - cleaned and validated
df_cleaned = clean_and_validate(df_raw)
df_cleaned.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("date") \
    .save("/data/silver/source_name")

# Gold layer - aggregated and business logic applied
df_aggregated = apply_business_logic(df_cleaned)
df_aggregated.write.format("delta") \
    .mode("overwrite") \
    .save("/data/gold/business_view")
```

---

## 6. Error Handling Patterns

### 6.1 Try-Catch Pattern
Handle errors gracefully.
```python
from pyspark.sql.utils import AnalysisException

def safe_read(spark, path):
    try:
        return spark.read.parquet(path)
    except AnalysisException as e:
        print(f"Error reading {path}: {e}")
        return spark.createDataFrame([], schema=StructType([]))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# UDF with error handling
@udf(StringType())
def safe_parse(value):
    try:
        return json.loads(value)['key']
    except (json.JSONDecodeError, KeyError):
        return None
```

### 6.2 Dead Letter Queue Pattern
Capture failed records.
```python
from pyspark.sql.functions import when

def process_with_dlq(df, processing_func):
    """Process data and capture failures"""
    # Add processing result column
    df_processed = df.withColumn("processed", processing_func(col("data")))
    
    # Split success and failures
    df_success = df_processed.filter(col("processed").isNotNull())
    df_failed = df_processed.filter(col("processed").isNull())
    
    # Write failures to DLQ
    df_failed.write.mode("append").parquet("/path/to/dlq")
    
    return df_success

# Alternative: use try-catch UDF
@udf(StructType([
    StructField("result", StringType()),
    StructField("error", StringType())
]))
def process_with_error(value):
    try:
        result = complex_processing(value)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

### 6.3 Retry Pattern
Retry failed operations.
```python
from time import sleep

def retry_operation(func, max_retries=3, delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            sleep(delay * (2 ** attempt))

# Usage
df = retry_operation(lambda: spark.read.jdbc(url, table, properties))
```

### 6.4 Circuit Breaker Pattern
Prevent cascading failures.
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
            self.failure_count = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise
```

### 6.5 Validation with Early Exit Pattern
Fail fast on validation errors.
```python
def validate_and_process(df):
    """Validate data before expensive processing"""
    # Check row count
    if df.count() == 0:
        raise ValueError("DataFrame is empty")
    
    # Check required columns
    required_cols = ["id", "name", "date"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data quality
    null_count = df.filter(col("id").isNull()).count()
    if null_count > 0:
        raise ValueError(f"Found {null_count} null IDs")
    
    # Proceed with processing
    return process_data(df)
```

---

## 7. Testing Patterns

### 7.1 Unit Testing Pattern
Test transformation functions.
```python
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("test") \
        .master("local[*]") \
        .getOrCreate()

def test_transformation(spark):
    # Arrange
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("value", IntegerType())
    ])
    input_data = [(1, 10), (2, 20)]
    df_input = spark.createDataFrame(input_data, schema)
    
    # Act
    df_result = my_transformation(df_input)
    
    # Assert
    expected_data = [(1, 20), (2, 40)]
    df_expected = spark.createDataFrame(expected_data, schema)
    assert df_result.collect() == df_expected.collect()
```

### 7.2 Schema Testing Pattern
Validate DataFrame schemas.
```python
def assert_schema_equal(df, expected_schema):
    """Assert DataFrame has expected schema"""
    assert df.schema == expected_schema, \
        f"Schema mismatch:\nExpected: {expected_schema}\nActual: {df.schema}"

def test_schema(spark):
    df = create_my_dataframe(spark)
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False)
    ])
    assert_schema_equal(df, expected_schema)
```

### 7.3 Data Testing Pattern
Validate data quality in tests.
```python
def test_data_quality(spark):
    df = create_test_dataframe(spark)
    
    # Test no nulls in required columns
    null_count = df.filter(col("id").isNull()).count()
    assert null_count == 0, f"Found {null_count} null IDs"
    
    # Test value ranges
    invalid_ages = df.filter((col("age") < 0) | (col("age") > 150)).count()
    assert invalid_ages == 0, f"Found {invalid_ages} invalid ages"
    
    # Test uniqueness
    total_count = df.count()
    unique_count = df.select("id").distinct().count()
    assert total_count == unique_count, "Duplicate IDs found"
```

### 7.4 Integration Testing Pattern
Test end-to-end workflows.
```python
def test_etl_pipeline(spark, tmp_path):
    # Setup
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    
    # Create test data
    df_input = spark.createDataFrame([(1, "A"), (2, "B")])
    df_input.write.parquet(str(input_path))
    
    # Run pipeline
    run_etl_pipeline(spark, str(input_path), str(output_path))
    
    # Verify output
    df_output = spark.read.parquet(str(output_path))
    assert df_output.count() == 2
    assert "transformed_column" in df_output.columns
```

### 7.5 Mock Pattern
Mock external dependencies.
```python
from unittest.mock import Mock, patch

def test_with_mock_jdbc(spark, monkeypatch):
    # Mock JDBC read
    mock_df = spark.createDataFrame([(1, "test")])
    
    with patch.object(spark.read, 'jdbc', return_value=mock_df):
        result = my_function_using_jdbc(spark)
        assert result.count() == 1
```

---

## 8. Streaming Patterns

### 8.1 Structured Streaming Pattern
Process streaming data.
```python
# Read stream
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic") \
    .load()

# Transform
df_transformed = df_stream \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Write stream
query = df_transformed.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "/path/to/output") \
    .option("checkpointLocation", "/path/to/checkpoint") \
    .start()

query.awaitTermination()
```

### 8.2 Windowing Pattern
Time-based aggregations.
```python
from pyspark.sql.functions import window

# Tumbling window
df_windowed = df_stream \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(window(col("timestamp"), "5 minutes")) \
    .agg(count("*").alias("count"))

# Sliding window
df_sliding = df_stream \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(window(col("timestamp"), "10 minutes", "5 minutes")) \
    .agg(avg("value").alias("avg_value"))

# Session window
df_session = df_stream \
    .groupBy(
        session_window(col("timestamp"), "5 minutes"),
        col("user_id")
    ).count()
```

### 8.3 Watermarking Pattern
Handle late data.
```python
# Define watermark
df_with_watermark = df_stream \
    .withWatermark("event_time", "10 minutes")

# Aggregation with watermark
df_aggregated = df_with_watermark \
    .groupBy(
        window(col("event_time"), "5 minutes"),
        col("user_id")
    ).agg(sum("amount").alias("total"))

# Late data handling
query = df_aggregated.writeStream \
    .outputMode("update") \
    .option("checkpointLocation", "/checkpoint") \
    .start()
```

### 8.4 Trigger Patterns
Control micro-batch processing.
```python
# Continuous processing (experimental)
query = df.writeStream \
    .trigger(continuous="1 second") \
    .start()

# Fixed interval micro-batch
query = df.writeStream \
    .trigger(processingTime="10 seconds") \
    .start()

# One-time trigger (batch-like)
query = df.writeStream \
    .trigger(once=True) \
    .start()

# Available now (process all available data)
query = df.writeStream \
    .trigger(availableNow=True) \
    .start()
```

### 8.5 Stateful Operations Pattern
Maintain state across batches.
```python
from pyspark.sql.functions import *

# flatMapGroupsWithState
def update_state(key, values, state):
    # Custom stateful logic
    if state.exists:
        old_state = state.get
        new_state = old_state + sum(values)
    else:
        new_state = sum(values)
    
    state.update(new_state)
    return new_state

df_stateful = df_stream \
    .groupByKey(lambda x: x.key) \
    .flatMapGroupsWithState(update_state, ...)

# mapGroupsWithState for timeout-based operations
df_session = df_stream \
    .groupByKey(lambda x: x.user_id) \
    .mapGroupsWithState(update_session_state, ...)
```

### 8.6 Multiple Sinks Pattern
Write stream to multiple destinations.
```python
# Using foreachBatch
def write_to_multiple_sinks(batch_df, batch_id):
    # Write to Parquet
    batch_df.write.mode("append").parquet("/path/to/parquet")
    
    # Write to Delta
    batch_df.write.format("delta").mode("append").save("/path/to/delta")
    
    # Write to JDBC
    batch_df.write.jdbc(url, "table", mode="append")

query = df_stream.writeStream \
    .foreachBatch(write_to_multiple_sinks) \
    .start()
```

### 8.7 Exactly-Once Semantics Pattern
Ensure data consistency.
```python
# Idempotent writes with Delta Lake
query = df_stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/checkpoint") \
    .start("/path/to/delta")

# Kafka with idempotent writes
query = df_stream.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "output") \
    .option("checkpointLocation", "/checkpoint") \
    .start()
```

---

## 9. Machine Learning Patterns

### 9.1 Feature Engineering Pattern
Prepare features for ML.
```python
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Create feature pipeline
indexer = StringIndexer(inputCol="category", outputCol="category_index")
encoder = OneHotEncoder(inputCol="category_index", outputCol="category_vec")
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "category_vec"],
    outputCol="features"
)

pipeline = Pipeline(stages=[indexer, encoder, assembler])
model = pipeline.fit(df_train)
df_transformed = model.transform(df_train)
```

### 9.2 Train-Test Split Pattern
Split data for validation.
```python
# Random split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Stratified split (manual)
train_df = df.sampleBy("label", fractions={0: 0.8, 1: 0.8}, seed=42)
test_df = df.join(train_df, on="id", how="left_anti")

# Time-based split
train_df = df.filter(col("date") < "2024-01-01")
test_df = df.filter(col("date") >= "2024-01-01")
```

### 9.3 Cross-Validation Pattern
Tune hyperparameters.
```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

# Define model
lr = LogisticRegression()

# Parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Cross-validator
cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=param_grid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=5,
    seed=42
)

# Train
cv_model = cv.fit(train_df)
best_model = cv_model.bestModel
```

### 9.4 Model Persistence Pattern
Save and load models.
```python
# Save model
model.write().overwrite().save("/path/to/model")

# Load model
from pyspark.ml.classification import LogisticRegressionModel
loaded_model = LogisticRegressionModel.load("/path/to/model")

# Save pipeline
pipeline_model.write().overwrite().save("/path/to/pipeline")

# Load pipeline
from pyspark.ml import PipelineModel
loaded_pipeline = PipelineModel.load("/path/to/pipeline")
```

### 9.5 Batch Prediction Pattern
Score large datasets efficiently.
```python
# Batch scoring
predictions = model.transform(df_to_score)

# With repartitioning for parallelism
predictions = model.transform(
    df_to_score.repartition(200, "partition_key")
)

# Save predictions
predictions.select("id", "prediction", "probability") \
    .write.mode("overwrite") \
    .partitionBy("date") \
    .parquet("/path/to/predictions")
```

### 9.6 Feature Store Pattern
Centralize feature management.
```python
class FeatureStore:
    def __init__(self, spark):
        self.spark = spark
        self.base_path = "/feature_store"
    
    def write_feature(self, df, feature_name, version):
        path = f"{self.base_path}/{feature_name}/v{version}"
        df.write.format("delta").mode("overwrite").save(path)
    
    def read_feature(self, feature_name, version):
        path = f"{self.base_path}/{feature_name}/v{version}"
        return self.spark.read.format("delta").load(path)
    
    def get_features(self, entity_df, feature_list):
        """Join multiple features for training"""
        result = entity_df
        for feature_name, version in feature_list:
            feature_df = self.read_feature(feature_name, version)
            result = result.join(feature_df, "entity_id", "left")
        return result
```

### 9.7 Online-Offline Consistency Pattern
Ensure training and serving use same features.
```python
def compute_features(df, mode="offline"):
    """Compute features with same logic for training and serving"""
    df = df.withColumn("feature1", compute_feature1(col("raw_data")))
    df = df.withColumn("feature2", compute_feature2(col("raw_data")))
    
    if mode == "offline":
        # Additional features for batch training
        df = df.withColumn("historical_feature", compute_historical(col("data")))
    
    return df

# Offline (training)
df_train_features = compute_features(df_train, mode="offline")

# Online (serving)
df_serving_features = compute_features(df_realtime, mode="online")
```

---

## 10. Architecture Patterns

### 10.1 Repository Pattern
Abstract data access layer.
```python
from abc import ABC, abstractmethod

class DataRepository(ABC):
    @abstractmethod
    def read(self, **kwargs):
        pass
    
    @abstractmethod
    def write(self, df, **kwargs):
        pass

class ParquetRepository(DataRepository):
    def __init__(self, spark, base_path):
        self.spark = spark
        self.base_path = base_path
    
    def read(self, table_name):
        return self.spark.read.parquet(f"{self.base_path}/{table_name}")
    
    def write(self, df, table_name, mode="overwrite"):
        df.write.mode(mode).parquet(f"{self.base_path}/{table_name}")

class DeltaRepository(DataRepository):
    def __init__(self, spark, base_path):
        self.spark = spark
        self.base_path = base_path
    
    def read(self, table_name):
        return self.spark.read.format("delta").load(f"{self.base_path}/{table_name}")
    
    def write(self, df, table_name, mode="overwrite"):
        df.write.format("delta").mode(mode).save(f"{self.base_path}/{table_name}")
```

### 10.2 Factory Pattern
Create objects based on configuration.
```python
class DataSourceFactory:
    @staticmethod
    def create_reader(spark, source_type, **config):
        if source_type == "parquet":
            return spark.read.parquet(config['path'])
        elif source_type == "csv":
            return spark.read.option("header", "true").csv(config['path'])
        elif source_type == "jdbc":
            return spark.read.jdbc(
                config['url'],
                config['table'],
                properties=config.get('properties', {})
            )
        elif source_type == "kafka":
            return spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", config['servers']) \
                .option("subscribe", config['topic']) \
                .load()
        else:
            raise ValueError(f"Unknown source type: {source_type}")

# Usage
df = DataSourceFactory.create_reader(
    spark,
    "parquet",
    path="/data/input"
)
```

### 10.3 Builder Pattern
Construct complex objects step by step.
```python
class DataFrameBuilder:
    def __init__(self, spark):
        self.spark = spark
        self.df = None
        self.transformations = []
    
    def from_source(self, source_type, path):
        self.df = self.spark.read.format(source_type).load(path)
        return self
    
    def filter(self, condition):
        self.transformations.append(lambda df: df.filter(condition))
        return self
    
    def select_columns(self, *cols):
        self.transformations.append(lambda df: df.select(*cols))
        return self
    
    def with_column(self, col_name, expression):
        self.transformations.append(
            lambda df: df.withColumn(col_name, expression)
        )
        return self
    
    def build(self):
        result = self.df
        for transform in self.transformations:
            result = transform(result)
        return result

# Usage
df = DataFrameBuilder(spark) \
    .from_source("parquet", "/data/input") \
    .filter(col("age") > 18) \
    .select_columns("id", "name", "age") \
    .with_column("category", lit("adult")) \
    .build()
```

### 10.4 Strategy Pattern
Encapsulate algorithms.
```python
from abc import ABC, abstractmethod

class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, df, group_cols):
        pass

class SumStrategy(AggregationStrategy):
    def __init__(self, agg_col):
        self.agg_col = agg_col
    
    def aggregate(self, df, group_cols):
        return df.groupBy(*group_cols).agg(sum(self.agg_col).alias("total"))

class AvgStrategy(AggregationStrategy):
    def __init__(self, agg_col):
        self.agg_col = agg_col
    
    def aggregate(self, df, group_cols):
        return df.groupBy(*group_cols).agg(avg(self.agg_col).alias("average"))

class Aggregator:
    def __init__(self, strategy: AggregationStrategy):
        self.strategy = strategy
    
    def execute(self, df, group_cols):
        return self.strategy.aggregate(df, group_cols)

# Usage
aggregator = Aggregator(SumStrategy("amount"))
result = aggregator.execute(df, ["customer_id", "date"])
```

### 10.5 Observer Pattern
Monitor and react to events.
```python
class DataFrameObserver:
    def update(self, df):
        pass

class RowCountObserver(DataFrameObserver):
    def update(self, df):
        count = df.count()
        print(f"DataFrame has {count} rows")

class SchemaObserver(DataFrameObserver):
    def update(self, df):
        print(f"Schema: {df.schema}")

class DataFrameSubject:
    def __init__(self):
        self.observers = []
    
    def attach(self, observer):
        self.observers.append(observer)
    
    def notify(self, df):
        for observer in self.observers:
            observer.update(df)

# Usage
subject = DataFrameSubject()
subject.attach(RowCountObserver())
subject.attach(SchemaObserver())

df = spark.read.parquet("/data/input")
subject.notify(df)
```

### 10.6 Dependency Injection Pattern
Inject dependencies for testability.
```python
class DataProcessor:
    def __init__(self, reader, writer, transformer):
        self.reader = reader
        self.writer = writer
        self.transformer = transformer
    
    def process(self):
        df = self.reader.read()
        df_transformed = self.transformer.transform(df)
        self.writer.write(df_transformed)

# Production
processor = DataProcessor(
    reader=ProductionReader(spark),
    writer=ProductionWriter(spark),
    transformer=ProductionTransformer()
)

# Testing
processor = DataProcessor(
    reader=MockReader(),
    writer=MockWriter(),
    transformer=TestTransformer()
)
```

### 10.7 Configuration Pattern
Manage application configuration.
```python
import yaml
from dataclasses import dataclass

@dataclass
class SparkConfig:
    app_name: str
    master: str
    executor_memory: str
    executor_cores: int

@dataclass
class DataConfig:
    input_path: str
    output_path: str
    format: str
    partitions: list

class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_spark_config(self):
        return SparkConfig(**self.config['spark'])
    
    def get_data_config(self):
        return DataConfig(**self.config['data'])

# config.yaml
"""
spark:
  app_name: "MyApp"
  master: "yarn"
  executor_memory: "4g"
  executor_cores: 2

data:
  input_path: "/data/input"
  output_path: "/data/output"
  format: "parquet"
  partitions: ["year", "month"]
"""

# Usage
config_mgr = ConfigManager("config.yaml")
spark_config = config_mgr.get_spark_config()
data_config = config_mgr.get_data_config()
```

### 10.8 Medallion Architecture Pattern
Organize data lake in layers.
```python
class MedallionArchitecture:
    def __init__(self, spark):
        self.spark = spark
        self.bronze_path = "/data/bronze"
        self.silver_path = "/data/silver"
        self.gold_path = "/data/gold"
    
    def ingest_to_bronze(self, source_df, table_name):
        """Raw data ingestion"""
        source_df.write.format("delta") \
            .mode("append") \
            .save(f"{self.bronze_path}/{table_name}")
    
    def bronze_to_silver(self, table_name, transformation_func):
        """Clean and validate data"""
        df_bronze = self.spark.read.format("delta") \
            .load(f"{self.bronze_path}/{table_name}")
        
        df_silver = transformation_func(df_bronze)
        
        df_silver.write.format("delta") \
            .mode("overwrite") \
            .save(f"{self.silver_path}/{table_name}")
    
    def silver_to_gold(self, table_name, aggregation_func):
        """Business-level aggregations"""
        df_silver = self.spark.read.format("delta") \
            .load(f"{self.silver_path}/{table_name}")
        
        df_gold = aggregation_func(df_silver)
        
        df_gold.write.format("delta") \
            .mode("overwrite") \
            .save(f"{self.gold_path}/{table_name}")
```

### 10.9 Microservice Pattern for Data Processing
Modular, independent data services.
```python
class DataService:
    def __init__(self, spark, config):
        self.spark = spark
        self.config = config
    
    def process(self, input_data):
        raise NotImplementedError

class CleaningService(DataService):
    def process(self, input_data):
        return input_data.dropna().dropDuplicates()

class EnrichmentService(DataService):
    def process(self, input_data):
        lookup_df = self.spark.read.table(self.config['lookup_table'])
        return input_data.join(broadcast(lookup_df), "key", "left")

class AggregationService(DataService):
    def process(self, input_data):
        return input_data.groupBy("category").agg(
            sum("amount").alias("total"),
            count("*").alias("count")
        )

class DataPipeline:
    def __init__(self, services):
        self.services = services
    
    def execute(self, input_data):
        result = input_data
        for service in self.services:
            result = service.process(result)
        return result

# Usage
pipeline = DataPipeline([
    CleaningService(spark, {}),
    EnrichmentService(spark, {'lookup_table': 'dim_products'}),
    AggregationService(spark, {})
])

output = pipeline.execute(input_df)
```

### 10.10 Event Sourcing Pattern
Track all changes as events.
```python
def event_sourcing_append(df_events, table_name):
    """Append events to event log"""
    df_events.withColumn("event_timestamp", current_timestamp()) \
        .withColumn("event_id", monotonically_increasing_id()) \
        .write.format("delta") \
        .mode("append") \
        .save(f"/events/{table_name}")

def rebuild_state_from_events(spark, table_name, as_of_timestamp=None):
    """Rebuild current state from event log"""
    df_events = spark.read.format("delta").load(f"/events/{table_name}")
    
    if as_of_timestamp:
        df_events = df_events.filter(col("event_timestamp") <= as_of_timestamp)
    
    # Apply events in order
    window_spec = Window.partitionBy("entity_id") \
        .orderBy("event_timestamp") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    df_state = df_events.withColumn(
        "current_value",
        last("value", ignorenulls=True).over(window_spec)
    )
    
    return df_state.filter(col("is_latest"))
```

---

## 11. Advanced Transformation Patterns

### 11.1 Complex Type Handling Pattern
Work with maps, arrays, and structs.
```python
from pyspark.sql.functions import map_keys, map_values, array_contains, size, element_at

# Working with maps
df_with_map = df.withColumn("map_keys", map_keys(col("map_column"))) \
    .withColumn("map_values", map_values(col("map_column"))) \
    .withColumn("specific_value", col("map_column").getItem("key"))

# Working with arrays
df_with_array = df.withColumn("array_size", size(col("array_column"))) \
    .withColumn("contains_value", array_contains(col("array_column"), "value")) \
    .withColumn("first_element", element_at(col("array_column"), 1))

# Nested struct access
df_nested = df.select(
    col("struct_column.field1"),
    col("struct_column.nested_struct.field2")
)

# Transform array elements
from pyspark.sql.functions import transform, filter as array_filter, exists

df_transformed = df.withColumn(
    "transformed_array",
    transform(col("numbers"), lambda x: x * 2)
).withColumn(
    "filtered_array",
    array_filter(col("numbers"), lambda x: x > 10)
)
```

### 11.2 Column Expression Pattern
Create reusable column expressions.
```python
from pyspark.sql import Column
from pyspark.sql.functions import when, col, lit

class ColumnExpressions:
    @staticmethod
    def age_category(age_col: str) -> Column:
        """Categorize age into groups"""
        return when(col(age_col) < 18, "minor") \
            .when((col(age_col) >= 18) & (col(age_col) < 65), "adult") \
            .when(col(age_col) >= 65, "senior") \
            .otherwise("unknown")
    
    @staticmethod
    def is_weekend(date_col: str) -> Column:
        """Check if date is weekend"""
        from pyspark.sql.functions import dayofweek
        return dayofweek(col(date_col)).isin([1, 7])
    
    @staticmethod
    def full_name(first_col: str, last_col: str) -> Column:
        """Concatenate first and last name"""
        from pyspark.sql.functions import concat_ws
        return concat_ws(" ", col(first_col), col(last_col))

# Usage
df_categorized = df.withColumn("age_group", ColumnExpressions.age_category("age"))
df_weekend = df.withColumn("is_weekend", ColumnExpressions.is_weekend("date"))
```

### 11.3 Self-Join Pattern
Join dataframe with itself for complex comparisons.
```python
# Find pairs of related records
df_left = df.alias("left")
df_right = df.alias("right")

df_pairs = df_left.join(
    df_right,
    (col("left.category") == col("right.category")) & 
    (col("left.id") < col("right.id"))  # Avoid duplicates
).select(
    col("left.id").alias("id1"),
    col("right.id").alias("id2"),
    col("left.category")
)

# Find hierarchical relationships (manager-employee)
employees = df.alias("emp")
managers = df.alias("mgr")

df_hierarchy = employees.join(
    managers,
    col("emp.manager_id") == col("mgr.id"),
    "left"
).select(
    col("emp.id").alias("employee_id"),
    col("emp.name").alias("employee_name"),
    col("mgr.name").alias("manager_name")
)
```

### 11.4 Conditional Aggregation Pattern
Aggregate with conditions.
```python
from pyspark.sql.functions import sum, count, avg, when

# Multiple conditional aggregations in one pass
df_agg = df.groupBy("category").agg(
    count("*").alias("total_count"),
    sum(when(col("status") == "active", 1).otherwise(0)).alias("active_count"),
    sum(when(col("status") == "inactive", 1).otherwise(0)).alias("inactive_count"),
    avg(when(col("amount") > 100, col("amount"))).alias("avg_high_amount"),
    sum(when(col("region") == "US", col("revenue")).otherwise(0)).alias("us_revenue")
)

# Pivot with conditional aggregation
df_pivot_conditional = df.groupBy("date").agg(
    sum(when(col("category") == "A", col("amount"))).alias("category_a"),
    sum(when(col("category") == "B", col("amount"))).alias("category_b")
)
```

### 11.5 Fuzzy Matching Pattern
Match records with approximate similarity.
```python
from pyspark.sql.functions import levenshtein, soundex

# Levenshtein distance for fuzzy matching
df_fuzzy = df1.crossJoin(df2.select(
    col("name").alias("name2"),
    col("id").alias("id2")
)).withColumn("distance", levenshtein(col("name"), col("name2"))) \
  .filter(col("distance") <= 3)

# Soundex for phonetic matching
df_phonetic = df.withColumn("name_soundex", soundex(col("name")))

df_matched = df1.alias("a").join(
    df2.alias("b"),
    soundex(col("a.name")) == soundex(col("b.name")),
    "inner"
)

# Token-based matching
from pyspark.ml.feature import Tokenizer, HashingTF, MinHashLSH

tokenizer = Tokenizer(inputCol="name", outputCol="tokens")
hashingTF = HashingTF(inputCol="tokens", outputCol="features")
minhash = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)

pipeline = Pipeline(stages=[tokenizer, hashingTF, minhash])
model = pipeline.fit(df)
transformed = model.transform(df)

# Find similar records
similar = model.stages[-1].approxSimilarityJoin(
    transformed, transformed, 0.6, distCol="distance"
).filter(col("datasetA.id") < col("datasetB.id"))
```

---

## 12. Data Skew Handling Patterns

### 12.1 Skew Join Pattern
Handle skewed joins with isolation.
```python
from pyspark.sql.functions import broadcast, rand

def skewed_join(large_df, small_df, skewed_keys, join_key):
    """Handle skewed join by isolating hot keys"""
    
    # Separate skewed and non-skewed data
    skewed_large = large_df.filter(col(join_key).isin(skewed_keys))
    normal_large = large_df.filter(~col(join_key).isin(skewed_keys))
    
    # Normal join for non-skewed data
    normal_result = normal_large.join(small_df, join_key)
    
    # Broadcast join for skewed data
    skewed_result = skewed_large.join(broadcast(small_df), join_key)
    
    # Union results
    return normal_result.union(skewed_result)

# Usage
result = skewed_join(large_df, small_df, ["hot_key1", "hot_key2"], "key")
```

### 12.2 Adaptive Salting Pattern
Dynamic salting based on data distribution.
```python
from pyspark.sql.functions import col, lit, concat, floor, rand

def adaptive_salt_join(df_large, df_small, join_key, skew_threshold=10000):
    """Apply salting only to skewed keys"""
    
    # Identify skewed keys
    key_counts = df_large.groupBy(join_key).count()
    skewed_keys = key_counts.filter(col("count") > skew_threshold) \
        .select(join_key).rdd.flatMap(lambda x: x).collect()
    
    if not skewed_keys:
        # No skew detected, normal join
        return df_large.join(df_small, join_key)
    
    # Salt only skewed keys
    num_salts = 10
    
    df_large_salted = df_large.withColumn(
        "salt",
        when(col(join_key).isin(skewed_keys), floor(rand() * num_salts))
        .otherwise(lit(0))
    ).withColumn("salted_key", concat(col(join_key), lit("_"), col("salt")))
    
    # Replicate small df for skewed keys
    skewed_small = df_small.filter(col(join_key).isin(skewed_keys))
    normal_small = df_small.filter(~col(join_key).isin(skewed_keys))
    
    replicated_small = skewed_small.crossJoin(
        spark.range(num_salts).select(col("id").alias("salt"))
    ).withColumn("salted_key", concat(col(join_key), lit("_"), col("salt")))
    
    normal_small_salted = normal_small.withColumn("salt", lit(0)) \
        .withColumn("salted_key", concat(col(join_key), lit("_"), col("salt")))
    
    small_combined = replicated_small.union(normal_small_salted)
    
    # Join and cleanup
    result = df_large_salted.join(small_combined, "salted_key") \
        .drop("salt", "salted_key")
    
    return result
```

### 12.3 Iterative Broadcast Pattern
Handle medium-sized dimensions that don't fit in memory.
```python
def iterative_broadcast_join(large_df, medium_df, join_key, chunk_size=1000000):
    """Join by broadcasting medium df in chunks"""
    
    # Add row number to medium df
    from pyspark.sql.functions import monotonically_increasing_id
    medium_with_id = medium_df.withColumn("row_id", monotonically_increasing_id())
    
    # Calculate number of chunks
    total_rows = medium_df.count()
    num_chunks = (total_rows // chunk_size) + 1
    
    results = []
    for i in range(num_chunks):
        chunk = medium_with_id.filter(
            (col("row_id") >= i * chunk_size) & 
            (col("row_id") < (i + 1) * chunk_size)
        ).drop("row_id")
        
        # Broadcast join with chunk
        chunk_result = large_df.join(broadcast(chunk), join_key)
        results.append(chunk_result)
    
    # Union all results and deduplicate
    final_result = results[0]
    for result in results[1:]:
        final_result = final_result.union(result)
    
    return final_result.dropDuplicates()
```

---

## 13. Monitoring and Observability Patterns

### 13.1 Metrics Collection Pattern
Collect and track pipeline metrics.
```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class PipelineMetrics:
    job_name: str
    start_time: datetime
    end_time: datetime
    input_records: int
    output_records: int
    filtered_records: int
    error_records: int
    duration_seconds: float

class MetricsCollector:
    def __init__(self, spark):
        self.spark = spark
        self.metrics = []
    
    def collect_dataframe_metrics(self, df, stage_name):
        """Collect metrics for a dataframe"""
        count = df.count()
        
        metric = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'record_count': count,
            'partition_count': df.rdd.getNumPartitions(),
            'columns': len(df.columns)
        }
        
        self.metrics.append(metric)
        return metric
    
    def write_metrics(self, path):
        """Write metrics to storage"""
        metrics_df = self.spark.createDataFrame(self.metrics)
        metrics_df.write.mode("append") \
            .partitionBy("date") \
            .json(path)
    
    def log_metrics(self):
        """Log metrics to console"""
        for metric in self.metrics:
            print(json.dumps(metric, indent=2))

# Usage
collector = MetricsCollector(spark)
collector.collect_dataframe_metrics(df_input, "input")
df_transformed = transform(df_input)
collector.collect_dataframe_metrics(df_transformed, "transformed")
collector.write_metrics("/metrics")
```

### 13.2 Data Lineage Pattern
Track data lineage and transformations.
```python
class DataLineageTracker:
    def __init__(self):
        self.lineage = []
    
    def track_read(self, source_type, source_path, df):
        """Track data read operation"""
        self.lineage.append({
            'operation': 'READ',
            'source_type': source_type,
            'source_path': source_path,
            'record_count': df.count(),
            'timestamp': datetime.now().isoformat()
        })
    
    def track_transformation(self, operation_name, input_df, output_df):
        """Track transformation operation"""
        self.lineage.append({
            'operation': 'TRANSFORM',
            'transformation': operation_name,
            'input_records': input_df.count(),
            'output_records': output_df.count(),
            'timestamp': datetime.now().isoformat()
        })
    
    def track_write(self, destination_type, destination_path, df):
        """Track data write operation"""
        self.lineage.append({
            'operation': 'WRITE',
            'destination_type': destination_type,
            'destination_path': destination_path,
            'record_count': df.count(),
            'timestamp': datetime.now().isoformat()
        })
    
    def get_lineage(self):
        """Get full lineage"""
        return self.lineage
    
    def export_lineage(self, spark, output_path):
        """Export lineage to storage"""
        lineage_df = spark.createDataFrame(self.lineage)
        lineage_df.write.mode("overwrite").json(output_path)

# Usage
tracker = DataLineageTracker()
df = spark.read.parquet("/input")
tracker.track_read("parquet", "/input", df)

df_transformed = df.filter(col("age") > 18)
tracker.track_transformation("filter_adults", df, df_transformed)

df_transformed.write.parquet("/output")
tracker.track_write("parquet", "/output", df_transformed)
```

### 13.3 Performance Monitoring Pattern
Monitor query performance and resource usage.
```python
import time

class PerformanceMonitor:
    def __init__(self, spark):
        self.spark = spark
    
    def monitor_operation(self, operation_name):
        """Decorator to monitor operation performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                # Execute operation
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # Log metrics
                metrics = {
                    'operation': operation_name,
                    'duration_seconds': end_time - start_time,
                    'memory_delta_mb': end_memory - start_memory,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"Performance Metrics: {json.dumps(metrics, indent=2)}")
                
                return result
            return wrapper
        return decorator
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        sc = self.spark.sparkContext
        status = sc.statusTracker()
        # This is a simplified version
        return 0  # Implement actual memory tracking
    
    def analyze_plan(self, df):
        """Analyze query execution plan"""
        print("=== Physical Plan ===")
        df.explain("formatted")
        
        print("\n=== Cost Analysis ===")
        df.explain("cost")

# Usage
monitor = PerformanceMonitor(spark)

@monitor.monitor_operation("data_transformation")
def transform_data(df):
    return df.filter(col("age") > 18).groupBy("category").count()

result = transform_data(df)
```

---

## 14. Security and Compliance Patterns

### 14.1 Data Masking Pattern
Mask sensitive data for privacy.
```python
from pyspark.sql.functions import sha2, md5, regexp_replace, lit, when, length

class DataMasker:
    @staticmethod
    def mask_email(col_name):
        """Mask email addresses"""
        return regexp_replace(
            col(col_name),
            r"^([^@]{2})[^@]+(@.+)$",
            r"$1****$2"
        )
    
    @staticmethod
    def mask_phone(col_name):
        """Mask phone numbers"""
        return regexp_replace(
            col(col_name),
            r"(\d{3})(\d{3})(\d{4})",
            r"$1-***-$3"
        )
    
    @staticmethod
    def mask_credit_card(col_name):
        """Mask credit card numbers"""
        return regexp_replace(
            col(col_name),
            r"(\d{4})(\d{8})(\d{4})",
            r"$1-****-****-$3"
        )
    
    @staticmethod
    def hash_column(col_name, algorithm="sha256"):
        """Hash sensitive data"""
        return sha2(col(col_name), 256)
    
    @staticmethod
    def redact_column(col_name):
        """Completely redact sensitive data"""
        return lit("***REDACTED***")
    
    @staticmethod
    def mask_partial(col_name, visible_chars=4):
        """Show only last N characters"""
        return concat(
            lit("*" * 10),
            substring(col(col_name), -visible_chars, visible_chars)
        )

# Usage
df_masked = df.withColumn("email", DataMasker.mask_email("email")) \
    .withColumn("phone", DataMasker.mask_phone("phone")) \
    .withColumn("ssn", DataMasker.hash_column("ssn")) \
    .withColumn("notes", DataMasker.redact_column("notes"))
```

### 14.2 Row-Level Security Pattern
Filter data based on user permissions.
```python
class RowLevelSecurity:
    def __init__(self, spark):
        self.spark = spark
    
    def apply_security_filter(self, df, user_context):
        """Apply row-level security based on user context"""
        user_role = user_context.get('role')
        user_id = user_context.get('user_id')
        user_department = user_context.get('department')
        
        if user_role == 'admin':
            # Admins see everything
            return df
        elif user_role == 'manager':
            # Managers see their department
            return df.filter(col("department") == user_department)
        elif user_role == 'user':
            # Users see only their own data
            return df.filter(col("user_id") == user_id)
        else:
            # Unknown role sees nothing
            return df.filter(lit(False))
    
    def apply_column_security(self, df, user_context):
        """Apply column-level security"""
        user_role = user_context.get('role')
        
        # Define column access rules
        column_rules = {
            'admin': df.columns,
            'manager': [c for c in df.columns if c not in ['ssn', 'salary']],
            'user': [c for c in df.columns if c not in ['ssn', 'salary', 'performance_rating']]
        }
        
        allowed_columns = column_rules.get(user_role, ['id', 'name'])
        return df.select(*allowed_columns)

# Usage
security = RowLevelSecurity(spark)
user_context = {'role': 'manager', 'department': 'Engineering', 'user_id': '123'}

df_filtered = security.apply_security_filter(df, user_context)
df_secured = security.apply_column_security(df_filtered, user_context)
```

### 14.3 Audit Logging Pattern
Track data access and modifications.
```python
from pyspark.sql.functions import current_timestamp, lit, input_file_name

class AuditLogger:
    def __init__(self, spark, audit_table_path):
        self.spark = spark
        self.audit_table_path = audit_table_path
    
    def log_access(self, user_id, table_name, operation, row_count=None):
        """Log data access"""
        audit_record = self.spark.createDataFrame([{
            'timestamp': datetime.now(),
            'user_id': user_id,
            'table_name': table_name,
            'operation': operation,
            'row_count': row_count,
            'status': 'SUCCESS'
        }])
        
        audit_record.write.mode("append") \
            .partitionBy("date") \
            .parquet(self.audit_table_path)
    
    def log_modification(self, user_id, table_name, operation, records_affected):
        """Log data modifications"""
        audit_record = self.spark.createDataFrame([{
            'timestamp': datetime.now(),
            'user_id': user_id,
            'table_name': table_name,
            'operation': operation,
            'records_affected': records_affected,
            'status': 'SUCCESS'
        }])
        
        audit_record.write.mode("append") \
            .partitionBy("date") \
            .parquet(self.audit_table_path)
    
    def create_audit_trail(self, df, user_id, operation):
        """Add audit columns to dataframe"""
        return df.withColumn("audit_user", lit(user_id)) \
            .withColumn("audit_timestamp", current_timestamp()) \
            .withColumn("audit_operation", lit(operation))

# Usage
auditor = AuditLogger(spark, "/audit_logs")
auditor.log_access("user123", "customers", "READ", df.count())

df_with_audit = auditor.create_audit_trail(df, "user123", "UPDATE")
```

### 14.4 Data Encryption Pattern
Encrypt sensitive data at rest and in transit.
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, BinaryType
from cryptography.fernet import Fernet
import base64

class DataEncryption:
    def __init__(self, encryption_key):
        self.cipher = Fernet(encryption_key)
    
    def encrypt_udf(self):
        """UDF to encrypt data"""
        @udf(StringType())
        def encrypt(value):
            if value is None:
                return None
            encrypted = self.cipher.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        return encrypt
    
    def decrypt_udf(self):
        """UDF to decrypt data"""
        @udf(StringType())
        def decrypt(value):
            if value is None:
                return None
            try:
                decoded = base64.b64decode(value.encode())
                decrypted = self.cipher.decrypt(decoded)
                return decrypted.decode()
            except:
                return None
        return decrypt
    
    def encrypt_dataframe(self, df, columns_to_encrypt):
        """Encrypt specified columns"""
        encrypt_func = self.encrypt_udf()
        result_df = df
        for col_name in columns_to_encrypt:
            result_df = result_df.withColumn(
                col_name,
                encrypt_func(col(col_name))
            )
        return result_df
    
    def decrypt_dataframe(self, df, columns_to_decrypt):
        """Decrypt specified columns"""
        decrypt_func = self.decrypt_udf()
        result_df = df
        for col_name in columns_to_decrypt:
            result_df = result_df.withColumn(
                col_name,
                decrypt_func(col(col_name))
            )
        return result_df

# Usage
key = Fernet.generate_key()
encryptor = DataEncryption(key)

df_encrypted = encryptor.encrypt_dataframe(df, ["ssn", "credit_card"])
df_encrypted.write.parquet("/encrypted_data")

# Later, decrypt for authorized use
df_decrypted = encryptor.decrypt_dataframe(df_encrypted, ["ssn", "credit_card"])
```

---

## 15. Cost Optimization Patterns

### 15.1 Data Pruning Pattern
Minimize data scanned to reduce costs.
```python
class DataPruner:
    @staticmethod
    def prune_by_date_range(df, date_col, start_date, end_date):
        """Prune data by date range"""
        return df.filter(
            (col(date_col) >= start_date) & (col(date_col) <= end_date)
        )
    
    @staticmethod
    def prune_by_sampling(df, sample_rate=0.1):
        """Sample data for development/testing"""
        return df.sample(fraction=sample_rate, seed=42)
    
    @staticmethod
    def prune_columns(df, required_columns):
        """Select only necessary columns early"""
        return df.select(*required_columns)
    
    @staticmethod
    def prune_by_business_rules(df, conditions):
        """Apply business rules to reduce data volume"""
        result = df
        for condition in conditions:
            result = result.filter(condition)
        return result

# Usage
pruner = DataPruner()

# Prune early in the pipeline
df_pruned = pruner.prune_by_date_range(df, "transaction_date", "2024-01-01", "2024-12-31")
df_pruned = pruner.prune_columns(df_pruned, ["id", "amount", "date"])
df_pruned = pruner.prune_by_business_rules(df_pruned, [
    col("amount") > 0,
    col("status") == "completed"
])
```

### 15.2 Dynamic Resource Allocation Pattern
Optimize cluster resource usage.
```python
class ResourceOptimizer:
    def __init__(self, spark):
        self.spark = spark
    
    def configure_for_size(self, data_size_gb):
        """Configure Spark based on data size"""
        if data_size_gb < 10:
            # Small dataset
            self.spark.conf.set("spark.sql.shuffle.partitions", "50")
            self.spark.conf.set("spark.default.parallelism", "50")
        elif data_size_gb < 100:
            # Medium dataset
            self.spark.conf.set("spark.sql.shuffle.partitions", "200")
            self.spark.conf.set("spark.default.parallelism", "200")
        else:
            # Large dataset
            self.spark.conf.set("spark.sql.shuffle.partitions", "500")
            self.spark.conf.set("spark.default.parallelism", "500")
    
    def enable_dynamic_allocation(self):
        """Enable dynamic resource allocation"""
        self.spark.conf.set("spark.dynamicAllocation.enabled", "true")
        self.spark.conf.set("spark.dynamicAllocation.minExecutors", "1")
        self.spark.conf.set("spark.dynamicAllocation.maxExecutors", "100")
        self.spark.conf.set("spark.dynamicAllocation.initialExecutors", "10")
    
    def optimize_for_memory(self):
        """Optimize memory settings"""
        self.spark.conf.set("spark.memory.fraction", "0.8")
        self.spark.conf.set("spark.memory.storageFraction", "0.3")
        self.spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")

# Usage
optimizer = ResourceOptimizer(spark)
optimizer.enable_dynamic_allocation()
optimizer.configure_for_size(50)  # 50 GB dataset
```

### 15.3 Incremental Processing Pattern
Process only changed data to reduce costs.
```python
class IncrementalProcessor:
    def __init__(self, spark, checkpoint_path):
        self.spark = spark
        self.checkpoint_path = checkpoint_path
    
    def get_last_checkpoint(self):
        """Get last processed timestamp"""
        try:
            checkpoint_df = self.spark.read.parquet(self.checkpoint_path)
            last_checkpoint = checkpoint_df.agg({"timestamp": "max"}).collect()[0][0]
            return last_checkpoint
        except:
            return None
    
    def save_checkpoint(self, timestamp):
        """Save current checkpoint"""
        checkpoint_df = self.spark.createDataFrame([{
            'timestamp': timestamp,
            'processed_at': datetime.now()
        }])
        checkpoint_df.write.mode("append").parquet(self.checkpoint_path)
    
    def process_incremental(self, source_path, timestamp_col):
        """Process only new data since last checkpoint"""
        last_checkpoint = self.get_last_checkpoint()
        
        df = self.spark.read.parquet(source_path)
        
        if last_checkpoint:
            df_incremental = df.filter(col(timestamp_col) > last_checkpoint)
        else:
            df_incremental = df
        
        return df_incremental
    
    def merge_incremental(self, target_table, incremental_df, key_cols):
        """Merge incremental data into target"""
        from delta.tables import DeltaTable
        
        delta_table = DeltaTable.forName(self.spark, target_table)
        
        merge_condition = " AND ".join([
            f"target.{col} = source.{col}" for col in key_cols
        ])
        
        delta_table.alias("target").merge(
            incremental_df.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll() \
         .whenNotMatchedInsertAll() \
         .execute()

# Usage
processor = IncrementalProcessor(spark, "/checkpoints")
df_new = processor.process_incremental("/data/source", "updated_at")
processor.merge_incremental("target_table", df_new, ["id"])
processor.save_checkpoint(df_new.agg({"updated_at": "max"}).collect()[0][0])
```

---

## Summary

This comprehensive guide now covers **15 major categories with 80+ patterns**:

1. **Data Ingestion** - Schema evolution, incremental loads, multi-format readers
2. **Data Transformation** - Pipelines, UDFs, window functions, SCDs, pivots
3. **Data Quality** - Validation, null handling, outlier detection, profiling
4. **Performance** - Broadcast joins, partitioning, caching, bucketing, salting
5. **Storage** - Delta Lake, partitioning, compaction, data lake layers
6. **Error Handling** - Try-catch, dead letter queue, retry, circuit breaker
7. **Testing** - Unit tests, schema tests, integration tests, mocking
8. **Streaming** - Structured streaming, windowing, watermarking, stateful ops
9. **Machine Learning** - Feature engineering, cross-validation, model persistence
10. **Architecture** - Repository, factory, builder, strategy, observer patterns
11. **Advanced Transformations** - Complex types, self-joins, fuzzy matching, conditional aggregation
12. **Data Skew Handling** - Skew joins, adaptive salting, iterative broadcast
13. **Monitoring & Observability** - Metrics collection, lineage tracking, performance monitoring
14. **Security & Compliance** - Data masking, row-level security, audit logging, encryption
15. **Cost Optimization** - Data pruning, dynamic resource allocation, incremental processing

Each pattern addresses specific challenges in PySpark development and can be combined to build robust, scalable, secure, and cost-effective data applications.
