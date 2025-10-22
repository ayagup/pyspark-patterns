This is a comprehensive request! I'll generate a single PySpark program file, `pyspark_pattern_examples.py`, that demonstrates each pattern mentioned in your `pyspark.md` file.

Due to the length, this will be provided in multiple parts. Each section will correspond to a category in your markdown file.

```python pyspark_pattern_examples.py
import os
import shutil
import time # For streaming examples

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# --- Helper function to get SparkSession ---
def get_spark_session(app_name="PySparkPatternDemo", configs=None):
    builder = SparkSession.builder.appName(app_name)
    if configs:
        for key, value in configs.items():
            builder = builder.config(key, value)
    return builder.getOrCreate()

# --- Setup for I/O and Streaming examples ---
OUTPUT_BASE_DIR = "pyspark_pattern_output"
# Clean up previous runs for consistent examples
if os.path.exists(OUTPUT_BASE_DIR):
    shutil.rmtree(OUTPUT_BASE_DIR)
os.makedirs(OUTPUT_BASE_DIR)

# --- Initialize SparkSession (for patterns that need it) ---
# 1. Singleton SparkSession (handled by the get_spark_session helper)
#    We will reuse this `spark` object throughout the examples.
spark = get_spark_session(app_name="PySparkPatternDemoApp")

# 2. Configuring SparkSession (Runtime Properties)
#    Demonstrates setting a configuration at runtime.
print("\n--- I. Core SparkSession & Context Management ---")
print("1. Singleton SparkSession: Reusing 'spark' object.")
print("2. Configuring SparkSession (Runtime Properties): Setting shuffle partitions to 10")
spark.conf.set("spark.sql.shuffle.partitions", "10")
print(f"   Current spark.sql.shuffle.partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")

# 3. Accessing SparkContext
print("\n3. Accessing SparkContext:")
sc = spark.sparkContext
print(f"   SparkContextAppName: {sc.appName}")


print("\n--- II. Data Loading & Saving (I/O) ---")

# --- Dummy Data for I/O Examples ---
# Create some dummy data files
dummy_data_path = os.path.join(OUTPUT_BASE_DIR, "dummy_data")
os.makedirs(dummy_data_path, exist_ok=True)

# CSV
with open(os.path.join(dummy_data_path, "sample.csv"), "w") as f:
    f.write("id,name,age,city\n")
    f.write("1,Alice,30,New York\n")
    f.write("2,Bob,24,London\n")
    f.write("3,Charlie,35,Paris\n")

# Parquet
df_to_parquet = spark.createDataFrame([(1, "ProductA", 100), (2, "ProductB", 150)], ["id", "item", "price"])
df_to_parquet.write.mode("overwrite").parquet(os.path.join(dummy_data_path, "initial.parquet"))

# JSON
with open(os.path.join(dummy_data_path, "sample.json"), "w") as f:
    f.write('{"id": 1, "value": "A"}\n')
    f.write('{"id": 2, "value": "B"}\n')

# 1. Reading Various Data Formats (CSV, JSON, Parquet, JDBC - conceptual)
print("\n1. Reading Various Data Formats:")
df_csv = spark.read.csv(os.path.join(dummy_data_path, "sample.csv"), header=True, inferSchema=True)
print("   - Read CSV:")
df_csv.show(1)

df_json = spark.read.json(os.path.join(dummy_data_path, "sample.json"))
print("   - Read JSON:")
df_json.show(1)

df_parquet = spark.read.parquet(os.path.join(dummy_data_path, "initial.parquet"))
print("   - Read Parquet:")
df_parquet.show(1)

# JDBC (conceptual example, requires a database and driver)
print("   - JDBC (conceptual):")
print("     # df_jdbc = spark.read.format('jdbc') \\")
print("     #     .option('url', 'jdbc:postgresql://localhost:5432/mydb') \\")
print("     #     .option('dbtable', 'public.mytable') \\")
print("     #     .option('user', 'user') \\")
print("     #     .option('password', 'password') \\")
print("     #     .load()")
print("     # df_jdbc.show()")

# Delta/Iceberg (conceptual, requires connectors)
print("   - Delta/Iceberg (conceptual):")
print("     # df_delta = spark.read.format('delta').load('path/to/delta/table')")
print("     # df_iceberg = spark.read.format('iceberg').load('path/to/iceberg/table')")


# 2. Writing to Various Data Formats (modes: overwrite, append, ignore, errorIfExists)
print("\n2. Writing to Various Data Formats:")
df_write_example = spark.createDataFrame([(10, "DataX", 2023), (11, "DataY", 2024)], ["id", "data", "year"])

# Overwrite mode
output_parquet_overwrite = os.path.join(OUTPUT_BASE_DIR, "output.parquet")
df_write_example.write.mode("overwrite").parquet(output_parquet_overwrite)
print(f"   - Wrote to {output_parquet_overwrite} (overwrite mode)")

# Append mode
output_csv_append = os.path.join(OUTPUT_BASE_DIR, "output.csv")
df_write_example.write.mode("append").csv(output_csv_append, header=True)
print(f"   - Wrote to {output_csv_append} (append mode)")

# 3. Schema Provisioning/Enforcement on Read
print("\n3. Schema Provisioning/Enforcement on Read:")
schema = StructType([
    StructField("id_str", StringType(), True),
    StructField("name_str", StringType(), True),
    StructField("age_int", IntegerType(), True),
    StructField("city_str", StringType(), True)
])
df_csv_with_schema = spark.read.csv(os.path.join(dummy_data_path, "sample.csv"), header=True, schema=schema)
print("   - Read CSV with explicit schema (age_int column type):")
df_csv_with_schema.printSchema()
df_csv_with_schema.show(1)

# 4. Partitioned Writes
print("\n4. Partitioned Writes:")
df_partitioned_write = spark.createDataFrame([
    (1, "A", "2023-01-01"), (2, "B", "2023-01-01"),
    (3, "C", "2023-01-02"), (4, "D", "2023-01-02")
], ["id", "value", "date"])
output_partitioned_parquet = os.path.join(OUTPUT_BASE_DIR, "partitioned_data")
df_partitioned_write.write.mode("overwrite").partitionBy("date").parquet(output_partitioned_parquet)
print(f"   - Wrote partitioned data to {output_partitioned_parquet} (by date)")
# Verify partitions created
print(f"     Subdirectories: {os.listdir(output_partitioned_parquet)}")

# 5. Bucketing on Write (requires saving as a table)
print("\n5. Bucketing on Write:")
# For bucketing, Spark needs to manage the table. We'll use a temporary path for this example.
spark.sql("CREATE DATABASE IF NOT EXISTS dev_db")
spark.sql("USE dev_db")
df_bucketing_example = spark.createDataFrame([
    (1, "TypeA", 100), (2, "TypeB", 200), (3, "TypeA", 150), (4, "TypeC", 300),
    (5, "TypeB", 250), (6, "TypeA", 120)
], ["id", "category", "amount"])
temp_bucketed_table_path = os.path.join(OUTPUT_BASE_DIR, "bucketed_table")
df_bucketing_example.write \
    .option("path", temp_bucketed_table_path) \
    .mode("overwrite") \
    .bucketBy(4, "category") \
    .sortBy("category") \
    .saveAsTable("my_bucketed_table")
print(f"   - Wrote bucketed table 'my_bucketed_table' to {temp_bucketed_table_path} (4 buckets by category)")
print(f"     Bucketed files should be visible in {temp_bucketed_table_path}")
spark.sql("DROP TABLE IF EXISTS my_bucketed_table") # Clean up temp table
spark.sql("DROP DATABASE IF EXISTS dev_db CASCADE") # Clean up temp db

# 6. Reading/Writing from Cloud Storage (S3, ADLS, GCS - conceptual)
print("\n6. Reading/Writing from Cloud Storage (conceptual):")
print("   # S3: spark.read.parquet('s3a://your-bucket/path/to/data')")
print("   # ADLS: spark.read.csv('abfss://filesystem@account.dfs.core.windows.net/path/to/data')")
print("   # GCS: spark.read.json('gs://your-bucket/path/to/data')")

# 7. Reading from Streaming Sources (FileStream, Rate)
print("\n7. Reading from Streaming Sources (FileStream, Rate - conceptual for Kafka):")
streaming_input_path = os.path.join(OUTPUT_BASE_DIR, "streaming_input")
os.makedirs(streaming_input_path, exist_ok=True)

# FileStream (monitors directory for new files)
print("   - FileStream (conceptual):")
print(f"     # df_stream_file = spark.readStream.format('csv') \\")
print(f"     #     .option('header', 'true') \\")
print(f"     #     .schema(df_csv.schema) \\")
print(f"     #     .load('{streaming_input_path}')")
print("     # df_stream_file.isStreaming # True")

# Rate source (for testing)
print("   - Rate Source:")
df_stream_rate = spark.readStream.format("rate").option("rowsPerSecond", 1).load()
print("     # df_stream_rate.isStreaming # True")
# df_stream_rate.printSchema() # Will show 'timestamp', 'value'

# Kafka (conceptual, requires kafka broker and spark-sql-kafka-0-10_2.12 jar)
print("   - Kafka (conceptual):")
print("     # df_stream_kafka = spark.readStream.format('kafka') \\")
print("     #     .option('kafka.bootstrap.servers', 'host1:port1,host2:port2') \\")
print("     #     .option('subscribe', 'topic1') \\")
print("     #     .load()")


# 8. Writing to Streaming Sinks (Console, FileStream, Kafka, ForeachBatch)
print("\n8. Writing to Streaming Sinks (Console, ForeachBatch):")
# We'll use the 'rate' stream for these examples to avoid external dependencies.

# Console sink (for demonstration)
query_console = df_stream_rate.writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime="5 seconds") \
    .option("truncate", "false") \
    .start()
print("   - Console Sink: Started a stream to console. Will run for 5 seconds.")
time.sleep(5) # Let it run for a bit
query_console.stop()
print("     Console Sink: Stopped.")

# FileStream sink (writing to a directory)
streaming_output_path = os.path.join(OUTPUT_BASE_DIR, "streaming_output_files")
streaming_checkpoint_path_file = os.path.join(OUTPUT_BASE_DIR, "checkpoint_file")

query_file_sink = df_stream_rate.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", streaming_output_path) \
    .option("checkpointLocation", streaming_checkpoint_path_file) \
    .trigger(processingTime="5 seconds") \
    .start()
print(f"   - FileStream Sink: Started writing to {streaming_output_path}. Will run for 5 seconds.")
time.sleep(5)
query_file_sink.stop()
print("     FileStream Sink: Stopped. Check for parquet files in the output directory.")
print(f"     Files in {streaming_output_path}: {os.listdir(streaming_output_path)}")


# ForeachBatch sink (for custom logic)
streaming_checkpoint_path_foreach = os.path.join(OUTPUT_BASE_DIR, "checkpoint_foreach")

def foreach_batch_function(df_batch, batch_id):
    print(f"   - ForeachBatch Sink: Processing batch {batch_id} with {df_batch.count()} rows.")
    # Example: Write to a database, call an API, perform custom logic
    # df_batch.write.format("jdbc")...
    # df_batch.show()

query_foreach_batch = df_stream_rate.writeStream \
    .outputMode("append") \
    .foreachBatch(foreach_batch_function) \
    .option("checkpointLocation", streaming_checkpoint_path_foreach) \
    .trigger(processingTime="5 seconds") \
    .start()
print(f"   - ForeachBatch Sink: Started custom batch processing. Will run for 5 seconds.")
time.sleep(5)
query_foreach_batch.stop()
print("     ForeachBatch Sink: Stopped.")

# Kafka Sink (conceptual)
print("   - Kafka Sink (conceptual):")
print("     # query_kafka = df_stream.selectExpr('CAST(key AS STRING)', 'CAST(value AS STRING)') \\")
print("     #     .writeStream \\")
print("     #     .format('kafka') \\")
print("     #     .option('kafka.bootstrap.servers', 'host1:port1') \\")
print("     #     .option('topic', 'output_topic') \\")
print("     #     .option('checkpointLocation', 'path/to/checkpoint') \\")
print("     #     .start()")


print("\n--- III. DataFrame Transformations (Basic) ---")

# --- Sample DataFrame for basic transformations ---
data_basic = [
    (1, "Alice", 25, "New York", 50000, "HR"),
    (2, "Bob", 30, "London", 60000, "IT"),
    (3, "Charlie", 25, "New York", 55000, "IT"),
    (4, "David", None, "Paris", 70000, "HR"),
    (5, "Eve", 28, "London", 62000, "HR"),
    (6, "Frank", 35, "Berlin", 75000, "IT"),
    (7, "Grace", 25, "New York", 50000, "HR"),
    (8, "Heidi", 30, "London", 60000, None),
]
schema_basic = ["id", "name", "age", "city", "salary", "department"]
df_basic = spark.createDataFrame(data_basic, schema_basic)
print("   - Original DataFrame (df_basic):")
df_basic.show()
df_basic.printSchema()

# 1. Column Selection & Renaming (`select`, `selectExpr`, `withColumnRenamed`)
print("\n1. Column Selection & Renaming:")
df_selected = df_basic.select("id", col("name").alias("full_name"), "city")
print("   - Using select and alias:")
df_selected.show()

df_select_expr = df_basic.selectExpr("id", "name as employee_name", "salary * 1.1 as increased_salary")
print("   - Using selectExpr:")
df_select_expr.show()

df_renamed = df_basic.withColumnRenamed("name", "employee_name_renamed")
print("   - Using withColumnRenamed:")
df_renamed.show(1)

# 2. Filtering Data (`filter`, `where`)
print("\n2. Filtering Data:")
df_filtered_age = df_basic.filter(df_basic["age"] > 28)
print("   - Filter by age > 28:")
df_filtered_age.show()

df_filtered_complex = df_basic.where((col("city") == "New York") & (col("department") == "HR"))
print("   - Filter by city 'New York' AND department 'HR':")
df_filtered_complex.show()

# 3. Adding/Modifying Columns (`withColumn`)
print("\n3. Adding/Modifying Columns:")
df_with_status = df_basic.withColumn("salary_usd", col("salary") / 1.2) \
                         .withColumn("is_senior", when(col("age") >= 30, True).otherwise(False))
print("   - Added 'salary_usd' and 'is_senior' columns:")
df_with_status.show()

# 4. Dropping Columns (`drop`, `dropDuplicates`)
print("\n4. Dropping Columns:")
df_dropped = df_basic.drop("salary", "department")
print("   - Dropped 'salary' and 'department' columns:")
df_dropped.show()

# Create a DF with duplicates for dropDuplicates
data_duplicates = [("A", 10), ("B", 20), ("A", 10), ("C", 30)]
df_duplicates = spark.createDataFrame(data_duplicates, ["key", "value"])
print("   - Original DataFrame with duplicates:")
df_duplicates.show()
df_distinct = df_duplicates.dropDuplicates()
print("   - After dropDuplicates():")
df_distinct.show()
df_distinct_subset = df_duplicates.dropDuplicates(["key"]) # Drop duplicates based on 'key' column
print("   - After dropDuplicates(['key']):")
df_distinct_subset.show()


# 5. Handling Missing Values (`na.drop`, `na.fill`, `na.replace`)
print("\n5. Handling Missing Values:")
# Original df_basic has a null age and null department
print("   - Original df_basic with nulls:")
df_basic.show()

df_na_dropped = df_basic.na.drop(subset=["age", "department"]) # Drop rows where age OR department is null
print("   - After na.drop(subset=['age', 'department']):")
df_na_dropped.show()

df_na_filled = df_basic.na.fill({"age": 0, "department": "Unknown"})
print("   - After na.fill({'age': 0, 'department': 'Unknown'}):")
df_na_filled.show()

df_na_replaced = df_basic.na.replace("London", "UK_Capital", "city") # Replace specific value
print("   - After na.replace('London', 'UK_Capital', 'city'):")
df_na_replaced.show()

# 6. Sorting Data (`orderBy`, `sort`)
print("\n6. Sorting Data:")
df_sorted = df_basic.orderBy(col("department").asc_nulls_first(), col("age").desc())
print("   - Sorted by department ASC (nulls first), then age DESC:")
df_sorted.show()

df_sort_alias = df_basic.sort("city", "name") # 'sort' is an alias for 'orderBy'
print("   - Sorted by city ASC, then name ASC:")
df_sort_alias.show

```python pyspark_pattern_examples.py
# ... (previous code)

# 7. Aggregations (`groupBy`, `agg`)
print("\n7. Aggregations:")
df_agg_by_dept = df_basic.groupBy("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("average_salary"),
    max("age").alias("max_age")
)
print("   - Grouped by department, aggregated count, avg salary, max age:")
df_agg_by_dept.show()

# 8. Joining DataFrames (Inner, Outer, Left, Right, Semi, Anti)
print("\n8. Joining DataFrames:")
df_departments = spark.createDataFrame([
    ("HR", "Human Resources"),
    ("IT", "Information Technology"),
    ("Finance", "Finance Department")
], ["dept_code", "dept_name"])

df_joined_inner = df_basic.join(df_departments, df_basic["department"] == df_departments["dept_code"], "inner")
print("   - Inner Join (employees with known departments):")
df_joined_inner.show()

df_joined_left_outer = df_basic.join(df_departments, df_basic["department"] == df_departments["dept_code"], "left_outer")
print("   - Left Outer Join (all employees, matching department info or null):")
df_joined_left_outer.show()

df_joined_right_outer = df_basic.join(df_departments, df_basic["department"] == df_departments["dept_code"], "right_outer")
print("   - Right Outer Join (all departments, matching employee info or null):")
df_joined_right_outer.show()

df_joined_left_semi = df_basic.join(df_departments, df_basic["department"] == df_departments["dept_code"], "left_semi")
print("   - Left Semi Join (employees that have a matching department):")
df_joined_left_semi.show()

df_joined_left_anti = df_basic.join(df_departments, df_basic["department"] == df_departments["dept_code"], "left_anti")
print("   - Left Anti Join (employees that DO NOT have a matching department):")
df_joined_left_anti.show()


# 9. Unioning DataFrames (`union`, `unionByName`)
print("\n9. Unioning DataFrames:")
df_more_employees = spark.createDataFrame([
    (9, "Mia", 29, "Madrid", 58000, "IT"),
    (10, "Nolan", 40, "Sydney", 80000, "Finance")
], schema_basic)
print("   - Original df_basic:")
df_basic.show()
print("   - Additional Employees:")
df_more_employees.show()

df_union = df_basic.union(df_more_employees)
print("   - After union (requires same schema order):")
df_union.show()

# Demonstrate unionByName (allowing different column order, filling missing with null)
df_more_employees_reordered = spark.createDataFrame([
    ("Mia", 29, 9, "Madrid", 58000, "IT"), # id moved
    ("Nolan", 40, 10, "Sydney", 80000, "Finance")
], ["name", "age", "id", "city", "salary", "department"]) # schema order changed

df_union_by_name = df_basic.unionByName(df_more_employees_reordered, allowMissingColumns=True)
print("   - After unionByName (can handle different column order):")
df_union_by_name.show()


# 10. Type Casting (`cast`)
print("\n10. Type Casting:")
df_casted_age = df_basic.withColumn("age_str", col("age").cast(StringType())) \
                        .withColumn("id_long", col("id").cast(LongType()))
print("   - Casted 'age' to StringType and 'id' to LongType:")
df_casted_age.select("age", "age_str", "id", "id_long").printSchema()
df_casted_age.select("age", "age_str", "id", "id_long").show(2)


# 11. Sampling Data (`sample`)
print("\n11. Sampling Data:")
# Sample 50% of the data with replacement
df_sample_with_replacement = df_basic.sample(withReplacement=True, fraction=0.5, seed=42)
print("   - Sampled 50% with replacement:")
print(f"     Original count: {df_basic.count()}, Sampled count: {df_sample_with_replacement.count()}")
df_sample_with_replacement.show()

# Sample 50% of the data without replacement
df_sample_no_replacement = df_basic.sample(withReplacement=False, fraction=0.5, seed=42)
print("   - Sampled 50% without replacement:")
print(f"     Original count: {df_basic.count()}, Sampled count: {df_sample_no_replacement.count()}")
df_sample_no_replacement.show()

# 12. Repartitioning / Coalescing Data (`repartition`, `coalesce`)
print("\n12. Repartitioning / Coalescing Data:")
print(f"   - Original df_basic partitions: {df_basic.rdd.getNumPartitions()}")

df_repartitioned = df_basic.repartition(5) # Shuffle to 5 partitions
print(f"   - After repartition(5) partitions: {df_repartitioned.rdd.getNumPartitions()}")
df_repartitioned.write.mode("overwrite").parquet(os.path.join(OUTPUT_BASE_DIR, "repartitioned_data"))
print(f"     (Check {os.path.join(OUTPUT_BASE_DIR, 'repartitioned_data')} for 5 part files)")


df_coalesced = df_basic.coalesce(2) # Reduce to 2 partitions (less shuffling than repartition)
print(f"   - After coalesce(2) partitions: {df_coalesced.rdd.getNumPartitions()}")
df_coalesced.write.mode("overwrite").parquet(os.path.join(OUTPUT_BASE_DIR, "coalesced_data"))
print(f"     (Check {os.path.join(OUTPUT_BASE_DIR, 'coalesced_data')} for 2 part files)")


print("\n--- IV. DataFrame Transformations (Advanced) ---")

# --- Sample DataFrame for advanced transformations ---
data_advanced = [
    (1, "A", "Electronics", 100, "2023-01-01"),
    (1, "B", "Electronics", 150, "2023-01-02"),
    (1, "C", "Clothing", 50, "2023-01-03"),
    (2, "D", "Electronics", 200, "2023-01-01"),
    (2, "E", "Books", 30, "2023-01-04"),
    (3, "F", "Electronics", 120, "2023-01-01"),
    (3, "G", "Electronics", 80, "2023-01-05"),
    (4, "H", "Books", 40, "2023-01-02"),
]
schema_advanced = ["customer_id", "item", "category", "price", "order_date"]
df_advanced = spark.createDataFrame(data_advanced, schema_advanced)
print("   - Original DataFrame (df_advanced):")
df_advanced.show()

# 1. Window Functions (Ranking, Lag/Lead, Moving Averages, Cumulative Sums)
print("\n1. Window Functions:")

# Ranking - Rank items by price within each customer
window_spec_rank = Window.partitionBy("customer_id").orderBy(desc("price"))
df_ranked = df_advanced.withColumn("rank_by_price", rank().over(window_spec_rank))
print("   - Rank items by price within each customer:")
df_ranked.show()

# Lag/Lead - Previous day's price for each customer
window_spec_lag = Window.partitionBy("customer_id").orderBy("order_date")
df_lag = df_advanced.withColumn("previous_price", lag("price", 1).over(window_spec_lag))
print("   - Previous price for each customer by order date:")
df_lag.show()

# Cumulative Sum - Running total of prices per category
window_spec_cumulative = Window.partitionBy("category").orderBy("order_date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
df_cumulative = df_advanced.withColumn("cumulative_price", sum("price").over(window_spec_cumulative))
print("   - Cumulative price per category by order date:")
df_cumulative.show()

# 2. Exploding Arrays (`explode`, `posexplode`)
print("\n2. Exploding Arrays:")
data_explode = [
    (1, ["apple", "banana", "cherry"]),
    (2, ["date", "elderberry"])
]
df_array_col = spark.createDataFrame(data_explode, ["id", "fruits"])
print("   - Original DataFrame with array column:")
df_array_col.show()

df_exploded = df_array_col.withColumn("fruit", explode(col("fruits")))
print("   - After explode():")
df_exploded.show()

df_posexploded = df_array_col.withColumn("exploded", posexplode(col("fruits"))) \
                             .select("id", col("exploded.pos").alias("index"), col("exploded.col").alias("fruit_name"))
print("   - After posexplode():")
df_posexploded.show()

# 3. Pivoting / Unpivoting (`pivot`, `stack`)
print("\n3. Pivoting / Unpivoting:")

# Pivoting Example
print("   - Original df_advanced (for pivoting):")
df_advanced.show()
df_pivoted = df_advanced.groupBy("customer_id").pivot("category", ["Electronics", "Clothing", "Books"]).sum("price")
print("   - Pivoted by customer_id, category (sum of price):")
df_pivoted.show()

# Unpivoting (using stack)
print("   - Original pivoted DataFrame (for unpivoting):")
df_pivoted.show()
df_unpivoted = df_pivoted.select("customer_id", expr("stack(3, 'Electronics', Electronics, 'Clothing', Clothing, 'Books', Books) as (category, total_price)"))
print("   - Unpivoted back to long format:")
df_unpivoted.show()

# 4. Higher-Order Functions on Arrays and Maps (`transform`, `filter`, `exists`, `aggregate`, `map_arrays`)
print("\n4. Higher-Order Functions on Arrays and Maps:")
data_hof = [
    (1, [10, 20, 30], {"tag1": "value1", "tag2": "value2"}),
    (2, [5, 15], {"tag3": "value3"})
]
schema_hof = StructType([
    StructField("id", IntegerType(), False),
    StructField("values", ArrayType(IntegerType()), False),
    StructField("tags", MapType(StringType(), StringType()), False)
])
df_hof = spark.createDataFrame(data_hof, schema=schema_hof)
print("   - Original DataFrame with array and map columns:")
df_hof.show()
df_hof.printSchema()

# transform - double each value in the array
df_transformed_array = df_hof.withColumn("doubled_values", transform(col("values"), lambda x: x * 2))
print("   - After transform (doubled array values):")
df_transformed_array.show()

# filter - keep only values > 15
df_filtered_array = df_hof.withColumn("filtered_values", filter(col("values"), lambda x: x > 15))
print("   - After filter (values > 15):")
df_filtered_array.show()

# exists - check if any value in array > 25
df_exists_array = df_hof.withColumn("has_large_value", exists(col("values"), lambda x: x > 25))
print("   - After exists (any value > 25):")
df_exists_array.show()

# aggregate - sum of array elements
df_aggregated_array = df_hof.withColumn("sum_values", aggregate(col("values"), lit(0), lambda acc, x: acc + x))
print("   - After aggregate (sum of array elements):")
df_aggregated_array.show()

# map_from_arrays (not directly map_arrays, but a common HOF use case to create map from two arrays)
# Let's create an example where we have two arrays for keys and values
df_map_create = spark.createDataFrame([(1, ['k1', 'k2'], ['v1', 'v2'])], ['id', 'keys', 'vals'])
df_with_map = df_map_create.withColumn('created_map', map_from_arrays('keys', 'vals'))
print("   - After map_from_arrays:")
df_with_map.show()
df_with_map.printSchema()


# 5. Using SQL Expressions within DataFrames (`selectExpr`, `where`)
print("\n5. Using SQL Expressions within DataFrames:")
df_sql_expr_select = df_advanced.selectExpr("customer_id", "upper(item) as upper_item", "price * 0.9 as discounted_price")
print("   - Using selectExpr for uppercase and discounted price:")
df_sql_expr_select.show()

df_sql_expr_where = df_advanced.where("price > 100 AND category = 'Electronics'")
print("   - Using SQL expression in where clause:")
df_sql_expr_where.show()

# Using spark.sql for full SQL queries
df_advanced.createOrReplaceTempView("sales_data")
df_spark_sql = spark.sql("SELECT customer_id, SUM(price) as total_spent FROM sales_data GROUP BY customer_id HAVING SUM(price) > 200")
print("   - Using spark.sql for aggregate query:")
df_spark_sql.show()


# 6. Recursive CTEs (Common Table Expressions)
print("\n6. Recursive CTEs (Spark SQL only):")
# Create a dummy employee hierarchy table
spark.createDataFrame([
    (1, "Alice", None),
    (2, "Bob", 1),
    (3, "Charlie", 1),
    (4, "David", 2),
    (5, "Eve", 2),
    (6, "Frank", 3)
], ["id", "name", "manager_id"]).createOrReplaceTempView("employees_hierarchy")

print("   - Example: Find all subordinates for each employee using a recursive CTE:")
recursive_cte_query = """
    WITH RECURSIVE subordinates AS (
        SELECT id, name, manager_id, 0 as level
        FROM employees_hierarchy
        WHERE manager_id IS NULL -- The CEO/top-level
        UNION ALL
        SELECT e.id, e.name, e.manager_id, s.level + 1
        FROM employees_hierarchy e
        INNER JOIN subordinates s ON e.manager_id = s.id
    )
    SELECT * FROM subordinates ORDER BY level, id
"""
spark.sql(recursive_cte_query).show()


# 7. FoldLeft/Reduce operations (on DataFrames - conceptual, often UDAF or RDD for true foldLeft)
print("\n7. FoldLeft/Reduce operations (conceptual):")
# While direct foldLeft like Scala's on DF isn't common, we can simulate or use functools.reduce
# Example: Summing multiple columns
from functools import reduce
df_multiple_cols = spark.createDataFrame([(1, 10, 20, 30), (2, 5, 15, 25)], ["id", "col1", "col2", "col3"])
print("   - Original DataFrame for sum reduction:")
df_multiple_cols.show()

columns_to_sum = ["col1", "col2", "col3"]
df_sum_reduced = df_multiple_cols.withColumn("total_sum", reduce(lambda a, b: a + b, [col(c) for c in columns_to_sum]))
print("   - Sum of multiple columns using functools.reduce (conceptually foldLeft):")
df_sum_reduced.show()


# 8. Approximate Quantiles
print("\n8. Approximate Quantiles:")
# Using df_advanced 'price' column
quantiles_prices = df_advanced.approxQuantile("price", [0.25, 0.5, 0.75], 0.01) # 25th, 50th (median), 75th percentile with 1% error
print(f"   - Approximate Quantiles for 'price' (Q1, Median, Q3): {quantiles_prices}")
print(f"     25th Percentile (Q1): {quantiles_prices[0]}")
print(f"     Median (Q2): {quantiles_prices[1]}")
print(f"     75th Percentile (Q3): {quantiles_prices[2]}")


print("\n--- V. Schema & Type Handling ---")

# --- Sample data ---
data_schema = [
    (1, "ProductA", 10.5, "true", [1,2], {"key1": "val1"}),
    (2, "ProductB", 20.0, "false", [3], {"key2": "val2"}),
    (3, "ProductC", 15.7, "true", [], {})
]

# 1. Explicit Schema Definition (StructType)
print("\n1. Explicit Schema Definition:")
explicit_schema = StructType([
    StructField("item_id", IntegerType(), False),
    StructField("item_name", StringType(), True),
    StructField("price", DecimalType(10, 2), True), # Using DecimalType for precision
    StructField("is_available", BooleanType(), True),
    StructField("tags", ArrayType(IntegerType()), True),
    StructField("properties", MapType(StringType(), StringType()), True)
])
df_explicit_schema = spark.createDataFrame(data_schema, schema=explicit_schema)
print("   - DataFrame created with explicit schema:")
df_explicit_schema.printSchema()
df_explicit_schema.show(2)


# 2. Schema Evolution (mergeSchema)
print("\n2. Schema Evolution (mergeSchema):")
# Initial data with some schema
df1_schema_evolve = spark.createDataFrame([
    (1, "OldItem", 10.0),
    (2, "AnotherOld", 20.0)
], ["id", "name", "value"])

output_evolve_path = os.path.join(OUTPUT_BASE_DIR, "schema_evolve_data")
df1_schema_evolve.write.mode("overwrite").parquet(output_evolve_path)

# New data with an additional column
df2_schema_evolve = spark.createDataFrame([
    (3, "NewItem", 15.0, "CategoryA"),
    (4, "YetAnother", 25.0, "CategoryB")
], ["id", "name", "value", "category"])

# Write new data to the same path using append and mergeSchema
df2_schema_evolve.write.mode("append").option("mergeSchema", "true").parquet(output_evolve_path)

# Read the merged data
df_merged_schema = spark.read.option("mergeSchema", "true").parquet(output_evolve_path)
print("   - Schema evolved by merging (new 'category' column added):")
df_merged_schema.printSchema()
df_merged_schema.show()


# 3. Handling Complex Types (ArrayType, MapType, StructType)
print("\n3. Handling Complex Types:")
# df_explicit_schema already demonstrates ArrayType and MapType.
# Let's create one with StructType.
struct_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("metadata", StructType([
        StructField("author", StringType(), True),
        StructField("version", StringType(), True)
    ]), True)
])
df_struct = spark.createDataFrame([
    (1, Row(author="John Doe", version="1.0")),
    (2, Row(author="Jane Smith", version="1.1"))
], schema=struct_schema)
print("   - DataFrame with StructType column:")
df_struct.printSchema()
df_struct.show()

# Accessing elements of complex types
df_complex_access = df_struct.withColumn("author_name", col("metadata.author")) \
                             .withColumn("first_tag", col("tags")[0]) # from df_explicit_schema if available
print("   - Accessing elements within complex types (e.g., metadata.author, tags[0]):")
# To demonstrate 'tags[0]', we need to ensure df_explicit_schema is used if 'tags' is not in df_struct
if 'tags' in df_explicit_schema.columns:
    df_complex_access_tags = df_explicit_schema.withColumn("first_tag", col("tags")[0])
    df_complex_access_tags.select("item_id", "tags", "first_tag").show()
else:
    print("     (Skipped 'tags[0]' as 'tags' column not present in current example DF)")


# 4. Schema Inference
print("\n4. Schema Inference:")
# This is the default behavior when you don't provide an explicit schema.
# Demonstrated in many earlier examples (e.g., reading CSV/JSON without schema option).
print("   - Schema inference in action (e.g., when reading sample.csv without explicit schema):")
df_inferred_schema = spark.read.csv(os.path.join(dummy_data_path, "sample.csv"), header=True, inferSchema=True)
df_inferred_schema.printSchema()
print("     (Note the inferred data types, e.g., 'id' and 'age' as IntegerType)")


print("\n--- VI. Performance Optimization ---")

# --- Sample DataFrame for performance optimization examples ---
df_perf_opt = spark.range(1000000).withColumn("group", (col("id") % 100)) \
                   .withColumn("value", rand() * 100)
df_perf_opt = df_perf_opt.repartition(100, "group") # Start with some partitions
print("   - Original DataFrame (df_perf_opt) for optimization examples:")
print(f"     Count: {df_perf_opt.count()}, Partitions: {df_perf_opt.rdd.getNumPartitions()}")


# 1. Caching/Persisting DataFrames (`cache`, `persist`)
print("\n1. Caching/Persisting DataFrames:")
df_cached = df_perf_opt.filter(col("value") > 50).cache()
print("   - Cached df_cached. First action (count) triggers caching:")
print(f"     Cached count: {df_cached.count()}") # This will trigger the computation and cache
print("   - Subsequent actions are faster due to cache:")
start_time = time.time()
df_cached.groupBy("group").avg("value").count()
end_time = time.time()
print(f"     Time for cached operation: {end_time - start_time:.4f} seconds")

df_cached.unpersist() # Clean up cache
print("   - Unpersisted df_cached.")


# 2. Controlling Shuffle Partitions
print("\n2. Controlling Shuffle Partitions:")
# This was already demonstrated in SparkSession configuration and repartitioning.
# To reiterate, setting this configuration:
spark.conf.set("spark.sql.shuffle.partitions", "20") # Reduced for example
print(f"   - Set spark.sql.shuffle.partitions to {spark.conf.get('spark.sql.shuffle.partitions')}")
df_agg_shuffle = df_perf_opt.groupBy("group").count()
print(f"   - Aggregation will use {spark.conf.get('spark.sql.shuffle.partitions')} partitions for shuffle output.")
# df_agg_shuffle.explain() # Uncomment to see the exchange details


# 3. Broadcasting Small DataFrames/Variables
print("\n3. Broadcasting Small DataFrames/Variables:")
df_small = spark.createDataFrame([(i, f"Name_{i}") for i in range(10)], ["id", "name"])
df_large = spark.range(1000000).withColumnRenamed("id", "small_id")

# Explicit broadcast hint
df_broadcast_join = df_large.join(broadcast(df_small), df_large["small_id"] == df_small["id"])
print("   - Broadcast Join (explicit hint):")
print(f"     Joined count: {df_broadcast_join.count()}")
df_broadcast_join.explain(mode="formatted") # Look for "BroadcastHashJoin" in the plan

# Spark can also infer broadcasting if `spark.sql.autoBroadcastJoinThreshold` is met.
# To disable auto-broadcast for demonstration of manual hint:
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")


# 4. Optimizing Joins (Broadcast Join Hint, Join Reordering)
print("\n4. Optimizing Joins:")
print("   - Broadcast Join Hint: Already shown above.")
print("   - Join Reordering: Spark's CBO (Cost-Based Optimizer) handles this. Enable CBO for best results.")
# Example of setting CBO stats gathering for a table (conceptual as it requires a table)
print("     # spark.sql('ANALYZE TABLE my_table COMPUTE STATISTICS FOR COLUMNS col1, col2')")


# 5. Predicate Pushdown
print("\n5. Predicate Pushdown:")
# Spark pushes filters down to the data source (e.g., Parquet, JDBC) to reduce data scanned.
df_pushed_down = df_perf_opt.filter(col("group") == 50)
print("   - Predicate Pushdown example (filtering before reading data):")
df_pushed_down.write.mode("overwrite").parquet(os.path.join(OUTPUT_BASE_DIR, "pred_pushdown_test"))
# If you then read this filtered data, Spark tries to push the filter to the Parquet reader.
df_read_pushed = spark.read.parquet(os.path.join(OUTPUT_BASE_DIR, "pred_pushdown_test")).filter(col("value") > 90)
print("     Reading filtered data with another filter:")
df_read_pushed.explain(mode="formatted") # Look for "PushedFilters" in ParquetScan

# 6. Columnar Pruning
print("\n6. Columnar Pruning:")
# Spark reads only the columns actually needed for the query from the data source.
df_pruned = df_perf_opt.select("id", "value")
print("   - Columnar Pruning example (selecting only 'id' and 'value'):")
df_pruned.explain(mode="formatted") # Look for "PushedProjections" or "ReadSchema" showing only selected columns


# 7. Data Partitioning Strategies (Manual Partitioning, Bucketing)
print("\n7. Data Partitioning Strategies:")
# Manual Partitioning: Demonstrated in II.4 (Partitioned Writes) and III.12 (Repartitioning)
print("   - Manual Partitioning (e.g., `repartition()`, writing with `partitionBy()`):")
print("     Refer to examples II.4 and III.12.")

# Bucketing: Demonstrated in II.5 (Bucketing on Write)
print("   - Bucketing (e.g., `bucketBy().saveAsTable()`):")
print("     Refer to example II.5.")

# 8. Using `explain()` for Query Plan Analysis
print("\n8. Using `explain()` for Query Plan Analysis:")
print("   - df_agg_shuffle.explain(mode='extended'):")
df_agg_shuffle.explain(mode="extended")
print("     (Check the output above for detailed physical and logical plans)")

# 9. Spark UI Monitoring for Performance Bottlenecks
print("\n9. Spark UI Monitoring (Conceptual):")
print("   - When running locally, typically http://localhost:4040. On cluster, provided by cluster manager.")
print("   - Look for long-running stages, skewed tasks, garbage collection pauses, disk I/O.")

# 10. Memory Management (Storage Levels)
print("\n10. Memory Management (Storage Levels):")
# `cache()` uses MEMORY_AND_DISK by default. `persist()` allows specifying.
from pyspark import StorageLevel
df_persisted_disk = df_perf_opt.persist(StorageLevel.DISK_ONLY)
print(f"   - Persisted df_perf_opt to DISK_ONLY. Count: {df_persisted_disk.count()}")
df_persisted_disk.unpersist()
print("   - Unpersisted df_persisted_disk.")

# 11. Cost-Based Optimizer Configuration
print("\n11. Cost-Based Optimizer Configuration:")
# Enabled by default in modern Spark versions. Can be fine-tuned.
print(f"   - Is CBO enabled? spark.sql.cbo.enabled: {spark.conf.get('spark.sql.cbo.enabled')}")
# You can provide statistics for CBO to make better decisions (e.g., on tables)
print("     # spark.sql('ANALYZE TABLE my_table COMPUTE STATISTICS FOR ALL COLUMNS')")


print("\n--- VII. User-Defined Functions (UDFs) ---")

# --- Sample DataFrame for UDFs ---
df_udf = spark.createDataFrame([
    (1, "hello world", 10),
    (2, "spark rocks", 25),
    (3, "pyspark", 5)
], ["id", "text", "value"])
print("   - Original DataFrame (df_udf):")
df_udf.show()

# 1. Scalar UDFs (Python UDFs)
print("\n1. Scalar UDFs (Python UDFs):")
def to_upper_case(s):
    if s is not None:
        return s.upper()
    return None

upper_case_udf = udf(to_upper_case, StringType())
df_udf_upper = df_udf.withColumn("upper_text", upper_case_udf(col("text")))
print("   - Applied Python UDF to convert text to uppercase:")
df_udf_upper.show()


# 2. Vectorized/Pandas UDFs for performance improvement
print("\n2. Vectorized/Pandas UDFs:")
# Requires Pandas and PyArrow installed.
# Sum two columns using Pandas UDF
@pandas_udf("long", PandasUDFType.SCALAR)
def pandas_sum_udf(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return s1 + s2

df_pandas_udf = df_udf.withColumn("sum_value_id", pandas_sum_udf(col("value"), col("id")))
print("   - Applied Pandas UDF to sum 'value' and 'id' columns:")
df_pandas_udf.show()

# Grouped map Pandas UDF (e.g., for StandardScaler)
# This is more complex and involves a groupby().applyInPandas() pattern.
# Example: Normalize values within each 'id' group
@pandas_udf(df_udf.schema, PandasUDFType.GROUPED_MAP)
def normalize_group(pdf):
    # This function receives a Pandas DataFrame for each group
    min_val = pdf['value'].min()
    max_val = pdf['value'].max()
    if max_val == min_val: # Avoid division by zero
        pdf['normalized_value'] = 0.0
    else:
        pdf['normalized_value'] = (pdf['value'] - min_val) / (max_val - min_val)
    return pdf

# Append a new column 'normalized_value' to the schema
normalized_schema = df_udf.schema.add(StructField("normalized_value", DoubleType(), True))

df_grouped_map_udf = df_udf.groupBy("id").applyInPandas(normalize_group, schema=normalized_schema)
print("   - Applied Grouped Map Pandas UDF for normalization within groups:")
df_grouped_map_udf.show()


```python pyspark_pattern_examples.py
# ... (previous code)

# 3. User-Defined Aggregate Functions (UDAFs)
print("\n3. User-Defined Aggregate Functions (UDAFs) (Scala/Java mostly, complex in Python):")
# Python UDAFs are generally very complex to implement (require RDD-level implementation).
# Often, built-in aggregations or custom aggregations using Window Functions + HOFs are preferred.
# PySpark does not have a simple @udaf decorator like Scala.
# Example: A simplified Python UDAF concept (more of a grouped UDF than a true UDAF)
# This is usually achieved by `groupBy().agg(collect_list(col).alias("list_col"))` then apply a UDF on list_col
print("   - Python UDAFs are very complex; typically use built-in aggregates, window functions, or collect_list with a UDF.")
print("     Example: Calculate custom median per group (conceptual, as Spark has `percentile_approx`):")

@udf(DoubleType())
def calculate_median_udf(value_list):
    if not value_list:
        return None
    sorted_list = sorted(value_list)
    n = len(sorted_list)
    if n % 2 == 1:
        return float(sorted_list[n // 2])
    else:
        return (float(sorted_list[n // 2 - 1]) + float(sorted_list[n // 2])) / 2.0

df_udf_grouped = df_udf.groupBy("id").agg(collect_list("value").alias("value_list"))
df_median_per_group = df_udf_grouped.withColumn("median_value", calculate_median_udf(col("value_list")))
print("   - Group-wise median calculated using collect_list and a regular UDF:")
df_median_per_group.show()


# 4. SQL UDFs
print("\n4. SQL UDFs:")
# Register the Python UDF for use in Spark SQL
spark.udf.register("py_upper_case", to_upper_case, StringType())

df_udf.createOrReplaceTempView("udf_data")
df_sql_udf = spark.sql("SELECT id, text, py_upper_case(text) as upper_text_sql FROM udf_data")
print("   - Applied Python UDF via Spark SQL:")
df_sql_udf.show()


print("\n--- VIII. Structured Streaming ---")
# Streaming examples require a running query and checkpoint locations.
# These were partially demonstrated in II.7 and II.8. Let's make them more concrete.

streaming_input_dir = os.path.join(OUTPUT_BASE_DIR, "streaming_input_source")
streaming_output_dir = os.path.join(OUTPUT_BASE_DIR, "streaming_output_sink")
streaming_checkpoint_dir = os.path.join(OUTPUT_BASE_DIR, "streaming_checkpoint")

os.makedirs(streaming_input_dir, exist_ok=True)
os.makedirs(streaming_output_dir, exist_ok=True)
os.makedirs(streaming_checkpoint_dir, exist_ok=True)


# Generate some initial files for the file stream source
with open(os.path.join(streaming_input_dir, "data_batch_1.csv"), "w") as f:
    f.write("id,value,category\n")
    f.write("1,10,A\n")
    f.write("2,20,B\n")
time.sleep(1) # Ensure file timestamps are different

# Define a schema for the streaming input
stream_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("value", IntegerType(), True),
    StructField("category", StringType(), True)
])

# 1. Windowed Aggregations on Streams
print("\n1. Windowed Aggregations on Streams:")
df_stream_window_source = spark.readStream \
    .schema(stream_schema) \
    .csv(streaming_input_dir)

# tumbling window of 5 seconds, updating every 2 seconds
df_windowed_agg = df_stream_window_source.withWatermark("timestamp", "10 seconds") \
    .groupBy(
        window(col("timestamp"), "5 seconds", "2 seconds"),
        col("category")
    ).agg(
        sum("value").alias("sum_value"),
        count("id").alias("count_events")
    )

query_windowed = df_windowed_agg.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .option("checkpointLocation", os.path.join(streaming_checkpoint_dir, "windowed_agg")) \
    .start()
print("   - Started windowed aggregation stream (tumbling window 5s, slide 2s).")

# Simulate more data arriving
with open(os.path.join(streaming_input_dir, "data_batch_2.csv"), "w") as f:
    f.write(f"{int(time.time())},30,A\n")
    f.write(f"{int(time.time())},40,C\n")
time.sleep(5) # Allow some processing

with open(os.path.join(streaming_input_dir, "data_batch_3.csv"), "w") as f:
    f.write(f"{int(time.time())},50,B\n")
    f.write(f"{int(time.time())},60,A\n")
time.sleep(5) # Allow more processing

query_windowed.stop()
print("     Windowed aggregation stream stopped.")

# 2. Stateful Stream Processing (implicit in windowed agg/joins, or explicit with `mapGroupsWithState`)
print("\n2. Stateful Stream Processing (conceptual with `mapGroupsWithState`):")
print("   - `mapGroupsWithState` allows explicit state management per key.")
print("     # df_stateful = df_stream_source.groupByKey(lambda x: x.id) \\")
print("     #     .mapGroupsWithState(func, output_schema, GroupStateTimeout.NoTimeout())")
print("     Windowed aggregations are an implicit form of stateful processing.")

# 3. Watermarking for Late Data
print("\n3. Watermarking for Late Data:")
print("   - Demonstrated in the windowed aggregation example (`.withWatermark('timestamp', '10 seconds')`).")
print("     This tells Spark to drop data older than the watermark + allowed lateness.")

# 4. Joining Streams with Static DataFrames
print("\n4. Joining Streams with Static DataFrames:")
df_static_lookup = spark.createDataFrame([
    ("A", "Category One"),
    ("B", "Category Two"),
    ("C", "Category Three")
], ["category", "category_description"])

df_stream_static_join_source = spark.readStream \
    .schema(stream_schema) \
    .csv(streaming_input_dir)

df_stream_static_join = df_stream_static_join_source.join(
    df_static_lookup,
    on="category",
    how="left_outer"
)

query_static_join = df_stream_static_join.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .option("checkpointLocation", os.path.join(streaming_checkpoint_dir, "static_join")) \
    .start()
print("   - Started stream-to-static join. Will run for 5 seconds.")
time.sleep(5)
query_static_join.stop()
print("     Stream-to-static join stopped.")


# 5. Joining Streams with Streams (conceptual - requires specific join conditions)
print("\n5. Joining Streams with Streams (conceptual):")
print("   - Requires event time watermarks and join conditions on event time.")
print("     # stream1.withWatermark('time1', '10 minutes').join(stream2.withWatermark('time2', '1 minute'), \\")
print("     #     expr('id1 = id2 AND time1 >= time2 AND time1 <= time2 + interval 1 minute'))")

# 6. Different Output Modes (Append, Complete, Update)
print("\n6. Different Output Modes:")
print("   - Append: (default for many sinks) Only new rows added to the result table are written to sink. (Used for File, Kafka)")
print("   - Complete: The entire result table is written to the sink every time. (Used for `groupBy` aggregations)")
print("   - Update: Only rows that have been updated in the result table since the last trigger are written to the sink. (Used for `groupBy` aggregations)")
print("     Demonstrated with console sinks for windowed aggregations (`update`) and simple file streams (`append`).")


# 7. Idempotent Sinks (`foreachBatch`)
print("\n7. Idempotent Sinks (`foreachBatch`):")
print("   - Demonstrated in II.8. The `foreachBatch` function receives a DataFrame and batch_id, allowing custom idempotent writes.")

# 8. Checkpointing
print("\n8. Checkpointing:")
print("   - Demonstrated throughout the streaming examples by specifying `checkpointLocation` for each stream query.")
print("     This stores state, progress, and metadata to allow recovery from failures.")


print("\n--- IX. Machine Learning (MLlib) ---")

# --- Sample DataFrame for MLlib ---
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

data_ml = [
    (0, "A", 10.0, 1.0, 0),
    (1, "B", 20.0, 2.0, 1),
    (2, "A", 15.0, 1.5, 0),
    (3, "C", 25.0, 2.5, 1),
    (4, "B", 12.0, 1.2, 0),
    (5, "C", 30.0, 3.0, 1)
]
schema_ml = ["id", "category", "feature1", "feature2", "label"]
df_ml = spark.createDataFrame(data_ml, schema_ml)
print("   - Original DataFrame (df_ml) for MLlib examples:")
df_ml.show()


# 1. Feature Engineering (VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, Tokenizer)
print("\n1. Feature Engineering:")

# StringIndexer: Converts string column to index numbers
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
df_indexed = indexer.fit(df_ml).transform(df_ml)
print("   - After StringIndexer (category to categoryIndex):")
df_indexed.show()

# OneHotEncoder: Converts index numbers to one-hot encoded vectors
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
df_encoded = encoder.fit(df_indexed).transform(df_indexed)
print("   - After OneHotEncoder (categoryIndex to categoryVec):")
df_encoded.show()

# VectorAssembler: Combines feature columns into a single vector column
assembler = VectorAssembler(inputCols=["feature1", "feature2", "categoryVec"], outputCol="features")
df_assembled = assembler.transform(df_encoded)
print("   - After VectorAssembler (multiple features to single 'features' vector):")
df_assembled.show()

# StandardScaler: Scales features to unit variance and/or zero mean
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)
print("   - After StandardScaler (features to scaledFeatures):")
df_scaled.show()

# Tokenizer (conceptual, for text data)
print("   - Tokenizer (conceptual for text):")
print("     # from pyspark.ml.feature import Tokenizer")
print("     # tokenizer = Tokenizer(inputCol='text_col', outputCol='words')")


# 2. Model Training (Estimators, Transformers)
print("\n2. Model Training (Estimators, Transformers):")
# Estimators: Learn from data (e.g., LogisticRegression) -> produce Model (Transformer)
# Transformers: Transform data (e.g., VectorAssembler, or a trained LogisticRegressionModel)

# We have `df_scaled` with 'scaledFeatures' and 'label'
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=10)
lr_model = lr.fit(df_scaled) # Fitting an Estimator
print("   - Trained Logistic Regression Model.")

# Make predictions using the trained model (a Transformer)
predictions = lr_model.transform(df_scaled)
print("   - Predictions from Logistic Regression Model:")
predictions.select("id", "label", "prediction", "probability").show()


# 3. Model Evaluation (Evaluators)
print("\n3. Model Evaluation (Evaluators):")
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"   - Area Under ROC (AUC) for the Logistic Regression model: {auc:.4f}")


# 4. ML Pipelines for Workflow Orchestration
print("\n4. ML Pipelines for Workflow Orchestration:")
# A Pipeline chains multiple Transformers and Estimators together.
pipeline = Pipeline(stages=[
    indexer,
    encoder,
    assembler,
    scaler,
    lr
])

# Fit the pipeline to data
pipeline_model = pipeline.fit(df_ml)
print("   - Trained a full ML Pipeline.")

# Make predictions using the pipeline model
pipeline_predictions = pipeline_model.transform(df_ml)
print("   - Predictions from ML Pipeline:")
pipeline_predictions.select("id", "category", "feature1", "feature2", "label", "prediction").show()


# 5. Hyperparameter Tuning (CrossValidator, TrainValidationSplit)
print("\n5. Hyperparameter Tuning (CrossValidator, TrainValidationSplit):")
# Use the pipeline and define a parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=2, # Use 2 folds for faster example
    seed=42
)

cv_model = crossval.fit(df_ml)
print("   - Performed Cross-Validation for hyperparameter tuning.")
print(f"     Best model's best parameters: {cv_model.bestModel.stages[-1].extractParamMap()}")


# 6. Model Persistence (Saving/Loading Models)
print("\n6. Model Persistence (Saving/Loading Models):")
model_path = os.path.join(OUTPUT_BASE_DIR, "lr_pipeline_model")

# Save the trained pipeline model
pipeline_model.write().overwrite().save(model_path)
print(f"   - Saved pipeline model to: {model_path}")

# Load the model back
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load(model_path)
print(f"   - Loaded pipeline model from: {model_path}")

# Use the loaded model for new predictions
new_data = spark.createDataFrame([(6, "A", 18.0, 1.8, 0)], schema_ml)
loaded_predictions = loaded_model.transform(new_data)
print("   - Predictions using loaded model on new data:")
loaded_predictions.select("id", "label", "prediction").show()


print("\n--- X. Error Handling & Debugging ---")

# 1. Using `try-except` blocks for I/O operations
print("\n1. Using `try-except` blocks for I/O operations:")
invalid_path = os.path.join(OUTPUT_BASE_DIR, "non_existent_dir", "file.csv")
try:
    spark.read.csv(invalid_path, header=True).show()
except Exception as e:
    print(f"   - Caught expected error when reading non-existent path: {e}")

# 2. Logging Spark operations
print("\n2. Logging Spark operations:")
print("   - Spark logs are usually printed to console (stdout/stderr) or configured log files.")
print("   - You can configure logging levels in `log4j.properties` (for Java/Scala) or via Python's logging module.")
import logging
logging.getLogger("py4j").setLevel(logging.ERROR) # Suppress py4j debug logs
logging.getLogger("pyspark").setLevel(logging.INFO) # Set PySpark specific logs
print("   - Setting logging levels (e.g., logging.getLogger('pyspark').setLevel(logging.INFO))")

# 3. Inspecting DataFrame content (`show`, `head`, `take`, `collect`, `printSchema`, `describe`)
print("\n3. Inspecting DataFrame content:")
print("   - `df.show()`: Show top 20 rows.")
df_basic.show(2)
print("   - `df.head(n)`: Returns the first n rows as a list of Row objects.")
print(f"     df_basic.head(1): {df_basic.head(1)}")
print("   - `df.take(n)`: Returns the first n rows as a list of Row objects.")
print(f"     df_basic.take(1): {df_basic.take(1)}")
print("   - `df.collect()`: Returns all rows as a list of Row objects (use with caution on large DFs).")
# print(f"     df_basic.collect(): {df_basic.collect()}")
print("   - `df.printSchema()`: Prints the schema of the DataFrame.")
df_basic.printSchema()
print("   - `df.describe()`: Computes summary statistics for numeric columns.")
df_basic.describe("age", "salary").show()

# 4. Analyzing Stack Traces for Job Failures
print("\n4. Analyzing Stack Traces for Job Failures (Conceptual):")
print("   - When a Spark job fails, examine the full stack trace in the Spark UI or console.")
print("   - Look for `Caused by:` to pinpoint the root cause.")
print("   - Often `NullPointerException`, `AnalysisException`, `TaskFailedException` are common.")

# 5. Spark UI for Debugging Failed Stages/Tasks
print("\n5. Spark UI for Debugging Failed Stages/Tasks (Conceptual):")
print("   - Check the Jobs/Stages/Tasks tabs in the Spark UI (e.g., `http://localhost:4040`) for errors, input/output metrics, and task durations.")
print("   - Use the 'Logs' link for specific executor or driver logs.")


print("\n--- XI. Deployment & Orchestration ---")

# 1. Spark-Submit Job Submission
print("\n1. Spark-Submit Job Submission (Conceptual):")
print("   - This entire script is intended to be run via `spark-submit` for deployment.")
print("     # spark-submit --master yarn --deploy-mode client your_script.py arg1 arg2")
print("   - Key parameters: `--master`, `--deploy-mode`, `--num-executors`, `--executor-memory`, etc.")

# 2. Passing Configuration and Arguments to Spark Jobs
print("\n2. Passing Configuration and Arguments to Spark Jobs:")
print("   - Command-line arguments: `spark-submit your_script.py --input-path s3://...`")
import sys
if len(sys.argv) > 1:
    print(f"   - Received command-line arguments: {sys.argv[1:]}")
else:
    print("   - No command-line arguments received (run with `spark-submit ... script.py arg1 arg2` to test)")

# 3. Managing Dependencies (JARs, Python files)
print("\n3. Managing Dependencies:")
print("   - `--jars`: For Java/Scala JARs (e.g., JDBC drivers, Delta Lake JARs).")
print("     # spark-submit --jars postgresql-42.2.5.jar your_script.py")
print("   - `--py-files`: For additional Python `.py`, `.zip`, `.egg` files.")
print("     # spark-submit --py-files my_utils.zip your_script.py")
print("   - Conda/Virtualenv packing: For complex Python dependencies, often zip the entire env.")

# 4. Environment Variable Configuration
print("\n4. Environment Variable Configuration:")
print("   - Set before `spark-submit` to influence Spark or your application.")
print("     # export MY_APP_VAR='some_value'")
print("     # spark-submit ...")
os.environ["MY_APP_VAR_DEMO"] = "This is from an env var"
print(f"   - Read from Python: os.environ.get('MY_APP_VAR_DEMO'): {os.environ.get('MY_APP_VAR_DEMO')}")


print("\n--- XII. Testing ---")

# 1. Unit Testing Spark Code (e.g., using `spark-testing-base` or creating local `SparkSession` for tests)
print("\n1. Unit Testing Spark Code:")
print("   - For unit tests, create a local SparkSession:")
print("     # test_spark = SparkSession.builder.appName('test').master('local[*]').getOrCreate()")
print("   - Use libraries like `pytest`.")
print("   - Compare DataFrames using `.collect()` and assert equality of rows, or `.exceptAll()` / `.subtract()`.")
print("   - `spark-testing-base` (Scala) or custom Python helpers for comparing DFs.")

# 2. Mocking External Dependencies for Tests
print("\n2. Mocking External Dependencies for Tests:")
print("   - When testing code that interacts with databases, S3, Kafka, etc., use Python's `unittest.mock` library.")
print("   - Mock the functions/methods that perform I/O to return predefined test data.")
print("     # from unittest.mock import patch")
print("     # @patch('your_module.read_from_s3', return_value=mock_df)")
print("     # def test_my_function(mock_read_s3): ...")


print("\n--- All PySpark pattern examples executed. ---")

# Stop the SparkSession
spark.stop()

# Clean up output directory
if os.path.exists(OUTPUT_BASE_DIR):
    print(f"\nCleaning up output directory: {OUTPUT_BASE_DIR}")
    shutil.rmtree(OUTPUT_BASE_DIR)
```
