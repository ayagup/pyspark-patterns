### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 7:04:45 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Storage Patterns
> - Parquet Storage Pattern
> - ORC Storage Pattern
> - Avro Storage Pattern
> - JSON Storage Pattern
> - CSV Storage Pattern
> - Delta Lake Pattern
> - Iceberg Pattern
> - Hudi Pattern
> - Compaction Pattern
> - Vacuum Pattern
> - Z-Ordering Pattern
> - Data Skipping Pattern
> - Predicate Pushdown Pattern
> - Columnar Storage Pattern
> - Row-based Storage Pattern
> - Compression Pattern
> - Partitioned Storage Pattern
> ```

#### _Assistant_

> I'll generate PySpark programs demonstrating each storage pattern. Let me create comprehensive examples for all these patterns.
> 
> ```python storage_patterns/parquet_storage.py
> """
> Parquet Storage Pattern
> - Columnar storage format
> - Efficient compression and encoding
> - Schema evolution support
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> 
> spark = SparkSession.builder \
>     .appName("ParquetStoragePattern") \
>     .config("spark.sql.parquet.compression.codec", "snappy") \
>     .getOrCreate()
> 
> # Create sample data
> data = [
>     (1, "John", "Engineering", 75000.0),
>     (2, "Jane", "Marketing", 65000.0),
>     (3, "Bob", "Sales", 55000.0)
> ]
> 
> schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("name", StringType(), True),
>     StructField("department", StringType(), True),
>     StructField("salary", DoubleType(), True)
> ])
> 
> df = spark.createDataFrame(data, schema)
> 
> # Write to Parquet with different compression codecs
> df.write.mode("overwrite") \
>     .option("compression", "snappy") \
>     .parquet("output/parquet/snappy")
> 
> df.write.mode("overwrite") \
>     .option("compression", "gzip") \
>     .parquet("output/parquet/gzip")
> 
> # Read Parquet files
> parquet_df = spark.read.parquet("output/parquet/snappy")
> parquet_df.show()
> 
> # Schema evolution example
> new_data = [
>     (4, "Alice", "HR", 70000.0, "New York")
> ]
> 
> new_schema = schema.add("location", StringType(), True)
> new_df = spark.createDataFrame(new_data, new_schema)
> 
> new_df.write.mode("append") \
>     .option("mergeSchema", "true") \
>     .parquet("output/parquet/evolved")
> 
> # Read with evolved schema
> evolved_df = spark.read.option("mergeSchema", "true").parquet("output/parquet/evolved")
> evolved_df.printSchema()
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/orc_storage.py
> """
> ORC Storage Pattern
> - Optimized Row Columnar format
> - High compression ratios
> - Built-in indexes and statistics
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
> from datetime import datetime
> 
> spark = SparkSession.builder \
>     .appName("ORCStoragePattern") \
>     .config("spark.sql.orc.compression.codec", "zlib") \
>     .getOrCreate()
> 
> # Create sample data with timestamps
> data = [
>     (1, "Product A", 100, datetime(2024, 1, 1, 10, 0)),
>     (2, "Product B", 200, datetime(2024, 1, 2, 11, 0)),
>     (3, "Product C", 150, datetime(2024, 1, 3, 12, 0))
> ]
> 
> schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("product_name", StringType(), True),
>     StructField("quantity", IntegerType(), True),
>     StructField("timestamp", TimestampType(), True)
> ])
> 
> df = spark.createDataFrame(data, schema)
> 
> # Write ORC with different compression
> df.write.mode("overwrite") \
>     .option("compression", "zlib") \
>     .orc("output/orc/zlib")
> 
> df.write.mode("overwrite") \
>     .option("compression", "snappy") \
>     .orc("output/orc/snappy")
> 
> # Read ORC with predicate pushdown
> orc_df = spark.read.orc("output/orc/zlib")
> filtered_df = orc_df.filter(orc_df.quantity > 120)
> filtered_df.show()
> 
> # ORC supports ACID properties and schema evolution
> print("ORC Statistics:")
> print(f"Record count: {orc_df.count()}")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/avro_storage.py
> """
> Avro Storage Pattern
> - Row-based binary format
> - Schema embedded in data
> - Language-agnostic serialization
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
> 
> spark = SparkSession.builder \
>     .appName("AvroStoragePattern") \
>     .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0") \
>     .getOrCreate()
> 
> # Create nested data structure
> data = [
>     (1, "User1", ["tag1", "tag2", "tag3"]),
>     (2, "User2", ["tag2", "tag4"]),
>     (3, "User3", ["tag1", "tag5", "tag6"])
> ]
> 
> schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("username", StringType(), True),
>     StructField("tags", ArrayType(StringType()), True)
> ])
> 
> df = spark.createDataFrame(data, schema)
> 
> # Write to Avro
> df.write.mode("overwrite") \
>     .format("avro") \
>     .option("compression", "snappy") \
>     .save("output/avro/data")
> 
> # Read from Avro
> avro_df = spark.read.format("avro").load("output/avro/data")
> avro_df.show(truncate=False)
> 
> # Schema evolution with Avro
> avro_df.printSchema()
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/json_storage.py
> """
> JSON Storage Pattern
> - Human-readable format
> - Nested data support
> - Schema-on-read
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, from_json
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType
> 
> spark = SparkSession.builder \
>     .appName("JSONStoragePattern") \
>     .getOrCreate()
> 
> # Create nested JSON-like data
> data = [
>     (1, "Order1", {"item": "Laptop", "price": 1200}),
>     (2, "Order2", {"item": "Phone", "price": 800}),
>     (3, "Order3", {"item": "Tablet", "price": 500})
> ]
> 
> schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("order_id", StringType(), True),
>     StructField("details", MapType(StringType(), StringType()), True)
> ])
> 
> df = spark.createDataFrame(data, schema)
> 
> # Write to JSON
> df.write.mode("overwrite") \
>     .option("compression", "gzip") \
>     .json("output/json/data")
> 
> # Write pretty-printed JSON
> df.write.mode("overwrite") \
>     .option("pretty", "true") \
>     .json("output/json/pretty")
> 
> # Read JSON with schema inference
> json_df = spark.read.json("output/json/data")
> json_df.show(truncate=False)
> 
> # Read JSON lines format
> df.select("id", "order_id").write.mode("overwrite") \
>     .json("output/json/lines")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/csv_storage.py
> """
> CSV Storage Pattern
> - Simple text-based format
> - Wide compatibility
> - Header support
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> 
> spark = SparkSession.builder \
>     .appName("CSVStoragePattern") \
>     .getOrCreate()
> 
> # Create sample data
> data = [
>     (1, "Alice", 28, 75000.50),
>     (2, "Bob", 35, 85000.75),
>     (3, "Charlie", 42, 95000.00)
> ]
> 
> schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("name", StringType(), True),
>     StructField("age", IntegerType(), True),
>     StructField("salary", DoubleType(), True)
> ])
> 
> df = spark.createDataFrame(data, schema)
> 
> # Write CSV with headers
> df.write.mode("overwrite") \
>     .option("header", "true") \
>     .option("delimiter", ",") \
>     .csv("output/csv/with_header")
> 
> # Write CSV with custom delimiter
> df.write.mode("overwrite") \
>     .option("header", "true") \
>     .option("delimiter", "|") \
>     .csv("output/csv/pipe_delimited")
> 
> # Read CSV with schema inference
> csv_df = spark.read \
>     .option("header", "true") \
>     .option("inferSchema", "true") \
>     .csv("output/csv/with_header")
> 
> csv_df.show()
> 
> # Read CSV with explicit schema
> csv_df_schema = spark.read \
>     .option("header", "true") \
>     .schema(schema) \
>     .csv("output/csv/with_header")
> 
> csv_df_schema.printSchema()
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/delta_lake.py
> """
> Delta Lake Pattern
> - ACID transactions
> - Time travel
> - Schema evolution
> - Upserts (MERGE operations)
> """
> from pyspark.sql import SparkSession
> from delta import *
> 
> spark = SparkSession.builder \
>     .appName("DeltaLakePattern") \
>     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
>     .getOrCreate()
> 
> # Create initial data
> data = [
>     (1, "Product A", 100, 10.50),
>     (2, "Product B", 200, 20.75),
>     (3, "Product C", 150, 15.25)
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "quantity", "price"])
> 
> # Write to Delta table
> df.write.format("delta").mode("overwrite").save("output/delta/products")
> 
> # Read Delta table
> delta_df = spark.read.format("delta").load("output/delta/products")
> delta_df.show()
> 
> # Update data (ACID transaction)
> updates = spark.createDataFrame([(1, "Product A Updated", 120, 12.00)], 
>                                 ["id", "name", "quantity", "price"])
> 
> deltaTable = DeltaTable.forPath(spark, "output/delta/products")
> 
> # MERGE operation (Upsert)
> deltaTable.alias("target").merge(
>     updates.alias("source"),
>     "target.id = source.id"
> ).whenMatchedUpdateAll() \
>  .whenNotMatchedInsertAll() \
>  .execute()
> 
> # Time travel - read previous version
> print("\n--- Current version ---")
> spark.read.format("delta").load("output/delta/products").show()
> 
> print("\n--- Previous version (version 0) ---")
> spark.read.format("delta").option("versionAsOf", 0).load("output/delta/products").show()
> 
> # Delete operation
> deltaTable.delete("quantity < 150")
> 
> # Vacuum old files (7 days retention)
> # deltaTable.vacuum(168)  # hours
> 
> # Schema evolution
> new_data = spark.createDataFrame([(4, "Product D", 300, 25.00, "Electronics")], 
>                                  ["id", "name", "quantity", "price", "category"])
> 
> new_data.write.format("delta") \
>     .mode("append") \
>     .option("mergeSchema", "true") \
>     .save("output/delta/products")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/iceberg_pattern.py
> """
> Apache Iceberg Pattern
> - Table format for huge analytic datasets
> - Hidden partitioning
> - Time travel and rollback
> - Schema evolution
> """
> from pyspark.sql import SparkSession
> 
> spark = SparkSession.builder \
>     .appName("IcebergPattern") \
>     .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
>     .config("spark.sql.catalog.spark_catalog.type", "hive") \
>     .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
>     .config("spark.sql.catalog.local.type", "hadoop") \
>     .config("spark.sql.catalog.local.warehouse", "output/iceberg") \
>     .getOrCreate()
> 
> # Create Iceberg table
> spark.sql("""
>     CREATE TABLE IF NOT EXISTS local.db.sales (
>         id INT,
>         product STRING,
>         quantity INT,
>         price DOUBLE,
>         sale_date DATE
>     ) USING iceberg
>     PARTITIONED BY (days(sale_date))
> """)
> 
> # Insert data
> spark.sql("""
>     INSERT INTO local.db.sales VALUES 
>     (1, 'Product A', 100, 10.50, DATE '2024-01-01'),
>     (2, 'Product B', 200, 20.75, DATE '2024-01-02'),
>     (3, 'Product C', 150, 15.25, DATE '2024-01-03')
> """)
> 
> # Query Iceberg table
> spark.sql("SELECT * FROM local.db.sales").show()
> 
> # Time travel query
> spark.sql("SELECT * FROM local.db.sales FOR SYSTEM_VERSION AS OF 1").show()
> 
> # Schema evolution - add column
> spark.sql("ALTER TABLE local.db.sales ADD COLUMN category STRING")
> 
> # Update records
> spark.sql("""
>     UPDATE local.db.sales 
>     SET category = 'Electronics' 
>     WHERE id = 1
> """)
> 
> # Partition evolution (no data rewrite needed)
> spark.sql("ALTER TABLE local.db.sales ADD PARTITION FIELD bucket(10, id)")
> 
> # Show table history
> spark.sql("SELECT * FROM local.db.sales.history").show(truncate=False)
> 
> # Show snapshots
> spark.sql("SELECT * FROM local.db.sales.snapshots").show(truncate=False)
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/hudi_pattern.py
> """
> Apache Hudi Pattern
> - Upserts and incremental processing
> - Copy-on-Write (COW) and Merge-on-Read (MOR) tables
> - Record-level updates
> """
> from pyspark.sql import SparkSession
> 
> spark = SparkSession.builder \
>     .appName("HudiPattern") \
>     .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.hudi.catalog.HoodieCatalog") \
>     .config("spark.sql.extensions", "org.apache.spark.sql.hudi.HoodieSparkSessionExtension") \
>     .getOrCreate()
> 
> # Hudi configuration
> hudi_options = {
>     'hoodie.table.name': 'users_table',
>     'hoodie.datasource.write.recordkey.field': 'id',
>     'hoodie.datasource.write.partitionpath.field': 'department',
>     'hoodie.datasource.write.table.name': 'users_table',
>     'hoodie.datasource.write.operation': 'upsert',
>     'hoodie.datasource.write.precombine.field': 'ts',
>     'hoodie.upsert.shuffle.parallelism': 2,
>     'hoodie.insert.shuffle.parallelism': 2
> }
> 
> # Create initial data
> data = [
>     (1, "John", "Engineering", 75000, 1704067200000),
>     (2, "Jane", "Marketing", 65000, 1704067200000),
>     (3, "Bob", "Sales", 55000, 1704067200000)
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "department", "salary", "ts"])
> 
> # Write to Hudi (Copy-on-Write)
> df.write.format("hudi") \
>     .options(**hudi_options) \
>     .mode("overwrite") \
>     .save("output/hudi/users_cow")
> 
> # Read from Hudi
> hudi_df = spark.read.format("hudi").load("output/hudi/users_cow")
> hudi_df.show()
> 
> # Upsert data
> updates = spark.createDataFrame([
>     (1, "John Updated", "Engineering", 80000, 1704153600000),
>     (4, "Alice", "HR", 70000, 1704153600000)
> ], ["id", "name", "department", "salary", "ts"])
> 
> updates.write.format("hudi") \
>     .options(**hudi_options) \
>     .mode("append") \
>     .save("output/hudi/users_cow")
> 
> # Read updated data
> updated_df = spark.read.format("hudi").load("output/hudi/users_cow")
> print("\n--- After Upsert ---")
> updated_df.show()
> 
> # Incremental query
> incremental_df = spark.read.format("hudi") \
>     .option("hoodie.datasource.query.type", "incremental") \
>     .option("hoodie.datasource.read.begin.instanttime", "20240101000000") \
>     .load("output/hudi/users_cow")
> 
> print("\n--- Incremental Changes ---")
> incremental_df.show()
> 
> # Time travel query
> spark.read.format("hudi") \
>     .option("as.of.instant", "20240101000000") \
>     .load("output/hudi/users_cow").show()
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/compaction_pattern.py
> """
> Compaction Pattern
> - Merge small files into larger ones
> - Improve query performance
> - Reduce metadata overhead
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import os
> 
> spark = SparkSession.builder \
>     .appName("CompactionPattern") \
>     .config("spark.sql.files.maxRecordsPerFile", 10000) \
>     .getOrCreate()
> 
> # Create data that will generate many small files
> base_path = "output/compaction/raw"
> 
> for i in range(10):
>     data = [(j, f"Record_{i}_{j}") for j in range(100)]
>     df = spark.createDataFrame(data, ["id", "value"])
>     df.write.mode("append").parquet(base_path)
> 
> print(f"\n--- Before Compaction ---")
> print(f"Number of files: {len([f for f in os.listdir(base_path) if f.endswith('.parquet')])}")
> 
> # Read all small files
> df_to_compact = spark.read.parquet(base_path)
> 
> # Compaction - repartition and write to new location
> compacted_path = "output/compaction/compacted"
> df_to_compact.repartition(2).write.mode("overwrite").parquet(compacted_path)
> 
> print(f"\n--- After Compaction ---")
> print(f"Number of files: {len([f for f in os.listdir(compacted_path) if f.endswith('.parquet')])}")
> print(f"Total records: {df_to_compact.count()}")
> 
> # Coalesce for fewer partitions (no shuffle)
> df_to_compact.coalesce(1).write.mode("overwrite").parquet("output/compaction/coalesced")
> 
> # For Delta Lake compaction
> # OPTIMIZE delta.`/path/to/table`
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/vacuum_pattern.py
> """
> Vacuum Pattern
> - Remove old/unused files
> - Reclaim storage space
> - Maintain only necessary versions
> """
> from pyspark.sql import SparkSession
> from delta import *
> from datetime import datetime, timedelta
> 
> spark = SparkSession.builder \
>     .appName("VacuumPattern") \
>     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
>     .getOrCreate()
> 
> # Create Delta table
> data = [(1, "Data1"), (2, "Data2"), (3, "Data3")]
> df = spark.createDataFrame(data, ["id", "value"])
> delta_path = "output/vacuum/delta_table"
> 
> df.write.format("delta").mode("overwrite").save(delta_path)
> 
> # Perform multiple updates to create versions
> for i in range(5):
>     update_data = [(1, f"Updated_{i}"), (2, f"Updated_{i}")]
>     update_df = spark.createDataFrame(update_data, ["id", "value"])
>     
>     deltaTable = DeltaTable.forPath(spark, delta_path)
>     deltaTable.alias("target").merge(
>         update_df.alias("source"),
>         "target.id = source.id"
>     ).whenMatchedUpdateAll().execute()
> 
> print("\n--- Before Vacuum ---")
> deltaTable = DeltaTable.forPath(spark, delta_path)
> 
> # Show all files
> import os
> all_files = []
> for root, dirs, files in os.walk(delta_path):
>     all_files.extend([f for f in files if f.endswith('.parquet')])
> print(f"Number of parquet files: {len(all_files)}")
> 
> # Vacuum old files (retain 0 hours for demo - use 168 hours/7 days in production)
> # Need to disable retention check for demo
> spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
> 
> deltaTable.vacuum(0)  # Remove files older than 0 hours (for demo only!)
> 
> print("\n--- After Vacuum ---")
> all_files = []
> for root, dirs, files in os.walk(delta_path):
>     all_files.extend([f for f in files if f.endswith('.parquet')])
> print(f"Number of parquet files: {len(all_files)}")
> 
> # Best practice: vacuum with retention period
> # deltaTable.vacuum(168)  # 7 days retention
> 
> # Dry run to see what would be deleted
> # deltaTable.vacuum(168, dryRun=True)
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/z_ordering_pattern.py
> """
> Z-Ordering Pattern
> - Multi-dimensional clustering
> - Colocate related data
> - Improve query performance for multiple columns
> """
> from pyspark.sql import SparkSession
> from delta import *
> from pyspark.sql.functions import col
> import random
> 
> spark = SparkSession.builder \
>     .appName("ZOrderingPattern") \
>     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
>     .getOrCreate()
> 
> # Generate sample data with multiple dimensions
> data = []
> for i in range(10000):
>     data.append((
>         i,
>         random.choice(["USA", "UK", "Canada", "Australia"]),
>         random.choice(["Electronics", "Clothing", "Food", "Books"]),
>         random.randint(1, 100),
>         random.uniform(10, 1000)
>     ))
> 
> df = spark.createDataFrame(data, ["id", "country", "category", "quantity", "price"])
> 
> delta_path = "output/zorder/sales"
> 
> # Write initial data
> df.write.format("delta").mode("overwrite").save(delta_path)
> 
> print("\n--- Before Z-Ordering ---")
> # Query on multiple columns
> result1 = spark.read.format("delta").load(delta_path) \
>     .filter((col("country") == "USA") & (col("category") == "Electronics")) \
>     .count()
> print(f"Filtered records: {result1}")
> 
> # Apply Z-Ordering on country and category columns
> deltaTable = DeltaTable.forPath(spark, delta_path)
> deltaTable.optimize().executeZOrderBy("country", "category")
> 
> print("\n--- After Z-Ordering ---")
> # Same query should be faster due to data skipping
> result2 = spark.read.format("delta").load(delta_path) \
>     .filter((col("country") == "USA") & (col("category") == "Electronics")) \
>     .count()
> print(f"Filtered records: {result2}")
> 
> # View table details
> spark.sql(f"DESCRIBE DETAIL delta.`{delta_path}`").show(truncate=False)
> 
> # Z-Ordering is particularly effective when:
> # - Queries filter on multiple columns
> # - Data has high cardinality
> # - Columns are frequently used together in WHERE clauses
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/data_skipping_pattern.py
> """
> Data Skipping Pattern
> - Skip reading irrelevant data files
> - Use statistics and metadata
> - Optimize query performance
> """
> from pyspark.sql import SparkSession
> from delta import *
> from pyspark.sql.functions import col, year, month
> from datetime import datetime, timedelta
> 
> spark = SparkSession.builder \
>     .appName("DataSkippingPattern") \
>     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
>     .getOrCreate()
> 
> # Generate time-series data
> base_date = datetime(2024, 1, 1)
> data = []
> for i in range(1000):
>     current_date = base_date + timedelta(days=i % 365)
>     data.append((
>         i,
>         current_date,
>         f"Event_{i}",
>         i % 10
>     ))
> 
> df = spark.createDataFrame(data, ["id", "event_date", "event_type", "value"])
> 
> # Add partitioning columns
> df = df.withColumn("year", year(col("event_date"))) \
>        .withColumn("month", month(col("event_date")))
> 
> delta_path = "output/dataskipping/events"
> 
> # Write with partitioning
> df.write.format("delta") \
>     .partitionBy("year", "month") \
>     .mode("overwrite") \
>     .save(delta_path)
> 
> # Collect statistics for data skipping
> deltaTable = DeltaTable.forPath(spark, delta_path)
> deltaTable.optimize().executeCompaction()
> 
> print("\n--- Data Skipping with Partition Pruning ---")
> # This query will skip entire partitions
> result1 = spark.read.format("delta").load(delta_path) \
>     .filter((col("year") == 2024) & (col("month") == 1)) \
>     .count()
> print(f"Records in Jan 2024: {result1}")
> 
> # Enable Delta stats collection
> spark.conf.set("spark.databricks.delta.stats.collect", "true")
> 
> # Query with value-based filtering (uses statistics)
> print("\n--- Data Skipping with Statistics ---")
> result2 = spark.read.format("delta").load(delta_path) \
>     .filter(col("value") > 8) \
>     .count()
> print(f"Records with value > 8: {result2}")
> 
> # Show files that would be scanned
> explain_df = spark.read.format("delta").load(delta_path) \
>     .filter((col("year") == 2024) & (col("month") == 1) & (col("value") > 5))
> explain_df.explain(True)
> 
> # Delta automatically maintains min/max statistics for:
> # - First 32 columns
> # - Numeric types, strings, dates, timestamps
> # - Enables data skipping at file level
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/predicate_pushdown_pattern.py
> """
> Predicate Pushdown Pattern
> - Push filters to storage layer
> - Reduce data transfer
> - Improve query performance
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> 
> spark = SparkSession.builder \
>     .appName("PredicatePushdownPattern") \
>     .config("spark.sql.parquet.filterPushdown", "true") \
>     .config("spark.sql.orc.filterPushdown", "true") \
>     .getOrCreate()
> 
> # Create sample data
> data = [(i, f"Name_{i}", i % 10, i * 1.5) for i in range(10000)]
> df = spark.createDataFrame(data, ["id", "name", "category", "value"])
> 
> # Write to Parquet
> parquet_path = "output/predicate_pushdown/parquet"
> df.write.mode("overwrite").parquet(parquet_path)
> 
> print("\n--- Without Predicate Pushdown Optimization ---")
> # This would read all data then filter
> spark.conf.set("spark.sql.parquet.filterPushdown", "false")
> df_no_pushdown = spark.read.parquet(parquet_path)
> result1 = df_no_pushdown.filter(col("category") == 5).count()
> print(f"Filtered records: {result1}")
> df_no_pushdown.filter(col("category") == 5).explain(True)
> 
> print("\n--- With Predicate Pushdown ---")
> # Enable predicate pushdown
> spark.conf.set("spark.sql.parquet.filterPushdown", "true")
> df_with_pushdown = spark.read.parquet(parquet_path)
> result2 = df_with_pushdown.filter(col("category") == 5).count()
> print(f"Filtered records: {result2}")
> df_with_pushdown.filter(col("category") == 5).explain(True)
> 
> # Parquet supports pushdown for:
> # - Equality (=)
> # - Comparison (<, >, <=, >=)
> # - IN, IS NULL, IS NOT NULL
> # - String starts with
> 
> # Example with multiple predicates
> print("\n--- Multiple Predicates ---")
> result3 = df_with_pushdown.filter(
>     (col("category") == 5) & (col("value") > 50)
> ).count()
> print(f"Filtered with multiple predicates: {result3}")
> 
> # ORC example
> orc_path = "output/predicate_pushdown/orc"
> df.write.mode("overwrite").orc(orc_path)
> 
> orc_df = spark.read.orc(orc_path)
> orc_result = orc_df.filter(col("category") == 5).count()
> print(f"\nORC filtered records: {orc_result}")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/columnar_storage_pattern.py
> """
> Columnar Storage Pattern
> - Store data by columns instead of rows
> - Efficient compression
> - Better for analytical queries
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, avg, sum
> import time
> 
> spark = SparkSession.builder \
>     .appName("ColumnarStoragePattern") \
>     .getOrCreate()
> 
> # Generate wide table data
> data = []
> for i in range(100000):
>     row = [i] + [f"col_{j}_{i}" for j in range(10)] + [i * j for j in range(10)]
>     data.append(tuple(row))
> 
> columns = ["id"] + [f"string_col_{j}" for j in range(10)] + [f"numeric_col_{j}" for j in range(10)]
> df = spark.createDataFrame(data, columns)
> 
> # Write to row-based format (CSV)
> csv_path = "output/columnar/csv"
> start = time.time()
> df.write.mode("overwrite").csv(csv_path)
> csv_write_time = time.time() - start
> print(f"CSV write time: {csv_write_time:.2f}s")
> 
> # Write to columnar format (Parquet)
> parquet_path = "output/columnar/parquet"
> start

#### _User_

> Continue your response exactly where you left off:

#### _Assistant_

> ```python storage_patterns/columnar_storage_pattern.py
> """
> Columnar Storage Pattern
> - Store data by columns instead of rows
> - Efficient compression
> - Better for analytical queries
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, avg, sum
> import time
> 
> spark = SparkSession.builder \
>     .appName("ColumnarStoragePattern") \
>     .getOrCreate()
> 
> # Generate wide table data
> data = []
> for i in range(100000):
>     row = [i] + [f"col_{j}_{i}" for j in range(10)] + [i * j for j in range(10)]
>     data.append(tuple(row))
> 
> columns = ["id"] + [f"string_col_{j}" for j in range(10)] + [f"numeric_col_{j}" for j in range(10)]
> df = spark.createDataFrame(data, columns)
> 
> # Write to row-based format (CSV)
> csv_path = "output/columnar/csv"
> start = time.time()
> df.write.mode("overwrite").csv(csv_path)
> csv_write_time = time.time() - start
> print(f"CSV write time: {csv_write_time:.2f}s")
> 
> # Write to columnar format (Parquet)
> parquet_path = "output/columnar/parquet"
> start = time.time()
> df.write.mode("overwrite").parquet(parquet_path)
> parquet_write_time = time.time() - start
> print(f"Parquet write time: {parquet_write_time:.2f}s")
> 
> # Analytical query - select only few columns
> print("\n--- Columnar vs Row-based for SELECT specific columns ---")
> 
> # CSV (row-based) - must read all columns
> start = time.time()
> csv_df = spark.read.csv(csv_path)
> result1 = csv_df.select("_c0", "_c11", "_c12").count()
> csv_read_time = time.time() - start
> print(f"CSV read time (3 columns): {csv_read_time:.2f}s")
> 
> # Parquet (columnar) - reads only needed columns
> start = time.time()
> parquet_df = spark.read.parquet(parquet_path)
> result2 = parquet_df.select("id", "numeric_col_0", "numeric_col_1").count()
> parquet_read_time = time.time() - start
> print(f"Parquet read time (3 columns): {parquet_read_time:.2f}s")
> 
> # Aggregation query - columnar advantage
> print("\n--- Aggregation Performance ---")
> start = time.time()
> parquet_agg = parquet_df.select(
>     avg("numeric_col_0").alias("avg_0"),
>     sum("numeric_col_1").alias("sum_1")
> ).collect()
> parquet_agg_time = time.time() - start
> print(f"Parquet aggregation time: {parquet_agg_time:.2f}s")
> 
> # Benefits of columnar storage:
> # 1. Better compression (similar values together)
> # 2. Column pruning (read only needed columns)
> # 3. Vectorized processing
> # 4. Encoding schemes (dictionary, run-length)
> 
> # Write ORC (another columnar format)
> orc_path = "output/columnar/orc"
> df.write.mode("overwrite").orc(orc_path)
> 
> orc_df = spark.read.orc(orc_path)
> print(f"\nORC record count: {orc_df.count()}")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/row_based_storage_pattern.py
> """
> Row-based Storage Pattern
> - Store complete rows together
> - Better for transactional workloads
> - Efficient for full-row reads
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import time
> 
> spark = SparkSession.builder \
>     .appName("RowBasedStoragePattern") \
>     .getOrCreate()
> 
> # Create transactional-style data
> data = []
> for i in range(10000):
>     data.append((
>         i,
>         f"User_{i}",
>         f"user_{i}@email.com",
>         f"Address_{i}",
>         f"City_{i}",
>         f"State_{i}",
>         i * 100
>     ))
> 
> columns = ["id", "name", "email", "address", "city", "state", "balance"]
> df = spark.createDataFrame(data, columns)
> 
> # Row-based formats: CSV, JSON, Avro
> csv_path = "output/rowbased/csv"
> json_path = "output/rowbased/json"
> avro_path = "output/rowbased/avro"
> 
> # Write to row-based formats
> df.write.mode("overwrite").csv(csv_path)
> df.write.mode("overwrite").json(json_path)
> df.write.mode("overwrite").format("avro").save(avro_path)
> 
> print("--- Row-based Storage - Full Row Retrieval ---")
> 
> # Scenario 1: Fetch complete user records (row-based is efficient)
> start = time.time()
> csv_df = spark.read.option("header", "false").csv(csv_path)
> user_records = csv_df.filter(col("_c0") == "5000").collect()
> csv_time = time.time() - start
> print(f"CSV full row read time: {csv_time:.4f}s")
> 
> start = time.time()
> json_df = spark.read.json(json_path)
> user_records = json_df.filter(col("id") == 5000).collect()
> json_time = time.time() - start
> print(f"JSON full row read time: {json_time:.4f}s")
> 
> start = time.time()
> avro_df = spark.read.format("avro").load(avro_path)
> user_records = avro_df.filter(col("id") == 5000).collect()
> avro_time = time.time() - start
> print(f"Avro full row read time: {avro_time:.4f}s")
> 
> # Row-based storage is better for:
> # 1. OLTP workloads (insert, update, delete)
> # 2. Reading entire records
> # 3. Write-heavy applications
> # 4. Small, frequent queries
> 
> # Compare with columnar for full row access
> parquet_path = "output/rowbased/parquet"
> df.write.mode("overwrite").parquet(parquet_path)
> 
> start = time.time()
> parquet_df = spark.read.parquet(parquet_path)
> parquet_records = parquet_df.filter(col("id") == 5000).collect()
> parquet_time = time.time() - start
> print(f"Parquet full row read time: {parquet_time:.4f}s")
> 
> print("\n--- Use Cases for Row-based Storage ---")
> print("1. Transactional systems (CRUD operations)")
> print("2. Streaming/event data")
> print("3. Log files")
> print("4. Small datasets with frequent updates")
> print("5. When most queries need all columns")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/compression_pattern.py
> """
> Compression Pattern
> - Reduce storage footprint
> - Trade CPU for I/O
> - Various codec options
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, rand
> import os
> 
> spark = SparkSession.builder \
>     .appName("CompressionPattern") \
>     .getOrCreate()
> 
> # Generate compressible data (repeated patterns)
> data = []
> for i in range(100000):
>     data.append((
>         i,
>         f"Category_{i % 100}",  # High repetition
>         f"Product_{i % 500}",
>         i % 1000,
>         "A" * 50  # Highly compressible
>     ))
> 
> df = spark.createDataFrame(data, ["id", "category", "product", "quantity", "description"])
> 
> print("--- Compression Codec Comparison ---\n")
> 
> # Parquet compression codecs
> codecs = ["uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"]
> 
> results = []
> 
> for codec in codecs:
>     try:
>         path = f"output/compression/parquet_{codec}"
>         df.write.mode("overwrite") \
>             .option("compression", codec) \
>             .parquet(path)
>         
>         # Get directory size
>         size = sum(
>             os.path.getsize(os.path.join(dirpath, filename))
>             for dirpath, _, filenames in os.walk(path)
>             for filename in filenames
>         )
>         
>         # Read and verify
>         read_df = spark.read.parquet(path)
>         count = read_df.count()
>         
>         results.append((codec, size, count))
>         print(f"{codec:15} - Size: {size:10} bytes, Records: {count}")
>     except Exception as e:
>         print(f"{codec:15} - Not supported or error: {str(e)}")
> 
> # ORC compression codecs
> print("\n--- ORC Compression ---")
> orc_codecs = ["none", "snappy", "zlib", "lzo"]
> 
> for codec in orc_codecs:
>     try:
>         path = f"output/compression/orc_{codec}"
>         df.write.mode("overwrite") \
>             .option("compression", codec) \
>             .orc(path)
>         
>         size = sum(
>             os.path.getsize(os.path.join(dirpath, filename))
>             for dirpath, _, filenames in os.walk(path)
>             for filename in filenames
>         )
>         
>         print(f"{codec:15} - Size: {size:10} bytes")
>     except Exception as e:
>         print(f"{codec:15} - Not supported: {str(e)}")
> 
> # CSV with compression
> print("\n--- CSV Compression ---")
> csv_path = "output/compression/csv_gzip"
> df.write.mode("overwrite") \
>     .option("compression", "gzip") \
>     .csv(csv_path)
> 
> csv_size = sum(
>     os.path.getsize(os.path.join(dirpath, filename))
>     for dirpath, _, filenames in os.walk(csv_path)
>     for filename in filenames
> )
> print(f"CSV with gzip - Size: {csv_size} bytes")
> 
> # JSON with compression
> json_path = "output/compression/json_gzip"
> df.write.mode("overwrite") \
>     .option("compression", "gzip") \
>     .json(json_path)
> 
> json_size = sum(
>     os.path.getsize(os.path.join(dirpath, filename))
>     for dirpath, _, filenames in os.walk(json_path)
>     for filename in filenames
> )
> print(f"JSON with gzip - Size: {json_size} bytes")
> 
> print("\n--- Compression Recommendations ---")
> print("Snappy: Fast compression/decompression, moderate compression ratio")
> print("Gzip: Better compression ratio, slower than Snappy")
> print("LZ4: Very fast, good for streaming")
> print("Zstd: Best compression ratio, configurable levels")
> print("Brotli: Excellent compression, slower")
> 
> spark.stop()
> ```
> 
> ```python storage_patterns/partitioned_storage_pattern.py
> """
> Partitioned Storage Pattern
> - Organize data by partition columns
> - Enable partition pruning
> - Improve query performance
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, year, month, dayofmonth, current_date
> from datetime import datetime, timedelta
> 
> spark = SparkSession.builder \
>     .appName("PartitionedStoragePattern") \
>     .getOrCreate()
> 
> # Generate time-series data with multiple dimensions
> base_date = datetime(2024, 1, 1)
> data = []
> for i in range(50000):
>     current_date = base_date + timedelta(days=i % 365)
>     data.append((
>         i,
>         current_date,
>         f"Country_{i % 5}",
>         f"Product_{i % 20}",
>         i * 1.5
>     ))
> 
> df = spark.createDataFrame(data, ["id", "event_date", "country", "product", "amount"])
> 
> # Add partition columns
> from pyspark.sql.functions import year, month, dayofmonth
> 
> df = df.withColumn("year", year(col("event_date"))) \
>        .withColumn("month", month(col("event_date"))) \
>        .withColumn("day", dayofmonth(col("event_date")))
> 
> # Pattern 1: Single-level partitioning
> print("--- Single-level Partitioning by Country ---")
> single_partition_path = "output/partitioned/single_level"
> df.write.mode("overwrite") \
>     .partitionBy("country") \
>     .parquet(single_partition_path)
> 
> # Query with partition pruning
> result1 = spark.read.parquet(single_partition_path) \
>     .filter(col("country") == "Country_0") \
>     .count()
> print(f"Records for Country_0: {result1}")
> 
> # Pattern 2: Multi-level partitioning
> print("\n--- Multi-level Partitioning by Year/Month ---")
> multi_partition_path = "output/partitioned/multi_level"
> df.write.mode("overwrite") \
>     .partitionBy("year", "month") \
>     .parquet(multi_partition_path)
> 
> # Query benefits from hierarchical pruning
> result2 = spark.read.parquet(multi_partition_path) \
>     .filter((col("year") == 2024) & (col("month") == 1)) \
>     .count()
> print(f"Records for Jan 2024: {result2}")
> 
> # Pattern 3: Multi-dimensional partitioning
> print("\n--- Multi-dimensional Partitioning ---")
> multi_dim_path = "output/partitioned/multi_dimension"
> df.write.mode("overwrite") \
>     .partitionBy("country", "year", "month") \
>     .parquet(multi_dim_path)
> 
> result3 = spark.read.parquet(multi_dim_path) \
>     .filter(
>         (col("country") == "Country_0") & 
>         (col("year") == 2024) & 
>         (col("month") == 1)
>     ).count()
> print(f"Records for Country_0 in Jan 2024: {result3}")
> 
> # Pattern 4: Bucket partitioning (Hive-style)
> print("\n--- Bucketing Pattern ---")
> bucketed_path = "output/partitioned/bucketed"
> 
> df.write.mode("overwrite") \
>     .bucketBy(10, "id") \
>     .sortBy("id") \
>     .option("path", bucketed_path) \
>     .saveAsTable("bucketed_table")
> 
> # Pattern 5: Dynamic partitioning
> print("\n--- Dynamic Partitioning ---")
> spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
> 
> # Add new data for a specific partition
> new_data = [(50001, datetime(2024, 2, 1), "Country_0", "Product_0", 100.0)]
> new_df = spark.createDataFrame(new_data, ["id", "event_date", "country", "product", "amount"])
> new_df = new_df.withColumn("year", year(col("event_date"))) \
>                .withColumn("month", month(col("event_date")))
> 
> new_df.write.mode("overwrite") \
>     .partitionBy("country", "year", "month") \
>     .parquet(multi_dim_path)
> 
> # Pattern 6: Partition discovery
> print("\n--- Partition Discovery ---")
> discovered_df = spark.read.parquet(multi_partition_path)
> print("Discovered partitions:")
> discovered_df.select("year", "month").distinct().orderBy("year", "month").show()
> 
> # Best practices
> print("\n--- Partitioning Best Practices ---")
> print("1. Partition on columns frequently used in WHERE clauses")
> print("2. Avoid high cardinality columns (creates too many partitions)")
> print("3. Avoid low cardinality columns (doesn't provide enough pruning)")
> print("4. Target partition size: 128MB - 1GB")
> print("5. Limit partition depth (2-3 levels)")
> print("6. Consider time-based partitioning for time-series data")
> 
> # Show partition statistics
> print("\n--- Partition Statistics ---")
> spark.sql(f"""
>     SELECT year, month, COUNT(*) as record_count
>     FROM parquet.`{multi_partition_path}`
>     GROUP BY year, month
>     ORDER BY year, month
> """).show()
> 
> spark.stop()
> ```
> 
> These comprehensive examples demonstrate all the major storage patterns in PySpark. Each pattern includes:
> 
> - **Practical implementations** with real-world scenarios
> - **Configuration options** for optimization
> - **Performance considerations** and best practices
> - **Use case descriptions** explaining when to use each pattern
> 
> You can use the Apply Button to save these files to your project, or switch to Agent Mode if you'd like me to automatically create and organize them into your workspace!