### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:14:13 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> ### **II. Data Loading & Saving (I/O)**
> *   Reading Various Data Formats (CSV, JSON, Parquet, ORC, JDBC, Delta, Iceberg)
> *   Writing to Various Data Formats (modes: overwrite, append, ignore, errorIfExists)
> *   Schema Provisioning/Enforcement on Read
> *   Partitioned Writes
> *   Bucketing on Write
> *   Reading/Writing from Cloud Storage (S3, ADLS, GCS)
> *   Reading from Streaming Sources (Kafka, FileStream, Rate)
> *   Writing to Streaming Sinks (Console, FileStream, Kafka, ForeachBatch)
> ```

#### _Assistant_

> # PySpark Data Loading & Saving Patterns
> 
> Here's a comprehensive guide with examples for each pattern:
> 
> ## 1. Reading Various Data Formats
> 
> ```python reading_formats.py
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> 
> spark = SparkSession.builder \
>     .appName("Reading Data Formats") \
>     .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0,org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.3.1") \
>     .getOrCreate()
> 
> # ============================================
> # 1. Reading CSV
> # ============================================
> csv_df = spark.read \
>     .option("header", "true") \
>     .option("inferSchema", "true") \
>     .option("delimiter", ",") \
>     .csv("data/input.csv")
> 
> # ============================================
> # 2. Reading JSON
> # ============================================
> json_df = spark.read \
>     .option("multiLine", "true") \
>     .json("data/input.json")
> 
> # ============================================
> # 3. Reading Parquet
> # ============================================
> parquet_df = spark.read \
>     .parquet("data/input.parquet")
> 
> # ============================================
> # 4. Reading ORC
> # ============================================
> orc_df = spark.read \
>     .orc("data/input.orc")
> 
> # ============================================
> # 5. Reading from JDBC
> # ============================================
> jdbc_df = spark.read \
>     .format("jdbc") \
>     .option("url", "jdbc:postgresql://localhost:5432/mydb") \
>     .option("dbtable", "employees") \
>     .option("user", "username") \
>     .option("password", "password") \
>     .option("driver", "org.postgresql.Driver") \
>     .load()
> 
> # JDBC with custom query
> jdbc_query_df = spark.read \
>     .format("jdbc") \
>     .option("url", "jdbc:postgresql://localhost:5432/mydb") \
>     .option("query", "SELECT * FROM employees WHERE salary > 50000") \
>     .option("user", "username") \
>     .option("password", "password") \
>     .load()
> 
> # ============================================
> # 6. Reading Delta Lake
> # ============================================
> delta_df = spark.read \
>     .format("delta") \
>     .load("data/delta_table")
> 
> # Reading specific version (time travel)
> delta_version_df = spark.read \
>     .format("delta") \
>     .option("versionAsOf", 5) \
>     .load("data/delta_table")
> 
> # Reading as of timestamp
> delta_timestamp_df = spark.read \
>     .format("delta") \
>     .option("timestampAsOf", "2024-01-01 00:00:00") \
>     .load("data/delta_table")
> 
> # ============================================
> # 7. Reading Apache Iceberg
> # ============================================
> iceberg_df = spark.read \
>     .format("iceberg") \
>     .load("catalog.db.iceberg_table")
> 
> # Time travel with Iceberg
> iceberg_snapshot_df = spark.read \
>     .format("iceberg") \
>     .option("snapshot-id", 123456789) \
>     .load("catalog.db.iceberg_table")
> 
> csv_df.show(5)
> json_df.show(5)
> ```
> 
> ## 2. Writing to Various Data Formats
> 
> ```python writing_formats.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> 
> spark = SparkSession.builder \
>     .appName("Writing Data Formats") \
>     .getOrCreate()
> 
> # Sample DataFrame
> data = [
>     (1, "John", 28, "Engineering", 75000),
>     (2, "Jane", 32, "Marketing", 68000),
>     (3, "Bob", 45, "Engineering", 92000),
>     (4, "Alice", 29, "Sales", 61000)
> ]
> df = spark.createDataFrame(data, ["id", "name", "age", "department", "salary"])
> 
> # ============================================
> # Write Modes
> # ============================================
> # 1. overwrite: Deletes existing data and writes new data
> # 2. append: Adds new data to existing data
> # 3. ignore: Writes only if target doesn't exist (silently skip if exists)
> # 4. errorIfExists (default): Throws error if target already exists
> 
> # ============================================
> # Writing CSV
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .option("header", "true") \
>     .option("delimiter", ",") \
>     .csv("output/csv_data")
> 
> # ============================================
> # Writing JSON
> # ============================================
> df.write \
>     .mode("append") \
>     .json("output/json_data")
> 
> # ============================================
> # Writing Parquet
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .option("compression", "snappy") \
>     .parquet("output/parquet_data")
> 
> # ============================================
> # Writing ORC
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .orc("output/orc_data")
> 
> # ============================================
> # Writing to JDBC
> # ============================================
> df.write \
>     .format("jdbc") \
>     .mode("append") \
>     .option("url", "jdbc:postgresql://localhost:5432/mydb") \
>     .option("dbtable", "employees") \
>     .option("user", "username") \
>     .option("password", "password") \
>     .option("driver", "org.postgresql.Driver") \
>     .save()
> 
> # ============================================
> # Writing Delta Lake
> # ============================================
> df.write \
>     .format("delta") \
>     .mode("overwrite") \
>     .save("output/delta_table")
> 
> # Delta with merge/upsert capability
> from delta.tables import DeltaTable
> 
> if DeltaTable.isDeltaTable(spark, "output/delta_table"):
>     deltaTable = DeltaTable.forPath(spark, "output/delta_table")
>     
>     deltaTable.alias("target").merge(
>         df.alias("source"),
>         "target.id = source.id"
>     ).whenMatchedUpdateAll() \
>      .whenNotMatchedInsertAll() \
>      .execute()
> 
> # ============================================
> # Writing Iceberg
> # ============================================
> df.write \
>     .format("iceberg") \
>     .mode("overwrite") \
>     .save("catalog.db.iceberg_table")
> 
> # ============================================
> # Demonstrating different modes
> # ============================================
> # Mode: ignore (no error if path exists)
> df.write \
>     .mode("ignore") \
>     .parquet("output/ignore_mode")
> 
> # Mode: errorIfExists (default - will throw error if exists)
> try:
>     df.write \
>         .mode("errorIfExists") \
>         .parquet("output/error_mode")
> except Exception as e:
>     print(f"Error: {e}")
> ```
> 
> ## 3. Schema Provisioning/Enforcement on Read
> 
> ```python schema_enforcement.py
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
> 
> spark = SparkSession.builder \
>     .appName("Schema Provisioning") \
>     .getOrCreate()
> 
> # ============================================
> # Define Explicit Schema
> # ============================================
> employee_schema = StructType([
>     StructField("id", IntegerType(), nullable=False),
>     StructField("name", StringType(), nullable=False),
>     StructField("age", IntegerType(), nullable=True),
>     StructField("department", StringType(), nullable=True),
>     StructField("salary", DoubleType(), nullable=True),
>     StructField("hire_date", TimestampType(), nullable=True)
> ])
> 
> # ============================================
> # Reading CSV with Schema Enforcement
> # ============================================
> csv_with_schema = spark.read \
>     .schema(employee_schema) \
>     .option("header", "true") \
>     .option("mode", "PERMISSIVE") \
>     .csv("data/employees.csv")
> 
> # ============================================
> # Schema Modes
> # ============================================
> # PERMISSIVE (default): Sets null for corrupt records
> permissive_df = spark.read \
>     .schema(employee_schema) \
>     .option("mode", "PERMISSIVE") \
>     .option("columnNameOfCorruptRecord", "_corrupt_record") \
>     .csv("data/employees.csv")
> 
> # DROPMALFORMED: Drops rows with corrupt records
> dropmalformed_df = spark.read \
>     .schema(employee_schema) \
>     .option("mode", "DROPMALFORMED") \
>     .csv("data/employees.csv")
> 
> # FAILFAST: Throws exception on corrupt records
> try:
>     failfast_df = spark.read \
>         .schema(employee_schema) \
>         .option("mode", "FAILFAST") \
>         .csv("data/employees.csv")
> except Exception as e:
>     print(f"Schema mismatch error: {e}")
> 
> # ============================================
> # Reading JSON with Schema
> # ============================================
> json_with_schema = spark.read \
>     .schema(employee_schema) \
>     .json("data/employees.json")
> 
> # ============================================
> # Reading Parquet with Schema Merging
> # ============================================
> parquet_merged_schema = spark.read \
>     .option("mergeSchema", "true") \
>     .parquet("data/employees_partitioned")
> 
> # ============================================
> # Schema Evolution with Delta
> # ============================================
> # Allow schema evolution when writing
> from pyspark.sql.functions import lit
> 
> new_data = csv_with_schema.withColumn("email", lit("example@company.com"))
> 
> new_data.write \
>     .format("delta") \
>     .mode("append") \
>     .option("mergeSchema", "true") \
>     .save("output/delta_with_evolution")
> 
> # ============================================
> # DDL String Schema
> # ============================================
> ddl_schema = "id INT, name STRING, age INT, department STRING, salary DOUBLE"
> 
> csv_with_ddl = spark.read \
>     .schema(ddl_schema) \
>     .option("header", "true") \
>     .csv("data/employees.csv")
> 
> csv_with_schema.printSchema()
> permissive_df.show()
> ```
> 
> ## 4. Partitioned Writes
> 
> ```python partitioned_writes.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, year, month, dayofmonth
> 
> spark = SparkSession.builder \
>     .appName("Partitioned Writes") \
>     .getOrCreate()
> 
> # Sample data with dates
> data = [
>     (1, "John", "Engineering", "2023-01-15", 75000),
>     (2, "Jane", "Marketing", "2023-02-20", 68000),
>     (3, "Bob", "Engineering", "2023-01-10", 92000),
>     (4, "Alice", "Sales", "2023-03-05", 61000),
>     (5, "Charlie", "Marketing", "2023-02-14", 71000),
>     (6, "David", "Engineering", "2023-03-22", 85000)
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "department", "hire_date", "salary"])
> df = df.withColumn("hire_date", col("hire_date").cast("date"))
> 
> # ============================================
> # Single Column Partitioning
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .parquet("output/partitioned_by_dept")
> 
> # ============================================
> # Multiple Column Partitioning
> # ============================================
> df_with_date_parts = df \
>     .withColumn("year", year(col("hire_date"))) \
>     .withColumn("month", month(col("hire_date")))
> 
> df_with_date_parts.write \
>     .mode("overwrite") \
>     .partitionBy("department", "year", "month") \
>     .parquet("output/partitioned_multi")
> 
> # ============================================
> # Partitioned Write with Different Formats
> # ============================================
> # Parquet with partitioning
> df.write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .option("compression", "snappy") \
>     .parquet("output/partitioned_parquet")
> 
> # ORC with partitioning
> df.write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .orc("output/partitioned_orc")
> 
> # Delta with partitioning
> df.write \
>     .format("delta") \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .save("output/partitioned_delta")
> 
> # ============================================
> # Dynamic Partitioning
> # ============================================
> # Control max partitions written
> spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
> 
> df.write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .parquet("output/dynamic_partitioned")
> 
> # ============================================
> # Reading Partitioned Data
> # ============================================
> # Read with partition pruning
> partitioned_df = spark.read \
>     .parquet("output/partitioned_by_dept")
> 
> # Filter on partition column (partition pruning)
> engineering_df = partitioned_df.filter(col("department") == "Engineering")
> 
> # ============================================
> # Repartition before writing
> # ============================================
> df.repartition(4, "department").write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .parquet("output/repartitioned_write")
> 
> # ============================================
> # Coalesce to reduce files per partition
> # ============================================
> df.coalesce(1).write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .parquet("output/coalesced_partitioned")
> 
> print("Partitioned data written successfully")
> partitioned_df.show()
> ```
> 
> ## 5. Bucketing on Write
> 
> ```python bucketing_writes.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> 
> spark = SparkSession.builder \
>     .appName("Bucketing on Write") \
>     .getOrCreate()
> 
> # Sample data
> data = [
>     (1, "John", 28, "Engineering", 75000),
>     (2, "Jane", 32, "Marketing", 68000),
>     (3, "Bob", 45, "Engineering", 92000),
>     (4, "Alice", 29, "Sales", 61000),
>     (5, "Charlie", 35, "Marketing", 71000),
>     (6, "David", 41, "Engineering", 85000),
>     (7, "Eve", 27, "Sales", 59000),
>     (8, "Frank", 38, "Engineering", 88000)
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "age", "department", "salary"])
> 
> # ============================================
> # Simple Bucketing
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .bucketBy(4, "department") \
>     .sortBy("salary") \
>     .saveAsTable("bucketed_employees")
> 
> # ============================================
> # Bucketing with Parquet Format
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .bucketBy(4, "department") \
>     .sortBy("salary") \
>     .option("path", "output/bucketed_parquet") \
>     .saveAsTable("bucketed_employees_parquet")
> 
> # ============================================
> # Bucketing on Multiple Columns
> # ============================================
> df.write \
>     .mode("overwrite") \
>     .bucketBy(8, "department", "age") \
>     .sortBy("salary", "name") \
>     .saveAsTable("multi_bucket_employees")
> 
> # ============================================
> # Bucketing with Partitioning
> # ============================================
> # Add year column for partitioning
> from pyspark.sql.functions import lit
> 
> df_with_year = df.withColumn("year", lit(2023))
> 
> df_with_year.write \
>     .mode("overwrite") \
>     .partitionBy("year") \
>     .bucketBy(4, "department") \
>     .sortBy("salary") \
>     .saveAsTable("partitioned_bucketed_employees")
> 
> # ============================================
> # Reading Bucketed Tables
> # ============================================
> bucketed_df = spark.table("bucketed_employees")
> 
> # Bucket pruning happens automatically when filtering on bucket column
> dept_df = bucketed_df.filter(col("department") == "Engineering")
> 
> # ============================================
> # Benefits of Bucketing for Joins
> # ============================================
> # Create another bucketed table with same bucketing scheme
> dept_data = [
>     ("Engineering", "Tech", 50),
>     ("Marketing", "Business", 30),
>     ("Sales", "Business", 40)
> ]
> dept_df_data = spark.createDataFrame(dept_data, ["department", "division", "headcount"])
> 
> dept_df_data.write \
>     .mode("overwrite") \
>     .bucketBy(4, "department") \
>     .sortBy("headcount") \
>     .saveAsTable("bucketed_departments")
> 
> # Join bucketed tables (no shuffle needed!)
> employees_bucketed = spark.table("bucketed_employees")
> departments_bucketed = spark.table("bucketed_departments")
> 
> joined_df = employees_bucketed.join(
>     departments_bucketed,
>     "department",
>     "inner"
> )
> 
> # ============================================
> # Important Notes
> # ============================================
> print("Bucketing Notes:")
> print("1. Bucketing only works with saveAsTable()")
> print("2. Number of buckets should be power of 2 or multiple of partition count")
> print("3. Bucketing reduces shuffle in joins when both tables are bucketed on join key")
> print("4. Sorted buckets improve range queries and aggregations")
> 
> joined_df.show()
> ```
> 
> ## 6. Reading/Writing from Cloud Storage
> 
> ```python cloud_storage_io.py
> from pyspark.sql import SparkSession
> 
> # ============================================
> # AWS S3 Configuration
> # ============================================
> spark_s3 = SparkSession.builder \
>     .appName("S3 Data Access") \
>     .config("spark.hadoop.fs.s3a.access.key", "YOUR_ACCESS_KEY") \
>     .config("spark.hadoop.fs.s3a.secret.key", "YOUR_SECRET_KEY") \
>     .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
>     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
>     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
>     .getOrCreate()
> 
> # ============================================
> # Reading from S3
> # ============================================
> # Reading CSV from S3
> s3_csv_df = spark_s3.read \
>     .option("header", "true") \
>     .csv("s3a://my-bucket/data/input.csv")
> 
> # Reading Parquet from S3
> s3_parquet_df = spark_s3.read \
>     .parquet("s3a://my-bucket/data/parquet_data/")
> 
> # Reading Delta from S3
> s3_delta_df = spark_s3.read \
>     .format("delta") \
>     .load("s3a://my-bucket/data/delta_table/")
> 
> # ============================================
> # Writing to S3
> # ============================================
> # Writing Parquet to S3
> s3_csv_df.write \
>     .mode("overwrite") \
>     .parquet("s3a://my-bucket/output/parquet_data/")
> 
> # Writing partitioned data to S3
> s3_csv_df.write \
>     .mode("overwrite") \
>     .partitionBy("department") \
>     .parquet("s3a://my-bucket/output/partitioned_data/")
> 
> # ============================================
> # Azure Data Lake Storage (ADLS) Gen2
> # ============================================
> spark_adls = SparkSession.builder \
>     .appName("ADLS Data Access") \
>     .config("spark.hadoop.fs.azure.account.auth.type", "OAuth") \
>     .config("spark.hadoop.fs.azure.account.oauth.provider.type", 
>             "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider") \
>     .config("spark.hadoop.fs.azure.account.oauth2.client.id", "YOUR_CLIENT_ID") \
>     .config("spark.hadoop.fs.azure.account.oauth2.client.secret", "YOUR_CLIENT_SECRET") \
>     .config("spark.hadoop.fs.azure.account.oauth2.client.endpoint", 
>             "https://login.microsoftonline.com/YOUR_TENANT_ID/oauth2/token") \
>     .getOrCreate()
> 
> # Alternative: Using account key
> spark_adls_key = SparkSession.builder \
>     .appName("ADLS with Key") \
>     .config("spark.hadoop.fs.azure.account.key.mystorageaccount.dfs.core.windows.net", 
>             "YOUR_ACCOUNT_KEY") \
>     .getOrCreate()
> 
> # ============================================
> # Reading from ADLS Gen2
> # ============================================
> # Reading CSV from ADLS
> adls_csv_df = spark_adls.read \
>     .option("header", "true") \
>     .csv("abfss://container@mystorageaccount.dfs.core.windows.net/data/input.csv")
> 
> # Reading Parquet from ADLS
> adls_parquet_df = spark_adls.read \
>     .parquet("abfss://container@mystorageaccount.dfs.core.windows.net/data/parquet/")
> 
> # ============================================
> # Writing to ADLS Gen2
> # ============================================
> adls_csv_df.write \
>     .mode("overwrite") \
>     .parquet("abfss://container@mystorageaccount.dfs.core.windows.net/output/parquet/")
> 
> # ============================================
> # Google Cloud Storage (GCS)
> # ============================================
> spark_gcs = SparkSession.builder \
>     .appName("GCS Data Access") \
>     .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
>     .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", 
>             "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
>     .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
>     .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", 
>             "/path/to/keyfile.json") \
>     .getOrCreate()
> 
> # ============================================
> # Reading from GCS
> # ============================================
> # Reading CSV from GCS
> gcs_csv_df = spark_gcs.read \
>     .option("header", "true") \
>     .csv("gs://my-bucket/data/input.csv")
> 
> # Reading Parquet from GCS
> gcs_parquet_df = spark_gcs.read \
>     .parquet("gs://my-bucket/data/parquet_data/")
> 
> # Reading Delta from GCS
> gcs_delta_df = spark_gcs.read \
>     .format("delta") \
>     .load("gs://my-bucket/data/delta_table/")
> 
> # ============================================
> # Writing to GCS
> # ============================================
> gcs_csv_df.write \
>     .mode("overwrite") \
>     .parquet("gs://my-bucket/output/parquet_data/")
> 
> # Writing with partitioning
> gcs_csv_df.write \
>     .mode("overwrite") \
>     .partitionBy("year", "month") \
>     .parquet("gs://my-bucket/output/partitioned_data/")
> 
> # ============================================
> # Cloud Storage Best Practices
> # ============================================
> # Use repartition/coalesce to optimize file sizes
> optimized_df = gcs_csv_df.repartition(10)
> 
> optimized_df.write \
>     .mode("overwrite") \
>     .parquet("gs://my-bucket/output/optimized_data/")
> 
> # Using credentials from environment or IAM roles (preferred)
> spark_iam = SparkSession.builder \
>     .appName("Cloud Storage with IAM") \
>     .getOrCreate()
> 
> # For S3 with IAM roles (no keys needed)
> iam_s3_df = spark_iam.read \
>     .parquet("s3a://my-bucket/data/")
> 
> # ============================================
> # Multi-cloud Data Transfer
> # ============================================
> # Read from S3 and write to GCS
> s3_to_gcs_df = spark_s3.read.parquet("s3a://source-bucket/data/")
> s3_to_gcs_df.write.mode("overwrite").parquet("gs://dest-bucket/data/")
> 
> print("Cloud storage operations completed")
> ```
> 
> ## 7. Reading from Streaming Sources
> 
> ```python streaming_sources.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, from_json, window, current_timestamp
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
> 
> spark = SparkSession.builder \
>     .appName("Streaming Sources") \
>     .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
>     .getOrCreate()
> 
> spark.sparkContext.setLogLevel("WARN")
> 
> # ============================================
> # 1. Kafka Streaming Source
> # ============================================
> # Define schema for Kafka messages
> event_schema = StructType([
>     StructField("user_id", IntegerType(), False),
>     StructField("event_type", StringType(), False),
>     StructField("product_id", IntegerType(), True),
>     StructField("amount", DoubleType(), True),
>     StructField("timestamp", TimestampType(), True)
> ])
> 
> # Read from Kafka
> kafka_stream_df = spark.readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "events") \
>     .option("startingOffsets", "earliest") \
>     .load()
> 
> # Parse Kafka messages
> parsed_kafka_df = kafka_stream_df.select(
>     col("key").cast("string"),
>     from_json(col("value").cast("string"), event_schema).alias("data"),
>     col("topic"),
>     col("partition"),
>     col("offset"),
>     col("timestamp")
> ).select("key", "data.*", "topic", "partition", "offset", "timestamp")
> 
> # Multiple topics
> multi_topic_df = spark.readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "topic1,topic2,topic3") \
>     .load()
> 
> # Topic pattern
> pattern_topic_df = spark.readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribePattern", "events-.*") \
>     .load()
> 
> # Kafka with specific offsets
> specific_offset_df = spark.readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "events") \
>     .option("startingOffsets", """{"events":{"0":23,"1":45}}""") \
>     .load()
> 
> # ============================================
> # 2. File Stream Source
> # ============================================
> # Read CSV files as stream
> file_stream_csv = spark.readStream \
>     .format("csv") \
>     .option("header", "true") \
>     .option("maxFilesPerTrigger", 1) \
>     .schema(event_schema) \
>     .load("data/streaming_input/")
> 
> # Read JSON files as stream
> file_stream_json = spark.readStream \
>     .format("json") \
>     .option("maxFilesPerTrigger", 2) \
>     .schema(event_schema) \
>     .load("data/json_stream/")
> 
> # Read Parquet files as stream
> file_stream_parquet = spark.readStream \
>     .format("parquet") \
>     .option("maxFilesPerTrigger", 1) \
>     .schema(event_schema) \
>     .load("data/parquet_stream/")
> 
> # ============================================
> # 3. Rate Stream Source (for testing)
> # ============================================
> # Generate test data at specified rate
> rate_stream = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 10) \
>     .option("numPartitions", 2) \
>     .load()
> 
> # Rate source with ramp up
> rate_rampup = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 100) \
>     .option("rampUpTime", "10s") \
>     .load()
> 
> # ============================================
> # 4. Socket Stream Source (for testing/demo)
> # ============================================
> socket_stream = spark.readStream \
>     .format("socket") \
>     .option("host", "localhost") \
>     .option("port", 9999) \
>     .load()
> 
> # ============================================
> # Advanced Kafka Options
> # ============================================
> kafka_advanced = spark.readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "events") \
>     .option("kafka.security.protocol", "SASL_SSL") \
>     .option("kafka.sasl.mechanism", "PLAIN") \
>     .option("kafka.sasl.jaas.config", 
>             'org.apache.kafka.common.security.plain.PlainLoginModule required username="user" password="pass";') \
>     .option("maxOffsetsPerTrigger", 10000) \
>     .option("failOnDataLoss", "false") \
>     .load()
> 
> # ============================================
> # Watermarking and Event Time
> # ============================================
> # Apply watermark for late data handling
> watermarked_stream = parsed_kafka_df \
>     .withWatermark("timestamp", "10 minutes") \
>     .groupBy(
>         window(col("timestamp"), "5 minutes", "1 minute"),
>         col("event_type")
>     ) \
>     .count()
> 
> print("Streaming sources configured")
> ```
> 
> ## 8. Writing to Streaming Sinks
> 
> ```python streaming_sinks.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, to_json, struct, window, current_timestamp
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType
> 
> spark = SparkSession.builder \
>     .appName("Streaming Sinks") \
>     .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
>     .getOrCreate()
> 
> # Create a sample streaming source (rate source for demo)
> input_stream = spark.readStream \
>     .format("rate") \
>     .option("rowsPerSecond", 5) \
>     .load() \
>     .withColumn("event_type", (col("value") % 3).cast("string")) \
>     .withColumn("user_id", (col("value") % 100).cast("int"))
> 
> # ============================================
> # 1. Console Sink (for debugging)
> # ============================================
> console_query = input_stream.writeStream \
>     .format("console") \
>     .outputMode("append") \
>     .option("numRows", 20) \
>     .option("truncate", False) \
>     .trigger(processingTime="5 seconds") \
>     .start()
> 
> # ============================================
> # 2. File Sink (Parquet, JSON, CSV, ORC)
> # ============================================
> # Write to Parquet files
> parquet_query = input_stream.writeStream \
>     .format("parquet") \
>     .outputMode("append") \
>     .option("path

#### _User_

> Continue your response exactly where you left off:

#### _Assistant_

> ```python streaming_sinks.py
> # ... existing code ...
> 
> # Write to Parquet files
> parquet_query = input_stream.writeStream \
>     .format("parquet") \
>     .outputMode("append") \
>     .option("path", "output/streaming_parquet") \
>     .option("checkpointLocation", "checkpoints/parquet_checkpoint") \
>     .trigger(processingTime="10 seconds") \
>     .start()
> 
> # Write to JSON files
> json_query = input_stream.writeStream \
>     .format("json") \
>     .outputMode("append") \
>     .option("path", "output/streaming_json") \
>     .option("checkpointLocation", "checkpoints/json_checkpoint") \
>     .start()
> 
> # Write to CSV files
> csv_query = input_stream.writeStream \
>     .format("csv") \
>     .outputMode("append") \
>     .option("path", "output/streaming_csv") \
>     .option("checkpointLocation", "checkpoints/csv_checkpoint") \
>     .option("header", "true") \
>     .start()
> 
> # Partitioned file write
> partitioned_query = input_stream.writeStream \
>     .format("parquet") \
>     .outputMode("append") \
>     .option("path", "output/streaming_partitioned") \
>     .option("checkpointLocation", "checkpoints/partitioned_checkpoint") \
>     .partitionBy("event_type") \
>     .start()
> 
> # ============================================
> # 3. Kafka Sink
> # ============================================
> # Prepare data for Kafka (key and value must be strings or binary)
> kafka_output = input_stream.select(
>     col("user_id").cast("string").alias("key"),
>     to_json(struct("*")).alias("value")
> )
> 
> kafka_query = kafka_output.writeStream \
>     .format("kafka") \
>     .outputMode("append") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("topic", "output-events") \
>     .option("checkpointLocation", "checkpoints/kafka_checkpoint") \
>     .start()
> 
> # Write to multiple Kafka topics dynamically
> multi_topic_output = input_stream.select(
>     col("event_type").alias("topic"),
>     col("user_id").cast("string").alias("key"),
>     to_json(struct("*")).alias("value")
> )
> 
> multi_kafka_query = multi_topic_output.writeStream \
>     .format("kafka") \
>     .outputMode("append") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("checkpointLocation", "checkpoints/multi_kafka_checkpoint") \
>     .start()
> 
> # ============================================
> # 4. ForeachBatch Sink (Custom Processing)
> # ============================================
> def process_batch(batch_df, batch_id):
>     """
>     Custom function to process each micro-batch
>     """
>     print(f"Processing batch: {batch_id}")
>     
>     # Example 1: Write to multiple destinations
>     batch_df.write.mode("append").parquet(f"output/batch_{batch_id}/parquet")
>     batch_df.write.mode("append").json(f"output/batch_{batch_id}/json")
>     
>     # Example 2: Write to database
>     # batch_df.write.jdbc(url, table, mode="append")
>     
>     # Example 3: Custom business logic
>     count = batch_df.count()
>     print(f"Processed {count} records in batch {batch_id}")
>     
>     # Example 4: Conditional writes
>     if count > 0:
>         high_value = batch_df.filter(col("value") > 50)
>         if high_value.count() > 0:
>             high_value.write.mode("append").parquet("output/high_value_events")
> 
> foreachbatch_query = input_stream.writeStream \
>     .foreachBatch(process_batch) \
>     .outputMode("append") \
>     .option("checkpointLocation", "checkpoints/foreachbatch_checkpoint") \
>     .trigger(processingTime="15 seconds") \
>     .start()
> 
> # ============================================
> # Advanced ForeachBatch: Multi-Sink Pattern
> # ============================================
> def multi_sink_batch(batch_df, batch_id):
>     """
>     Write to multiple sinks with different transformations
>     """
>     # Persist to avoid recomputation
>     batch_df.persist()
>     
>     try:
>         # Sink 1: Write raw data to data lake
>         batch_df.write \
>             .mode("append") \
>             .parquet(f"output/data_lake/events/batch_{batch_id}")
>         
>         # Sink 2: Aggregated data to analytics table
>         aggregated = batch_df.groupBy("event_type").count()
>         aggregated.write \
>             .mode("append") \
>             .format("jdbc") \
>             .option("url", "jdbc:postgresql://localhost:5432/analytics") \
>             .option("dbtable", "event_counts") \
>             .option("user", "user") \
>             .option("password", "pass") \
>             .save()
>         
>         # Sink 3: Alert on specific conditions
>         anomalies = batch_df.filter(col("value") > 90)
>         if anomalies.count() > 0:
>             anomalies.write \
>                 .mode("append") \
>                 .json("output/alerts")
>     
>     finally:
>         batch_df.unpersist()
> 
> multi_sink_query = input_stream.writeStream \
>     .foreachBatch(multi_sink_batch) \
>     .outputMode("append") \
>     .option("checkpointLocation", "checkpoints/multi_sink_checkpoint") \
>     .start()
> 
> # ============================================
> # 5. Foreach Sink (Row-by-Row Processing)
> # ============================================
> class CustomRowWriter:
>     def open(self, partition_id, epoch_id):
>         """Called when starting to process a partition"""
>         print(f"Opening partition {partition_id} for epoch {epoch_id}")
>         return True
>     
>     def process(self, row):
>         """Process each row"""
>         # Example: Send to external API, database, etc.
>         print(f"Processing row: {row}")
>         # Custom logic here (e.g., HTTP POST, queue publish)
>     
>     def close(self, error):
>         """Called when done processing a partition"""
>         if error:
>             print(f"Error occurred: {error}")
>         print("Closing partition")
> 
> foreach_query = input_stream.writeStream \
>     .foreach(CustomRowWriter()) \
>     .outputMode("append") \
>     .option("checkpointLocation", "checkpoints/foreach_checkpoint") \
>     .start()
> 
> # ============================================
> # 6. Memory Sink (for testing)
> # ============================================
> memory_query = input_stream.writeStream \
>     .format("memory") \
>     .queryName("memory_table") \
>     .outputMode("append") \
>     .start()
> 
> # Query the in-memory table
> # spark.sql("SELECT * FROM memory_table").show()
> 
> # ============================================
> # 7. Delta Lake Sink
> # ============================================
> delta_query = input_stream.writeStream \
>     .format("delta") \
>     .outputMode("append") \
>     .option("checkpointLocation", "checkpoints/delta_checkpoint") \
>     .start("output/delta_stream")
> 
> # ============================================
> # Output Modes
> # ============================================
> # Create aggregated stream for different output modes
> aggregated_stream = input_stream \
>     .groupBy(
>         window(col("timestamp"), "1 minute"),
>         col("event_type")
>     ).count()
> 
> # Append mode: Only new rows (default)
> append_query = input_stream.writeStream \
>     .format("parquet") \
>     .outputMode("append") \
>     .option("path", "output/append_mode") \
>     .option("checkpointLocation", "checkpoints/append_checkpoint") \
>     .start()
> 
> # Complete mode: Entire result table
> complete_query = aggregated_stream.writeStream \
>     .format("memory") \
>     .queryName("complete_table") \
>     .outputMode("complete") \
>     .start()
> 
> # Update mode: Only updated rows
> update_query = aggregated_stream.writeStream \
>     .format("parquet") \
>     .outputMode("update") \
>     .option("path", "output/update_mode") \
>     .option("checkpointLocation", "checkpoints/update_checkpoint") \
>     .start()
> 
> # ============================================
> # Trigger Options
> # ============================================
> # Processing time trigger
> processing_time_query = input_stream.writeStream \
>     .format("console") \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> # Once trigger (process available data and stop)
> once_query = input_stream.writeStream \
>     .format("parquet") \
>     .trigger(once=True) \
>     .option("path", "output/once_trigger") \
>     .option("checkpointLocation", "checkpoints/once_checkpoint") \
>     .start()
> 
> # Continuous trigger (low-latency)
> continuous_query = input_stream.writeStream \
>     .format("console") \
>     .trigger(continuous="1 second") \
>     .start()
> 
> # Available now trigger (process all available data)
> available_now_query = input_stream.writeStream \
>     .format("parquet") \
>     .trigger(availableNow=True) \
>     .option("path", "output/available_now") \
>     .option("checkpointLocation", "checkpoints/available_now_checkpoint") \
>     .start()
> 
> # ============================================
> # Query Management
> # ============================================
> # Wait for specific query
> # console_query.awaitTermination()
> 
> # Wait for any query with timeout
> # console_query.awaitTermination(timeout=60)
> 
> # Stop specific query
> # console_query.stop()
> 
> # Get all active queries
> active_queries = spark.streams.active
> print(f"Active queries: {len(active_queries)}")
> 
> # Monitor query progress
> # for query in active_queries:
> #     print(f"Query: {query.name}, Status: {query.status}")
> #     print(f"Recent progress: {query.recentProgress}")
> 
> # Wait for all queries
> # spark.streams.awaitAnyTermination()
> 
> print("Streaming sinks configured and started")
> ```
> 
> ## Complete Example: End-to-End Streaming Pipeline
> 
> ```python complete_streaming_pipeline.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import (
>     col, from_json, to_json, struct, window, 
>     current_timestamp, expr, lit
> )
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
> 
> # ============================================
> # Initialize Spark with necessary packages
> # ============================================
> spark = SparkSession.builder \
>     .appName("Complete Streaming Pipeline") \
>     .config("spark.jars.packages", 
>             "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0," \
>             "io.delta:delta-core_2.12:2.4.0") \
>     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
>     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
>     .getOrCreate()
> 
> # ============================================
> # Define Schema
> # ============================================
> transaction_schema = StructType([
>     StructField("transaction_id", StringType(), False),
>     StructField("user_id", IntegerType(), False),
>     StructField("product_id", IntegerType(), False),
>     StructField("amount", DoubleType(), False),
>     StructField("transaction_time", TimestampType(), False),
>     StructField("status", StringType(), False)
> ])
> 
> # ============================================
> # Read from Kafka
> # ============================================
> kafka_input = spark.readStream \
>     .format("kafka") \
>     .option("kafka.bootstrap.servers", "localhost:9092") \
>     .option("subscribe", "transactions") \
>     .option("startingOffsets", "latest") \
>     .load()
> 
> # Parse Kafka messages
> parsed_transactions = kafka_input.select(
>     col("key").cast("string").alias("kafka_key"),
>     from_json(col("value").cast("string"), transaction_schema).alias("data"),
>     col("timestamp").alias("kafka_timestamp")
> ).select("kafka_key", "data.*", "kafka_timestamp")
> 
> # ============================================
> # Apply Transformations
> # ============================================
> # Add watermark for late data
> watermarked_data = parsed_transactions \
>     .withWatermark("transaction_time", "10 minutes")
> 
> # Filter valid transactions
> valid_transactions = watermarked_data \
>     .filter(col("status") == "completed") \
>     .filter(col("amount") > 0)
> 
> # Aggregate by time window
> windowed_aggregates = valid_transactions \
>     .groupBy(
>         window(col("transaction_time"), "5 minutes", "1 minute"),
>         col("product_id")
>     ).agg(
>         expr("count(*) as transaction_count"),
>         expr("sum(amount) as total_amount"),
>         expr("avg(amount) as avg_amount")
>     )
> 
> # ============================================
> # Write to Multiple Sinks using ForeachBatch
> # ============================================
> def write_to_multiple_sinks(batch_df, batch_id):
>     """Write each batch to multiple destinations"""
>     print(f"Processing batch {batch_id}")
>     
>     if batch_df.count() > 0:
>         # Sink 1: Raw data to Delta Lake
>         batch_df.write \
>             .format("delta") \
>             .mode("append") \
>             .save("output/delta_transactions")
>         
>         # Sink 2: Partitioned Parquet for archival
>         batch_df.write \
>             .mode("append") \
>             .partitionBy("status") \
>             .parquet("output/archived_transactions")
>         
>         # Sink 3: Send back to Kafka for downstream consumers
>         kafka_output = batch_df.select(
>             col("transaction_id").alias("key"),
>             to_json(struct("*")).alias("value")
>         )
>         
>         kafka_output.write \
>             .format("kafka") \
>             .option("kafka.bootstrap.servers", "localhost:9092") \
>             .option("topic", "processed-transactions") \
>             .save()
>         
>         print(f"Batch {batch_id} completed successfully")
> 
> # Start the streaming query
> streaming_query = valid_transactions.writeStream \
>     .foreachBatch(write_to_multiple_sinks) \
>     .outputMode("append") \
>     .option("checkpointLocation", "checkpoints/multi_sink_pipeline") \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> # ============================================
> # Write Aggregates to Console for monitoring
> # ============================================
> console_query = windowed_aggregates.writeStream \
>     .format("console") \
>     .outputMode("update") \
>     .option("truncate", False) \
>     .trigger(processingTime="30 seconds") \
>     .start()
> 
> # ============================================
> # Monitor and Manage
> # ============================================
> print("Streaming pipeline started successfully")
> print(f"Active queries: {[q.name for q in spark.streams.active]}")
> 
> # Wait for termination (remove in production)
> streaming_query.awaitTermination()
> ```
> 
> All patterns are now complete! Each example demonstrates production-ready code for:
> 
> 1. **Reading formats** - CSV, JSON, Parquet, ORC, JDBC, Delta, Iceberg
> 2. **Writing formats** - All modes (overwrite, append, ignore, errorIfExists)
> 3. **Schema enforcement** - Explicit schemas with validation modes
> 4. **Partitioned writes** - Single and multi-column partitioning
> 5. **Bucketing** - Optimizing joins and queries
> 6. **Cloud storage** - S3, ADLS Gen2, GCS integration
> 7. **Streaming sources** - Kafka, File, Rate, Socket
> 8. **Streaming sinks** - Console, File, Kafka, ForeachBatch, Foreach, Memory, Delta
> 
> You can use these as templates for your PySpark applications!