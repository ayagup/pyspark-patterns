### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:19:53 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Schema & Type Handling**
> *   Explicit Schema Definition (StructType)
> *   Schema Evolution (mergeSchema)
> *   Handling Complex Types (ArrayType, MapType, StructType)
> *   Schema Inference
> ```

#### _Assistant_

> I'll create comprehensive PySpark programs demonstrating each schema and type handling pattern.
> 
> ```python schema_definition.py
> """
> Pattern 1: Explicit Schema Definition using StructType
> Demonstrates defining schemas explicitly for better performance and data validation
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.types import (
>     StructType, StructField, StringType, IntegerType, 
>     DoubleType, TimestampType, BooleanType
> )
> from datetime import datetime
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Explicit Schema Definition") \
>     .getOrCreate()
> 
> # Define explicit schema
> user_schema = StructType([
>     StructField("user_id", IntegerType(), nullable=False),
>     StructField("username", StringType(), nullable=False),
>     StructField("email", StringType(), nullable=True),
>     StructField("age", IntegerType(), nullable=True),
>     StructField("balance", DoubleType(), nullable=True),
>     StructField("is_active", BooleanType(), nullable=True),
>     StructField("created_at", TimestampType(), nullable=True)
> ])
> 
> # Sample data
> data = [
>     (1, "john_doe", "john@email.com", 28, 1500.50, True, datetime(2023, 1, 15)),
>     (2, "jane_smith", "jane@email.com", 34, 2300.75, True, datetime(2023, 2, 20)),
>     (3, "bob_wilson", None, 45, 980.25, False, datetime(2023, 3, 10)),
>     (4, "alice_brown", "alice@email.com", 29, 3200.00, True, datetime(2023, 4, 5))
> ]
> 
> # Create DataFrame with explicit schema
> df = spark.createDataFrame(data, schema=user_schema)
> 
> print("=" * 80)
> print("EXPLICIT SCHEMA DEFINITION")
> print("=" * 80)
> 
> # Display schema
> print("\nDefined Schema:")
> df.printSchema()
> 
> # Show data
> print("\nData with Explicit Schema:")
> df.show(truncate=False)
> 
> # Demonstrate type enforcement
> print("\nSchema enforces data types - attempting invalid data will raise error")
> 
> # Read CSV with explicit schema
> csv_data = [
>     "1,john_doe,john@email.com,28,1500.50,true,2023-01-15",
>     "2,jane_smith,jane@email.com,34,2300.75,true,2023-02-20"
> ]
> 
> # Write to temp CSV
> csv_path = "/tmp/users_explicit.csv"
> with open(csv_path, 'w') as f:
>     f.write("user_id,username,email,age,balance,is_active,created_at\n")
>     f.write("\n".join(csv_data))
> 
> # Read with explicit schema
> df_csv = spark.read \
>     .schema(user_schema) \
>     .option("header", "true") \
>     .csv(csv_path)
> 
> print("\nData loaded from CSV with explicit schema:")
> df_csv.show(truncate=False)
> 
> spark.stop()
> ```
> 
> ```python schema_evolution.py
> """
> Pattern 2: Schema Evolution with mergeSchema
> Demonstrates handling schema changes when reading partitioned data
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> import shutil
> import os
> 
> # Initialize Spark Session with schema merge enabled
> spark = SparkSession.builder \
>     .appName("Schema Evolution") \
>     .config("spark.sql.parquet.mergeSchema", "true") \
>     .getOrCreate()
> 
> # Clean up previous runs
> output_path = "/tmp/evolving_data"
> if os.path.exists(output_path):
>     shutil.rmtree(output_path)
> 
> print("=" * 80)
> print("SCHEMA EVOLUTION DEMONSTRATION")
> print("=" * 80)
> 
> # Initial schema - Version 1
> print("\n1. Writing initial dataset (Version 1)...")
> initial_data = [
>     (1, "Product A", 100),
>     (2, "Product B", 150),
>     (3, "Product C", 200)
> ]
> 
> df_v1 = spark.createDataFrame(initial_data, ["id", "name", "quantity"])
> df_v1.write.mode("overwrite").parquet(f"{output_path}/batch1")
> 
> print("Version 1 Schema:")
> df_v1.printSchema()
> df_v1.show()
> 
> # Evolved schema - Version 2 (added price column)
> print("\n2. Writing evolved dataset (Version 2 - added 'price' column)...")
> evolved_data = [
>     (4, "Product D", 120, 29.99),
>     (5, "Product E", 180, 39.99),
>     (6, "Product F", 90, 19.99)
> ]
> 
> df_v2 = spark.createDataFrame(evolved_data, ["id", "name", "quantity", "price"])
> df_v2.write.mode("append").parquet(f"{output_path}/batch2")
> 
> print("Version 2 Schema:")
> df_v2.printSchema()
> df_v2.show()
> 
> # Further evolved schema - Version 3 (added category column)
> print("\n3. Writing further evolved dataset (Version 3 - added 'category' column)...")
> evolved_data_v3 = [
>     (7, "Product G", 200, 49.99, "Electronics"),
>     (8, "Product H", 150, 59.99, "Electronics"),
>     (9, "Product I", 100, 24.99, "Home")
> ]
> 
> df_v3 = spark.createDataFrame(evolved_data_v3, ["id", "name", "quantity", "price", "category"])
> df_v3.write.mode("append").parquet(f"{output_path}/batch3")
> 
> print("Version 3 Schema:")
> df_v3.printSchema()
> df_v3.show()
> 
> # Read with schema merge - combines all schemas
> print("\n4. Reading with mergeSchema=true (combines all columns)...")
> df_merged = spark.read \
>     .option("mergeSchema", "true") \
>     .parquet(output_path)
> 
> print("Merged Schema:")
> df_merged.printSchema()
> 
> print("\nMerged Data (nulls for missing columns in older batches):")
> df_merged.orderBy("id").show(truncate=False)
> 
> # Read without schema merge (fails or uses first schema only)
> print("\n5. Reading WITHOUT mergeSchema (uses first encountered schema)...")
> df_no_merge = spark.read.parquet(output_path)
> 
> print("Non-Merged Schema:")
> df_no_merge.printSchema()
> 
> # Handle missing columns
> print("\n6. Handling evolved schema in queries:")
> result = df_merged.select(
>     "id",
>     "name",
>     "quantity",
>     df_merged.price.cast("double").alias("price"),
>     df_merged.category.alias("category")
> ).fillna({"price": 0.0, "category": "Unknown"})
> 
> print("Cleaned Data with Defaults:")
> result.show(truncate=False)
> 
> spark.stop()
> ```
> 
> ```python complex_types.py
> """
> Pattern 3: Handling Complex Types (ArrayType, MapType, StructType)
> Demonstrates working with nested and complex data structures
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.types import (
>     StructType, StructField, StringType, IntegerType, 
>     ArrayType, MapType, DoubleType
> )
> from pyspark.sql.functions import (
>     explode, col, size, map_keys, map_values, 
>     array_contains, struct, to_json
> )
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Complex Types Handling") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("COMPLEX TYPES HANDLING")
> print("=" * 80)
> 
> # Define complex schema with nested structures
> complex_schema = StructType([
>     StructField("customer_id", IntegerType(), False),
>     StructField("name", StringType(), False),
>     
>     # ArrayType - list of items
>     StructField("phone_numbers", ArrayType(StringType()), True),
>     
>     # MapType - key-value pairs
>     StructField("preferences", MapType(StringType(), StringType()), True),
>     
>     # StructType - nested structure
>     StructField("address", StructType([
>         StructField("street", StringType(), True),
>         StructField("city", StringType(), True),
>         StructField("zip", StringType(), True)
>     ]), True),
>     
>     # Array of Structs
>     StructField("orders", ArrayType(StructType([
>         StructField("order_id", IntegerType(), True),
>         StructField("amount", DoubleType(), True),
>         StructField("items", ArrayType(StringType()), True)
>     ])), True)
> ])
> 
> # Sample complex data
> data = [
>     (
>         1,
>         "John Doe",
>         ["555-1234", "555-5678"],
>         {"theme": "dark", "language": "en", "notifications": "enabled"},
>         ("123 Main St", "New York", "10001"),
>         [
>             (101, 299.99, ["Laptop", "Mouse"]),
>             (102, 49.99, ["Keyboard"])
>         ]
>     ),
>     (
>         2,
>         "Jane Smith",
>         ["555-8765"],
>         {"theme": "light", "language": "es"},
>         ("456 Oak Ave", "Los Angeles", "90001"),
>         [
>             (201, 599.99, ["Phone", "Case", "Charger"]),
>             (202, 129.99, ["Headphones"])
>         ]
>     ),
>     (
>         3,
>         "Bob Wilson",
>         ["555-4321", "555-9999", "555-1111"],
>         {"language": "fr", "notifications": "disabled"},
>         ("789 Pine Rd", "Chicago", "60601"),
>         []
>     )
> ]
> 
> # Create DataFrame
> df = spark.createDataFrame(data, schema=complex_schema)
> 
> print("\n1. ORIGINAL COMPLEX SCHEMA:")
> df.printSchema()
> 
> print("\nOriginal Data:")
> df.show(truncate=False)
> 
> # Working with ArrayType
> print("\n2. WORKING WITH ARRAYS:")
> print("\nArray Operations:")
> df.select(
>     "customer_id",
>     "name",
>     "phone_numbers",
>     size("phone_numbers").alias("phone_count"),
>     array_contains("phone_numbers", "555-1234").alias("has_specific_number")
> ).show(truncate=False)
> 
> print("\nExploding Arrays (one row per phone number):")
> df.select(
>     "customer_id",
>     "name",
>     explode("phone_numbers").alias("phone")
> ).show(truncate=False)
> 
> # Working with MapType
> print("\n3. WORKING WITH MAPS:")
> print("\nMap Operations:")
> df.select(
>     "customer_id",
>     "name",
>     "preferences",
>     map_keys("preferences").alias("pref_keys"),
>     map_values("preferences").alias("pref_values")
> ).show(truncate=False)
> 
> print("\nAccessing Map Values:")
> df.select(
>     "customer_id",
>     "name",
>     col("preferences.theme").alias("theme"),
>     col("preferences.language").alias("language")
> ).show(truncate=False)
> 
> # Working with StructType
> print("\n4. WORKING WITH NESTED STRUCTS:")
> print("\nAccessing Nested Fields:")
> df.select(
>     "customer_id",
>     "name",
>     "address.street",
>     "address.city",
>     "address.zip"
> ).show(truncate=False)
> 
> print("\nAccessing as Struct:")
> df.select(
>     "customer_id",
>     "address"
> ).show(truncate=False)
> 
> # Working with Array of Structs
> print("\n5. WORKING WITH ARRAY OF STRUCTS:")
> print("\nExploding Nested Orders:")
> df.select(
>     "customer_id",
>     "name",
>     explode("orders").alias("order")
> ).select(
>     "customer_id",
>     "name",
>     "order.order_id",
>     "order.amount",
>     "order.items"
> ).show(truncate=False)
> 
> print("\nFully Exploded Orders and Items:")
> from pyspark.sql.functions import explode_outer
> 
> df.select(
>     "customer_id",
>     "name",
>     explode_outer("orders").alias("order")
> ).select(
>     "customer_id",
>     "name",
>     col("order.order_id").alias("order_id"),
>     col("order.amount").alias("amount"),
>     explode_outer("order.items").alias("item")
> ).show(truncate=False)
> 
> # Creating complex types
> print("\n6. CREATING COMPLEX TYPES:")
> from pyspark.sql.functions import array, create_map, lit
> 
> simple_df = spark.createDataFrame([
>     (1, "Alice", "New York"),
>     (2, "Bob", "Chicago")
> ], ["id", "name", "city"])
> 
> complex_created = simple_df.select(
>     "id",
>     "name",
>     # Create array
>     array(lit("email"), lit("sms")).alias("contact_methods"),
>     # Create map
>     create_map(
>         lit("city"), col("city"),
>         lit("status"), lit("active")
>     ).alias("metadata"),
>     # Create struct
>     struct(
>         col("name").alias("full_name"),
>         col("city").alias("location")
>     ).alias("person_info")
> )
> 
> print("\nCreated Complex Types:")
> complex_created.printSchema()
> complex_created.show(truncate=False)
> 
> # Convert to JSON
> print("\n7. CONVERTING COMPLEX TYPES TO JSON:")
> df.select(
>     "customer_id",
>     to_json("address").alias("address_json"),
>     to_json("orders").alias("orders_json")
> ).show(truncate=False)
> 
> spark.stop()
> ```
> 
> ```python schema_inference.py
> """
> Pattern 4: Schema Inference
> Demonstrates automatic schema detection from various data sources
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.types import *
> import json
> import os
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Schema Inference") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("SCHEMA INFERENCE DEMONSTRATION")
> print("=" * 80)
> 
> # 1. Inferring from Python data structures
> print("\n1. SCHEMA INFERENCE FROM PYTHON DATA:")
> 
> python_data = [
>     {"id": 1, "name": "Alice", "age": 30, "salary": 75000.50, "active": True},
>     {"id": 2, "name": "Bob", "age": 25, "salary": 65000.75, "active": True},
>     {"id": 3, "name": "Charlie", "age": 35, "salary": 85000.00, "active": False}
> ]
> 
> df_inferred = spark.createDataFrame(python_data)
> 
> print("Inferred Schema from Python dicts:")
> df_inferred.printSchema()
> df_inferred.show()
> 
> # 2. Inferring from JSON
> print("\n2. SCHEMA INFERENCE FROM JSON:")
> 
> json_data = [
>     '{"user_id": 1, "username": "john", "age": 28, "tags": ["python", "spark"], "metadata": {"role": "engineer"}}',
>     '{"user_id": 2, "username": "jane", "age": 32, "tags": ["java", "kafka"], "metadata": {"role": "architect"}}',
>     '{"user_id": 3, "username": "bob", "age": 45, "tags": ["scala"], "metadata": {"role": "manager", "team": "data"}}'
> ]
> 
> json_path = "/tmp/users_inference.json"
> with open(json_path, 'w') as f:
>     f.write('\n'.join(json_data))
> 
> # Infer schema from JSON
> df_json = spark.read.json(json_path)
> 
> print("Inferred Schema from JSON:")
> df_json.printSchema()
> df_json.show(truncate=False)
> 
> # 3. Inferring from CSV
> print("\n3. SCHEMA INFERENCE FROM CSV:")
> 
> csv_data = """user_id,name,email,age,balance,join_date
> 1,Alice Smith,alice@example.com,30,1500.50,2023-01-15
> 2,Bob Jones,bob@example.com,25,2300.75,2023-02-20
> 3,Charlie Brown,charlie@example.com,35,980.25,2023-03-10
> 4,Diana Prince,diana@example.com,28,3200.00,2023-04-05"""
> 
> csv_path = "/tmp/users_inference.csv"
> with open(csv_path, 'w') as f:
>     f.write(csv_data)
> 
> # Infer schema with type inference enabled
> df_csv = spark.read \
>     .option("header", "true") \
>     .option("inferSchema", "true") \
>     .csv(csv_path)
> 
> print("Inferred Schema from CSV (with inferSchema=true):")
> df_csv.printSchema()
> df_csv.show(truncate=False)
> 
> # Without inference (all strings)
> df_csv_no_infer = spark.read \
>     .option("header", "true") \
>     .option("inferSchema", "false") \
>     .csv(csv_path)
> 
> print("\nWithout Schema Inference (all StringType):")
> df_csv_no_infer.printSchema()
> 
> # 4. Inferring from Parquet
> print("\n4. SCHEMA INFERENCE FROM PARQUET:")
> 
> # Write data to Parquet (includes schema)
> parquet_path = "/tmp/users_inference.parquet"
> df_json.write.mode("overwrite").parquet(parquet_path)
> 
> # Read back - schema automatically inferred from metadata
> df_parquet = spark.read.parquet(parquet_path)
> 
> print("Schema automatically read from Parquet metadata:")
> df_parquet.printSchema()
> df_parquet.show(truncate=False)
> 
> # 5. Sampling for inference
> print("\n5. SCHEMA INFERENCE WITH SAMPLING:")
> 
> large_json_data = []
> for i in range(1000):
>     large_json_data.append(json.dumps({
>         "id": i,
>         "value": i * 1.5,
>         "text": f"item_{i}",
>         "flag": i % 2 == 0
>     }))
> 
> large_json_path = "/tmp/large_data.json"
> with open(large_json_path, 'w') as f:
>     f.write('\n'.join(large_json_data))
> 
> # Infer with sampling (default samplingRatio=1.0)
> df_sampled = spark.read \
>     .option("samplingRatio", "0.1") \
>     .json(large_json_path)
> 
> print("Schema inferred from 10% sample:")
> df_sampled.printSchema()
> print(f"Total records: {df_sampled.count()}")
> 
> # 6. Complex nested inference
> print("\n6. COMPLEX NESTED SCHEMA INFERENCE:")
> 
> nested_json = [
>     {
>         "id": 1,
>         "user": {"name": "Alice", "email": "alice@test.com"},
>         "orders": [
>             {"order_id": 101, "items": ["laptop", "mouse"]},
>             {"order_id": 102, "items": ["keyboard"]}
>         ],
>         "preferences": {"theme": "dark", "lang": "en"}
>     },
>     {
>         "id": 2,
>         "user": {"name": "Bob", "email": "bob@test.com"},
>         "orders": [
>             {"order_id": 201, "items": ["phone"]}
>         ],
>         "preferences": {"theme": "light", "lang": "es", "notifications": True}
>     }
> ]
> 
> nested_json_path = "/tmp/nested_inference.json"
> with open(nested_json_path, 'w') as f:
>     f.write('\n'.join([json.dumps(item) for item in nested_json]))
> 
> df_nested = spark.read.json(nested_json_path)
> 
> print("Inferred Complex Nested Schema:")
> df_nested.printSchema()
> df_nested.show(truncate=False)
> 
> # 7. Controlling inference behavior
> print("\n7. COMPARING INFERENCE STRATEGIES:")
> 
> # With column name inference from first row
> df_csv_first = spark.read \
>     .option("header", "true") \
>     .option("inferSchema", "true") \
>     .csv(csv_path)
> 
> # With explicit date format for better inference
> csv_with_dates = """date,value,category
> 2023-01-15,100.5,A
> 2023-02-20,200.75,B
> 2023-03-10,150.25,A"""
> 
> csv_dates_path = "/tmp/dates_inference.csv"
> with open(csv_dates_path, 'w') as f:
>     f.write(csv_with_dates)
> 
> df_dates = spark.read \
>     .option("header", "true") \
>     .option("inferSchema", "true") \
>     .option("dateFormat", "yyyy-MM-dd") \
>     .csv(csv_dates_path)
> 
> print("Inferred with Date Format:")
> df_dates.printSchema()
> df_dates.show(truncate=False)
> 
> # Performance consideration
> print("\n8. INFERENCE PERFORMANCE TIPS:")
> print("- Parquet/ORC: Schema stored in metadata (fast)")
> print("- JSON: Requires scanning data (slower)")
> print("- CSV: Requires full scan for inferSchema=true (slowest)")
> print("- Use samplingRatio for large datasets")
> print("- Explicit schemas are fastest and most reliable for production")
> 
> # Cleanup
> for path in [json_path, csv_path, parquet_path, large_json_path, nested_json_path, csv_dates_path]:
>     if os.path.exists(path):
>         os.remove(path)
> 
> spark.stop()
> ```
> 
> These programs demonstrate:
> 
> 1. **Explicit Schema Definition**: Defining schemas upfront for type safety and performance
> 2. **Schema Evolution**: Handling schema changes over time with `mergeSchema`
> 3. **Complex Types**: Working with arrays, maps, structs, and nested structures
> 4. **Schema Inference**: Automatic schema detection from various data sources
> 
> Each program includes detailed examples and can be run independently. They show best practices for production environments and common pitfalls to avoid.