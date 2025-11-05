### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:10:16 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> ### **I. Core SparkSession & Context Management**
> *   Singleton SparkSession
> *   Configuring SparkSession (Runtime Properties)
> *   Accessing SparkContext
> ```

#### _Assistant_

> Here are Python PySpark programs demonstrating each pattern:
> 
> ## 1. Singleton SparkSession
> 
> ```python singleton_sparksession.py
> """
> Singleton SparkSession Pattern
> Ensures only one SparkSession exists throughout the application lifecycle
> """
> 
> from pyspark.sql import SparkSession
> 
> 
> class SparkSessionSingleton:
>     """Singleton pattern for SparkSession"""
>     _instance = None
>     _lock = None
>     
>     def __new__(cls):
>         if cls._instance is None:
>             cls._instance = super(SparkSessionSingleton, cls).__new__(cls)
>             cls._instance._spark = None
>         return cls._instance
>     
>     def get_session(self, app_name="SingletonSparkApp"):
>         """Get or create SparkSession"""
>         if self._spark is None:
>             print(f"Creating new SparkSession: {app_name}")
>             self._spark = SparkSession.builder \
>                 .appName(app_name) \
>                 .master("local[*]") \
>                 .getOrCreate()
>         else:
>             print("Reusing existing SparkSession")
>         return self._spark
>     
>     def stop_session(self):
>         """Stop the SparkSession"""
>         if self._spark is not None:
>             print("Stopping SparkSession")
>             self._spark.stop()
>             self._spark = None
> 
> 
> # Usage Example
> if __name__ == "__main__":
>     # First call - creates new session
>     singleton1 = SparkSessionSingleton()
>     spark1 = singleton1.get_session("MyApp")
>     print(f"Session 1 ID: {id(spark1)}")
>     
>     # Second call - reuses existing session
>     singleton2 = SparkSessionSingleton()
>     spark2 = singleton2.get_session("MyApp")
>     print(f"Session 2 ID: {id(spark2)}")
>     
>     # Verify both are the same instance
>     print(f"Same instance: {spark1 is spark2}")
>     
>     # Create a simple DataFrame to test
>     data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
>     df = spark1.createDataFrame(data, ["name", "age"])
>     df.show()
>     
>     # Cleanup
>     singleton1.stop_session()
> ```
> 
> ## 2. Configuring SparkSession (Runtime Properties)
> 
> ```python configure_sparksession.py
> """
> SparkSession Configuration Pattern
> Demonstrates various runtime configuration options
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.conf import SparkConf
> 
> 
> def create_configured_spark_session():
>     """Create SparkSession with comprehensive runtime configurations"""
>     
>     # Method 1: Using SparkConf
>     conf = SparkConf()
>     conf.set("spark.app.name", "ConfiguredSparkApp")
>     conf.set("spark.executor.memory", "2g")
>     conf.set("spark.executor.cores", "2")
>     conf.set("spark.driver.memory", "1g")
>     conf.set("spark.sql.shuffle.partitions", "10")
>     conf.set("spark.default.parallelism", "8")
>     
>     spark = SparkSession.builder \
>         .config(conf=conf) \
>         .master("local[*]") \
>         .getOrCreate()
>     
>     return spark
> 
> 
> def create_spark_with_inline_config():
>     """Create SparkSession with inline configuration"""
>     
>     spark = SparkSession.builder \
>         .appName("InlineConfigApp") \
>         .master("local[*]") \
>         .config("spark.sql.shuffle.partitions", "50") \
>         .config("spark.sql.adaptive.enabled", "true") \
>         .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
>         .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
>         .config("spark.executor.memory", "4g") \
>         .config("spark.driver.maxResultSize", "2g") \
>         .config("spark.network.timeout", "600s") \
>         .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
>         .enableHiveSupport() \
>         .getOrCreate()
>     
>     return spark
> 
> 
> def modify_runtime_config(spark):
>     """Modify configuration at runtime"""
>     
>     print("=== Initial Configuration ===")
>     print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
>     print(f"Adaptive Execution: {spark.conf.get('spark.sql.adaptive.enabled')}")
>     
>     # Modify runtime configuration
>     spark.conf.set("spark.sql.shuffle.partitions", "200")
>     spark.conf.set("spark.sql.adaptive.enabled", "false")
>     
>     print("\n=== Modified Configuration ===")
>     print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
>     print(f"Adaptive Execution: {spark.conf.get('spark.sql.adaptive.enabled')}")
> 
> 
> def display_all_configurations(spark):
>     """Display all current Spark configurations"""
>     
>     print("\n=== All Spark Configurations ===")
>     all_configs = spark.sparkContext.getConf().getAll()
>     for key, value in sorted(all_configs):
>         print(f"{key}: {value}")
> 
> 
> # Usage Example
> if __name__ == "__main__":
>     # Create configured session
>     spark = create_spark_with_inline_config()
>     
>     # Display and modify configurations
>     modify_runtime_config(spark)
>     
>     # Show all configurations
>     display_all_configurations(spark)
>     
>     # Test with data
>     data = [(i, f"value_{i}") for i in range(1000)]
>     df = spark.createDataFrame(data, ["id", "value"])
>     
>     print(f"\n=== DataFrame Info ===")
>     print(f"Number of partitions: {df.rdd.getNumPartitions()}")
>     print(f"Row count: {df.count()}")
>     
>     # Cleanup
>     spark.stop()
> ```
> 
> ## 3. Accessing SparkContext
> 
> ```python accessing_sparkcontext.py
> """
> Accessing SparkContext Pattern
> Demonstrates how to access and use SparkContext from SparkSession
> """
> 
> from pyspark.sql import SparkSession
> from pyspark import SparkContext, SparkConf
> 
> 
> def demonstrate_sparkcontext_access():
>     """Show different ways to access SparkContext"""
>     
>     # Create SparkSession
>     spark = SparkSession.builder \
>         .appName("SparkContextAccess") \
>         .master("local[*]") \
>         .getOrCreate()
>     
>     # Method 1: Access through SparkSession (Recommended)
>     sc = spark.sparkContext
>     
>     print("=== SparkContext Information ===")
>     print(f"Application Name: {sc.appName}")
>     print(f"Master: {sc.master}")
>     print(f"Spark Version: {sc.version}")
>     print(f"Application ID: {sc.applicationId}")
>     print(f"Default Parallelism: {sc.defaultParallelism}")
>     print(f"Default Min Partitions: {sc.defaultMinPartitions}")
>     
>     return spark, sc
> 
> 
> def sparkcontext_operations(sc):
>     """Demonstrate common SparkContext operations"""
>     
>     print("\n=== SparkContext Operations ===")
>     
>     # 1. Create RDD using parallelize
>     rdd = sc.parallelize([1, 2, 3, 4, 5], numSlices=2)
>     print(f"RDD Partitions: {rdd.getNumPartitions()}")
>     print(f"RDD Data: {rdd.collect()}")
>     
>     # 2. Read text file
>     # Create a temporary file first
>     import tempfile
>     import os
>     
>     with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
>         f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
>         temp_file = f.name
>     
>     text_rdd = sc.textFile(temp_file)
>     print(f"\nText RDD Count: {text_rdd.count()}")
>     print(f"Text RDD Content: {text_rdd.collect()}")
>     
>     # Cleanup temp file
>     os.unlink(temp_file)
>     
>     # 3. Broadcast variable
>     broadcast_var = sc.broadcast({"key1": "value1", "key2": "value2"})
>     print(f"\nBroadcast Variable: {broadcast_var.value}")
>     
>     # 4. Accumulator
>     accumulator = sc.accumulator(0)
>     
>     def increment_accumulator(x):
>         accumulator.add(1)
>         return x * 2
>     
>     result_rdd = rdd.map(increment_accumulator)
>     result_rdd.collect()  # Trigger action to update accumulator
>     print(f"\nAccumulator Value: {accumulator.value}")
>     
>     # 5. Status and monitoring
>     print(f"\nSpark UI URL: {sc.uiWebUrl}")
>     print(f"Status Tracker: {sc.statusTracker()}")
> 
> 
> def sparkcontext_configuration(sc):
>     """Access and display SparkContext configuration"""
>     
>     print("\n=== SparkContext Configuration ===")
>     
>     # Get configuration
>     conf = sc.getConf()
>     
>     # Display specific configurations
>     print(f"App Name: {conf.get('spark.app.name')}")
>     print(f"Master: {conf.get('spark.master')}")
>     
>     # Get all configurations
>     print("\n=== All Configurations ===")
>     for key, value in conf.getAll():
>         if not key.startswith('spark.yarn'):  # Filter for readability
>             print(f"{key}: {value}")
>     
>     # Set log level
>     sc.setLogLevel("WARN")
>     print("\nLog level set to WARN")
> 
> 
> def sparkcontext_resource_info(sc):
>     """Display resource information from SparkContext"""
>     
>     print("\n=== Resource Information ===")
>     
>     # Get executor memory
>     executor_memory = sc.getConf().get("spark.executor.memory", "Not Set")
>     driver_memory = sc.getConf().get("spark.driver.memory", "Not Set")
>     
>     print(f"Executor Memory: {executor_memory}")
>     print(f"Driver Memory: {driver_memory}")
>     print(f"Default Parallelism: {sc.defaultParallelism}")
>     
>     # Check if running in local mode
>     is_local = sc.master.startswith("local")
>     print(f"Running in Local Mode: {is_local}")
> 
> 
> # Usage Example
> if __name__ == "__main__":
>     # Get SparkSession and SparkContext
>     spark, sc = demonstrate_sparkcontext_access()
>     
>     # Perform various SparkContext operations
>     sparkcontext_operations(sc)
>     
>     # Display configurations
>     sparkcontext_configuration(sc)
>     
>     # Display resource information
>     sparkcontext_resource_info(sc)
>     
>     # Integration example: Using both SparkSession and SparkContext
>     print("\n=== Integration Example ===")
>     
>     # Create DataFrame using SparkSession
>     df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "value"])
>     
>     # Convert to RDD using SparkContext
>     rdd = df.rdd
>     print(f"DataFrame -> RDD: {rdd.collect()}")
>     
>     # Process with RDD API
>     processed_rdd = rdd.map(lambda row: (row[0] * 10, row[1].upper()))
>     
>     # Convert back to DataFrame
>     processed_df = spark.createDataFrame(processed_rdd, ["id", "value"])
>     processed_df.show()
>     
>     # Cleanup
>     spark.stop()
> ```
> 
> Each program demonstrates a different core pattern:
> 
> 1. **Singleton SparkSession**: Ensures only one SparkSession instance exists, preventing resource conflicts
> 2. **Configuring SparkSession**: Shows multiple ways to configure Spark with runtime properties for optimization
> 3. **Accessing SparkContext**: Demonstrates how to access and use the underlying SparkContext for RDD operations, broadcasting, and accumulators
> 
> You can run each file independently with `python <filename>.py` after installing PySpark (`pip install pyspark`).