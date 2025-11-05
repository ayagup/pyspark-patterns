### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:49:43 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Error Handling & Debugging**
> *   Using `try-except` blocks for I/O operations
> *   Logging Spark operations
> *   Inspecting DataFrame content (`show`, `head`, `take`, `collect`, `printSchema`, `describe`)
> *   Analyzing Stack Traces for Job Failures
> *   Spark UI for Debugging Failed Stages/Tasks
> ```

#### _Assistant_

> I'll create a comprehensive PySpark program demonstrating all the error handling and debugging patterns you've requested.
> 
> ```python error_handling_debugging.py
> """
> PySpark Error Handling & Debugging Patterns
> Demonstrates best practices for debugging and error handling in PySpark applications
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, when, udf
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> import logging
> import sys
> from typing import Optional
> 
> # ============================================================================
> # PATTERN 1: Logging Configuration for Spark Operations
> # ============================================================================
> 
> def setup_logging():
>     """Configure logging for the application"""
>     # Create logger
>     logger = logging.getLogger('spark_debugging_demo')
>     logger.setLevel(logging.INFO)
>     
>     # Create console handler with formatting
>     console_handler = logging.StreamHandler(sys.stdout)
>     console_handler.setLevel(logging.INFO)
>     
>     # Create file handler
>     file_handler = logging.FileHandler('spark_operations.log')
>     file_handler.setLevel(logging.DEBUG)
>     
>     # Create formatter
>     formatter = logging.Formatter(
>         '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
>         datefmt='%Y-%m-%d %H:%M:%S'
>     )
>     
>     console_handler.setFormatter(formatter)
>     file_handler.setFormatter(formatter)
>     
>     # Add handlers to logger
>     logger.addHandler(console_handler)
>     logger.addHandler(file_handler)
>     
>     return logger
> 
> 
> # ============================================================================
> # PATTERN 2: Try-Except Blocks for I/O Operations
> # ============================================================================
> 
> def safe_read_csv(spark: SparkSession, file_path: str, logger: logging.Logger) -> Optional['DataFrame']:
>     """
>     Safely read CSV file with error handling
>     """
>     try:
>         logger.info(f"Attempting to read CSV file: {file_path}")
>         
>         df = spark.read \
>             .option("header", "true") \
>             .option("inferSchema", "true") \
>             .option("mode", "FAILFAST") \
>             .csv(file_path)
>         
>         logger.info(f"Successfully read {df.count()} records from {file_path}")
>         return df
>         
>     except FileNotFoundError as e:
>         logger.error(f"File not found: {file_path}")
>         logger.error(f"Error details: {str(e)}")
>         return None
>         
>     except Exception as e:
>         logger.error(f"Unexpected error reading CSV file: {file_path}")
>         logger.error(f"Error type: {type(e).__name__}")
>         logger.error(f"Error message: {str(e)}")
>         logger.error(f"Stack trace:", exc_info=True)
>         return None
> 
> 
> def safe_write_parquet(df: 'DataFrame', output_path: str, logger: logging.Logger) -> bool:
>     """
>     Safely write DataFrame to Parquet with error handling
>     """
>     try:
>         logger.info(f"Attempting to write DataFrame to: {output_path}")
>         
>         df.write \
>             .mode("overwrite") \
>             .parquet(output_path)
>         
>         logger.info(f"Successfully wrote DataFrame to {output_path}")
>         return True
>         
>     except PermissionError as e:
>         logger.error(f"Permission denied when writing to: {output_path}")
>         logger.error(f"Error details: {str(e)}")
>         return False
>         
>     except Exception as e:
>         logger.error(f"Failed to write DataFrame to {output_path}")
>         logger.error(f"Error type: {type(e).__name__}")
>         logger.error(f"Error message: {str(e)}")
>         logger.error(f"Stack trace:", exc_info=True)
>         return False
> 
> 
> def safe_read_json(spark: SparkSession, file_path: str, logger: logging.Logger, 
>                    schema: Optional[StructType] = None) -> Optional['DataFrame']:
>     """
>     Safely read JSON file with schema validation
>     """
>     try:
>         logger.info(f"Reading JSON file: {file_path}")
>         
>         if schema:
>             df = spark.read.schema(schema).json(file_path)
>             logger.info("Schema validation applied during read")
>         else:
>             df = spark.read.json(file_path)
>             logger.info("Schema inferred from data")
>         
>         logger.info(f"Successfully loaded {df.count()} records")
>         return df
>         
>     except ValueError as e:
>         logger.error(f"Schema mismatch or invalid data in: {file_path}")
>         logger.error(f"Error: {str(e)}")
>         return None
>         
>     except Exception as e:
>         logger.error(f"Error reading JSON: {str(e)}", exc_info=True)
>         return None
> 
> 
> # ============================================================================
> # PATTERN 3: Inspecting DataFrame Content
> # ============================================================================
> 
> def inspect_dataframe_comprehensive(df: 'DataFrame', logger: logging.Logger, name: str = "DataFrame"):
>     """
>     Comprehensive DataFrame inspection using multiple methods
>     """
>     logger.info(f"\n{'='*80}")
>     logger.info(f"INSPECTING: {name}")
>     logger.info(f"{'='*80}")
>     
>     try:
>         # 1. Print Schema
>         logger.info("\n--- SCHEMA ---")
>         df.printSchema()
>         
>         # 2. Show sample data
>         logger.info("\n--- SAMPLE DATA (show) ---")
>         df.show(5, truncate=False)
>         
>         # 3. Head - returns Row objects
>         logger.info("\n--- HEAD (first 3 rows as Row objects) ---")
>         head_rows = df.head(3)
>         for i, row in enumerate(head_rows, 1):
>             logger.info(f"Row {i}: {row}")
>         
>         # 4. Take - similar to head but explicit count
>         logger.info("\n--- TAKE (2 rows) ---")
>         taken_rows = df.take(2)
>         logger.info(f"Taken rows: {taken_rows}")
>         
>         # 5. Describe - statistical summary
>         logger.info("\n--- DESCRIBE (Statistical Summary) ---")
>         df.describe().show()
>         
>         # 6. Count and other metrics
>         logger.info("\n--- METRICS ---")
>         record_count = df.count()
>         column_count = len(df.columns)
>         logger.info(f"Total Records: {record_count}")
>         logger.info(f"Total Columns: {column_count}")
>         logger.info(f"Columns: {df.columns}")
>         
>         # 7. Data types
>         logger.info("\n--- DATA TYPES ---")
>         for field in df.dtypes:
>             logger.info(f"{field[0]}: {field[1]}")
>         
>         # 8. NULL counts
>         logger.info("\n--- NULL COUNTS ---")
>         from pyspark.sql.functions import col, sum as spark_sum, count
>         
>         null_counts = df.select([
>             spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
>             for c in df.columns
>         ])
>         null_counts.show()
>         
>         # WARNING: collect() brings all data to driver - use cautiously
>         logger.warning("\n--- COLLECT (USE WITH CAUTION) ---")
>         logger.warning("collect() brings all data to driver memory!")
>         logger.warning("Only use on small datasets or with .limit()")
>         
>         small_sample = df.limit(10).collect()
>         logger.info(f"Collected {len(small_sample)} rows (limited sample)")
>         
>     except Exception as e:
>         logger.error(f"Error during DataFrame inspection: {str(e)}", exc_info=True)
> 
> 
> # ============================================================================
> # PATTERN 4: Analyzing Stack Traces for Job Failures
> # ============================================================================
> 
> def analyze_stack_trace_example(spark: SparkSession, logger: logging.Logger):
>     """
>     Demonstrates how to capture and analyze stack traces from Spark job failures
>     """
>     logger.info("\n--- STACK TRACE ANALYSIS DEMO ---")
>     
>     # Create sample data with potential issues
>     data = [
>         ("Alice", 25, 50000),
>         ("Bob", 30, 60000),
>         ("Charlie", None, 70000),  # NULL age
>         ("David", 35, None),  # NULL salary
>     ]
>     
>     df = spark.createDataFrame(data, ["name", "age", "salary"])
>     
>     # Scenario 1: Division by zero
>     try:
>         logger.info("\nTest 1: Division by zero scenario")
>         
>         result = df.withColumn("ratio", col("salary") / (col("age") - 30))
>         result.show()
>         
>     except Exception as e:
>         logger.error("=" * 80)
>         logger.error("CAUGHT EXCEPTION: Division by zero")
>         logger.error(f"Exception Type: {type(e).__name__}")
>         logger.error(f"Exception Message: {str(e)}")
>         logger.error("\nFULL STACK TRACE:")
>         logger.error("=" * 80, exc_info=True)
>         
>         # Parse and analyze the stack trace
>         import traceback
>         tb_lines = traceback.format_exc().split('\n')
>         
>         logger.info("\n--- STACK TRACE ANALYSIS ---")
>         logger.info("Key points to look for:")
>         logger.info("1. The actual error message (last line)")
>         logger.info("2. The line of code that caused the error")
>         logger.info("3. The chain of function calls leading to the error")
>     
>     
>     # Scenario 2: UDF failure with better error handling
>     try:
>         logger.info("\nTest 2: UDF failure scenario")
>         
>         @udf(returnType=StringType())
>         def risky_udf(value):
>             """UDF that might fail"""
>             if value is None:
>                 raise ValueError("NULL value encountered in UDF")
>             return str(value).upper()
>         
>         result = df.withColumn("name_upper", risky_udf(col("age")))
>         result.show()
>         
>     except Exception as e:
>         logger.error("=" * 80)
>         logger.error("CAUGHT EXCEPTION: UDF Failure")
>         logger.error(f"Exception Type: {type(e).__name__}")
>         logger.error("\nDEBUGGING TIPS FOR UDF FAILURES:")
>         logger.error("1. UDFs hide the actual row causing the error")
>         logger.error("2. Add explicit NULL handling in UDFs")
>         logger.error("3. Use built-in functions when possible")
>         logger.error("4. Test UDFs on sample data first")
>         logger.error("=" * 80, exc_info=True)
> 
> 
> # ============================================================================
> # PATTERN 5: Spark UI Integration & Failed Stage/Task Analysis
> # ============================================================================
> 
> def spark_ui_debugging_example(spark: SparkSession, logger: logging.Logger):
>     """
>     Demonstrates operations that can be debugged using Spark UI
>     Provides guidance on what to look for in the UI
>     """
>     logger.info("\n" + "=" * 80)
>     logger.info("SPARK UI DEBUGGING GUIDE")
>     logger.info("=" * 80)
>     
>     # Print Spark UI URL
>     spark_ui_url = spark.sparkContext.uiWebUrl
>     logger.info(f"\nSpark UI URL: {spark_ui_url}")
>     logger.info("Access this URL in your browser to monitor the application")
>     
>     logger.info("\n--- KEY SPARK UI TABS ---")
>     logger.info("1. JOBS: See all jobs, their stages, and completion status")
>     logger.info("2. STAGES: Detailed stage information, task metrics")
>     logger.info("3. STORAGE: Cached/persisted DataFrames")
>     logger.info("4. ENVIRONMENT: Spark configuration")
>     logger.info("5. EXECUTORS: Executor status, memory usage, task distribution")
>     logger.info("6. SQL: Query execution plans and metrics")
>     
>     # Create a scenario that generates multiple stages
>     try:
>         logger.info("\n--- GENERATING MULTI-STAGE JOB ---")
>         
>         # Create larger dataset
>         data = [(i, f"user_{i}", i * 100, i % 10) 
>                 for i in range(1, 10001)]
>         df = spark.createDataFrame(data, ["id", "name", "amount", "category"])
>         
>         logger.info("Created DataFrame with 10,000 records")
>         logger.info("CHECK SPARK UI: Go to SQL tab to see the query plan")
>         
>         # Cache to see in Storage tab
>         df.cache()
>         logger.info("\nDataFrame cached - check Storage tab in Spark UI")
>         
>         # Trigger action
>         count = df.count()
>         logger.info(f"Record count: {count}")
>         
>         # Complex transformation creating multiple stages
>         logger.info("\n--- COMPLEX TRANSFORMATION (Multiple Stages) ---")
>         
>         result = df \
>             .filter(col("amount") > 500) \
>             .groupBy("category") \
>             .agg({"amount": "sum", "id": "count"}) \
>             .orderBy("category")
>         
>         logger.info("\nCHECK SPARK UI:")
>         logger.info("- Jobs tab: See the job and its stages")
>         logger.info("- Stages tab: Click on each stage to see:")
>         logger.info("  * Task completion times")
>         logger.info("  * Data shuffle metrics")
>         logger.info("  * Executor task distribution")
>         
>         result.show()
>         
>         # Simulate a skewed join scenario
>         logger.info("\n--- SIMULATING DATA SKEW (Watch for slow tasks) ---")
>         
>         # Create skewed data
>         skewed_data = [(i % 3, f"value_{i}") for i in range(10000)]
>         df2 = spark.createDataFrame(skewed_data, ["key", "value"])
>         
>         joined = df.join(df2, df["category"] == df2["key"], "inner")
>         
>         logger.info("\nPerforming join...")
>         logger.info("CHECK SPARK UI Stages:")
>         logger.info("- Look for tasks with significantly longer duration")
>         logger.info("- Check 'Shuffle Read Size / Records' for skew")
>         logger.info("- Review task metrics in the stage detail page")
>         
>         result_count = joined.count()
>         logger.info(f"Join result count: {result_count}")
>         
>     except Exception as e:
>         logger.error("=" * 80)
>         logger.error("JOB FAILED - USE SPARK UI TO DEBUG")
>         logger.error("=" * 80)
>         logger.error(f"\nError: {str(e)}")
>         logger.error("\nDEBUGGING STEPS:")
>         logger.error("1. Go to Spark UI > Jobs tab")
>         logger.error("2. Find the failed job (marked in red)")
>         logger.error("3. Click on it to see failed stages")
>         logger.error("4. Click on failed stage to see failed tasks")
>         logger.error("5. Review task error messages and logs")
>         logger.error("6. Check executor logs for detailed error info")
>         logger.error("=" * 80, exc_info=True)
> 
> 
> def demonstrate_debugging_workflow(spark: SparkSession, logger: logging.Logger):
>     """
>     Complete debugging workflow combining all patterns
>     """
>     logger.info("\n" + "=" * 80)
>     logger.info("COMPLETE DEBUGGING WORKFLOW")
>     logger.info("=" * 80)
>     
>     # Step 1: Safe data loading with try-except
>     logger.info("\nSTEP 1: Loading data with error handling")
>     
>     # Create sample data file
>     sample_data = [
>         ("John", 28, 55000, "Engineering"),
>         ("Jane", 32, 65000, "Marketing"),
>         ("Bob", 45, 85000, "Engineering"),
>         ("Alice", 29, 60000, "Sales"),
>         ("Charlie", 38, 75000, "Engineering"),
>     ]
>     
>     df = spark.createDataFrame(sample_data, ["name", "age", "salary", "department"])
>     
>     # Step 2: Inspect the data
>     logger.info("\nSTEP 2: Inspecting DataFrame")
>     inspect_dataframe_comprehensive(df, logger, "Employee Data")
>     
>     # Step 3: Perform transformation with logging
>     logger.info("\nSTEP 3: Applying transformations")
>     try:
>         logger.info("Calculating salary bands...")
>         
>         result = df.withColumn(
>             "salary_band",
>             when(col("salary") < 60000, "Low")
>             .when(col("salary") < 75000, "Medium")
>             .otherwise("High")
>         )
>         
>         logger.info("Transformation successful")
>         result.show()
>         
>         # Step 4: Aggregation with monitoring
>         logger.info("\nSTEP 4: Performing aggregation")
>         
>         summary = result.groupBy("department", "salary_band").count()
>         logger.info(f"Aggregation complete - {summary.count()} groups")
>         
>         summary.show()
>         
>         return result
>         
>     except Exception as e:
>         logger.error("Transformation failed", exc_info=True)
>         return None
> 
> 
> # ============================================================================
> # MAIN EXECUTION
> # ============================================================================
> 
> def main():
>     """Main execution function"""
>     
>     # Setup logging
>     logger = setup_logging()
>     logger.info("=" * 80)
>     logger.info("PYSPARK ERROR HANDLING & DEBUGGING PATTERNS")
>     logger.info("=" * 80)
>     
>     # Create Spark Session with UI enabled
>     spark = SparkSession.builder \
>         .appName("ErrorHandlingDebuggingDemo") \
>         .master("local[*]") \
>         .config("spark.ui.enabled", "true") \
>         .config("spark.ui.port", "4040") \
>         .config("spark.sql.shuffle.partitions", "4") \
>         .getOrCreate()
>     
>     # Set log level for Spark
>     spark.sparkContext.setLogLevel("WARN")
>     
>     logger.info(f"Spark Session created: {spark.version}")
>     logger.info(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")
>     
>     try:
>         # Pattern 1 & 2: Try-Except for I/O
>         logger.info("\n\n" + "=" * 80)
>         logger.info("PATTERN 1 & 2: I/O ERROR HANDLING")
>         logger.info("=" * 80)
>         
>         # Demonstrate safe read (will fail gracefully)
>         df = safe_read_csv(spark, "nonexistent_file.csv", logger)
>         
>         # Pattern 3: DataFrame Inspection
>         logger.info("\n\n" + "=" * 80)
>         logger.info("PATTERN 3: DATAFRAME INSPECTION")
>         logger.info("=" * 80)
>         
>         demonstrate_debugging_workflow(spark, logger)
>         
>         # Pattern 4: Stack Trace Analysis
>         logger.info("\n\n" + "=" * 80)
>         logger.info("PATTERN 4: STACK TRACE ANALYSIS")
>         logger.info("=" * 80)
>         
>         analyze_stack_trace_example(spark, logger)
>         
>         # Pattern 5: Spark UI Debugging
>         logger.info("\n\n" + "=" * 80)
>         logger.info("PATTERN 5: SPARK UI DEBUGGING")
>         logger.info("=" * 80)
>         
>         spark_ui_debugging_example(spark, logger)
>         
>         logger.info("\n" + "=" * 80)
>         logger.info("DEMO COMPLETE")
>         logger.info("=" * 80)
>         logger.info("\nKEY TAKEAWAYS:")
>         logger.info("1. Always wrap I/O operations in try-except blocks")
>         logger.info("2. Use comprehensive logging for production applications")
>         logger.info("3. Inspect DataFrames at key transformation points")
>         logger.info("4. Analyze stack traces methodically from bottom to top")
>         logger.info("5. Use Spark UI to identify performance bottlenecks")
>         logger.info("6. Avoid collect() on large datasets")
>         logger.info("7. Cache/persist strategically and monitor in Storage tab")
>         
>         input("\nPress Enter to stop Spark session (this will close the UI)...")
>         
>     except Exception as e:
>         logger.error("Application failed", exc_info=True)
>         
>     finally:
>         logger.info("\nStopping Spark Session...")
>         spark.stop()
>         logger.info("Spark Session stopped")
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> This comprehensive program demonstrates all the error handling and debugging patterns:
> 
> ## **Key Features:**
> 
> 1. **Try-Except for I/O**: Safe read/write functions with detailed error handling
> 2. **Logging**: Comprehensive logging setup with file and console handlers
> 3. **DataFrame Inspection**: All methods (show, head, take, collect, printSchema, describe)
> 4. **Stack Trace Analysis**: Examples showing how to capture and analyze errors
> 5. **Spark UI Integration**: Guidance on using the UI to debug failed stages/tasks
> 
> ## **To Run:**
> 
> ```bash
> python error_handling_debugging.py
> ```
> 
> The program will:
> - Start a local Spark session with UI at http://localhost:4040
> - Demonstrate each debugging pattern
> - Create sample scenarios that can be monitored in Spark UI
> - Log all operations to both console and `spark_operations.log`
> 
> Check the Spark UI while the program runs to see stages, tasks, and execution plans!