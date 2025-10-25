"""
PySpark Error Handling Patterns - Example Implementations
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from pyspark.sql.utils import AnalysisException
from time import sleep
import time

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ErrorHandlingPatterns") \
    .master("local[*]") \
    .getOrCreate()


# ============================================
# 6.1 Try-Catch Pattern
# ============================================
def safe_read(spark, path, format_type="parquet"):
    """Safely read data with error handling"""
    try:
        if format_type == "parquet":
            return spark.read.parquet(path)
        elif format_type == "csv":
            return spark.read.option("header", "true").csv(path)
        elif format_type == "json":
            return spark.read.json(path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    except AnalysisException as e:
        print(f"Error reading {path}: {e}")
        return spark.createDataFrame([], schema=StructType([]))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def try_catch_example():
    """Demonstrate try-catch pattern"""
    print("\n=== Try-Catch Pattern ===")
    
    # Try to read existing file
    df = spark.createDataFrame([
        (1, "Alice", 30),
        (2, "Bob", 25)
    ], ["id", "name", "age"])
    df.write.mode("overwrite").parquet("/tmp/valid_data.parquet")
    
    print("Reading valid file:")
    result = safe_read(spark, "/tmp/valid_data.parquet")
    if not result.rdd.isEmpty():
        result.show()
    
    # Try to read non-existent file
    print("\nReading non-existent file:")
    result = safe_read(spark, "/tmp/nonexistent_file.parquet")
    if result.rdd.isEmpty():
        print("Returned empty DataFrame due to error")


# ============================================
# 6.2 UDF with Error Handling
# ============================================
@udf(returnType=IntegerType())
def safe_divide(numerator, denominator):
    """Safe division UDF"""
    try:
        if denominator == 0:
            return None
        return int(numerator / denominator)
    except:
        return None


@udf(returnType=StringType())
def safe_parse_json(json_str):
    """Safe JSON parsing UDF"""
    import json
    try:
        data = json.loads(json_str)
        return data.get('key', None)
    except (json.JSONDecodeError, KeyError, AttributeError):
        return None


def udf_error_handling_example():
    """Demonstrate UDF error handling"""
    print("\n=== UDF Error Handling Pattern ===")
    
    # Create data with potential errors
    df = spark.createDataFrame([
        (1, 100, 10, '{"key": "value1"}'),
        (2, 200, 0, '{"key": "value2"}'),  # Division by zero
        (3, 300, 5, 'invalid json'),  # Invalid JSON
        (4, 400, 8, '{"other": "data"}'),  # Missing key
    ], ["id", "numerator", "denominator", "json_data"])
    
    print("Original Data:")
    df.show(truncate=False)
    
    # Apply safe UDFs
    df_result = df.withColumn("division_result", safe_divide(col("numerator"), col("denominator"))) \
        .withColumn("parsed_value", safe_parse_json(col("json_data")))
    
    print("\nAfter Safe UDFs (nulls for errors):")
    df_result.show(truncate=False)


# ============================================
# 6.3 Dead Letter Queue Pattern
# ============================================
def process_with_dlq(df, processing_func):
    """Process data and capture failures in DLQ"""
    # Add processing result column
    df_processed = df.withColumn("processed", processing_func(col("data")))
    
    # Split success and failures
    df_success = df_processed.filter(col("processed").isNotNull())
    df_failed = df_processed.filter(col("processed").isNull())
    
    return df_success, df_failed


@udf(returnType=IntegerType())
def risky_processing(value):
    """Processing function that may fail"""
    try:
        # Simulate processing that fails for certain values
        if value % 3 == 0:
            raise ValueError("Processing failed")
        return value * 2
    except:
        return None


def dlq_example():
    """Demonstrate Dead Letter Queue pattern"""
    print("\n=== Dead Letter Queue Pattern ===")
    
    # Create data
    df = spark.createDataFrame([
        (1, 10),
        (2, 15),
        (3, 20),
        (4, 30),  # Will fail
        (5, 25)
    ], ["id", "data"])
    
    print("Original Data:")
    df.show()
    
    # Process with DLQ
    df_success, df_failed = process_with_dlq(df, risky_processing)
    
    print("\nSuccessful Records:")
    df_success.show()
    
    print("\nFailed Records (DLQ):")
    df_failed.show()
    
    # In production, write failed records to DLQ
    # df_failed.write.mode("append").parquet("/dlq/failed_records")
    print(f"Success count: {df_success.count()}")
    print(f"Failed count: {df_failed.count()}")


# ============================================
# 6.4 Retry Pattern
# ============================================
def retry_operation(func, max_retries=3, delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            sleep(wait_time)


def retry_example():
    """Demonstrate retry pattern"""
    print("\n=== Retry Pattern ===")
    
    # Simulate flaky operation
    class FlakyOperation:
        def __init__(self):
            self.attempt = 0
        
        def execute(self):
            self.attempt += 1
            if self.attempt < 3:
                raise Exception(f"Simulated failure {self.attempt}")
            return "Success!"
    
    flaky_op = FlakyOperation()
    
    try:
        result = retry_operation(flaky_op.execute, max_retries=3, delay=0.5)
        print(f"Operation result: {result}")
    except Exception as e:
        print(f"Operation failed after retries: {e}")


# ============================================
# 6.5 Circuit Breaker Pattern
# ============================================
class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func):
        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                print("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - rejecting calls")
        
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
                print(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise


def circuit_breaker_example():
    """Demonstrate circuit breaker pattern"""
    print("\n=== Circuit Breaker Pattern ===")
    
    breaker = CircuitBreaker(failure_threshold=3, timeout=5)
    
    # Simulate failing operation
    def failing_operation():
        raise Exception("Operation failed")
    
    # Successful operation
    def successful_operation():
        return "Success!"
    
    # Test circuit breaker
    print("Testing circuit breaker with failing operations:")
    for i in range(5):
        try:
            result = breaker.call(failing_operation)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
    
    print(f"\nCircuit breaker state: {breaker.state}")
    print(f"Failure count: {breaker.failure_count}")


# ============================================
# 6.6 Validation with Early Exit
# ============================================
def validate_and_process(df):
    """Validate data before expensive processing"""
    # Check row count
    row_count = df.count()
    if row_count == 0:
        raise ValueError("DataFrame is empty")
    
    print(f"✓ Row count validation passed: {row_count} rows")
    
    # Check required columns
    required_cols = ["id", "name", "value"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ Column validation passed")
    
    # Check data quality
    null_count = df.filter(col("id").isNull()).count()
    if null_count > 0:
        raise ValueError(f"Found {null_count} null IDs")
    
    print(f"✓ Null validation passed")
    
    # Check value ranges
    invalid_values = df.filter(col("value") < 0).count()
    if invalid_values > 0:
        raise ValueError(f"Found {invalid_values} negative values")
    
    print(f"✓ Value range validation passed")
    
    # Proceed with processing
    print("All validations passed - proceeding with processing")
    return df.withColumn("processed_value", col("value") * 2)


def early_exit_example():
    """Demonstrate validation with early exit"""
    print("\n=== Validation with Early Exit Pattern ===")
    
    # Valid data
    df_valid = spark.createDataFrame([
        (1, "Alice", 100),
        (2, "Bob", 200),
        (3, "Charlie", 150)
    ], ["id", "name", "value"])
    
    print("Processing valid data:")
    try:
        result = validate_and_process(df_valid)
        result.show()
    except ValueError as e:
        print(f"Validation failed: {e}")
    
    # Invalid data (with negative value)
    df_invalid = spark.createDataFrame([
        (1, "Alice", 100),
        (2, "Bob", -50),  # Invalid
        (3, "Charlie", 150)
    ], ["id", "name", "value"])
    
    print("\nProcessing invalid data:")
    try:
        result = validate_and_process(df_invalid)
        result.show()
    except ValueError as e:
        print(f"Validation failed: {e}")


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("PySpark Error Handling Patterns Examples")
    print("=" * 60)
    
    try_catch_example()
    udf_error_handling_example()
    dlq_example()
    retry_example()
    circuit_breaker_example()
    early_exit_example()
    
    spark.stop()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
