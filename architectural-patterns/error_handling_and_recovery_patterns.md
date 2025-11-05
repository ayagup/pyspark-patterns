### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 7:15:14 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Error Handling & Recovery Patterns
> - Fault Tolerance Pattern
> - Lineage-based Recovery Pattern
> - Checkpoint Recovery Pattern
> - Task Retry Pattern
> - Stage Retry Pattern
> - Dead Letter Queue Pattern
> - Circuit Breaker Pattern
> - Graceful Degradation Pattern
> - Failover Pattern
> - Backup Pattern
> - Validation Pattern
> - Data Quality Check Pattern
> ```

#### _Assistant_

> I'll generate comprehensive PySpark programs demonstrating each error handling and recovery pattern.
> 
> ```python error_handling_patterns/fault_tolerance_pattern.py
> """
> Fault Tolerance Pattern
> Demonstrates built-in fault tolerance using RDD transformations and replication
> """
> from pyspark.sql import SparkSession
> from pyspark.storagelevel import StorageLevel
> 
> def fault_tolerance_pattern():
>     spark = SparkSession.builder \
>         .appName("FaultTolerancePattern") \
>         .config("spark.task.maxFailures", "4") \
>         .config("spark.speculation", "true") \
>         .getOrCreate()
>     
>     try:
>         # Create RDD with replication for fault tolerance
>         data = spark.sparkContext.parallelize(range(1, 1001), numSlices=10)
>         
>         # Persist with replication across nodes
>         data.persist(StorageLevel.MEMORY_AND_DISK_2)
>         
>         # Transformations are lazy and fault-tolerant via lineage
>         result = data.map(lambda x: x * 2) \
>                      .filter(lambda x: x % 3 == 0) \
>                      .reduce(lambda a, b: a + b)
>         
>         print(f"Fault-tolerant result: {result}")
>         
>         # DataFrame API with automatic fault tolerance
>         df = spark.range(1, 1001)
>         df.write.mode("overwrite").parquet("/tmp/fault_tolerant_output")
>         
>         data.unpersist()
>         
>     except Exception as e:
>         print(f"Error occurred: {e}")
>         # Spark automatically retries failed tasks
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     fault_tolerance_pattern()
> ```
> 
> ```python error_handling_patterns/lineage_recovery_pattern.py
> """
> Lineage-based Recovery Pattern
> Uses RDD lineage graph for automatic recovery without checkpointing
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, when
> 
> def lineage_recovery_pattern():
>     spark = SparkSession.builder \
>         .appName("LineageRecoveryPattern") \
>         .config("spark.task.maxFailures", "3") \
>         .getOrCreate()
>     
>     try:
>         # Build complex transformation lineage
>         df = spark.range(1, 10001)
>         
>         # Each transformation adds to lineage graph
>         step1 = df.withColumn("doubled", col("id") * 2)
>         step2 = step1.withColumn("status", 
>                                  when(col("doubled") % 2 == 0, "even")
>                                  .otherwise("odd"))
>         step3 = step2.filter(col("doubled") > 100)
>         step4 = step3.groupBy("status").count()
>         
>         # If any partition fails, Spark recomputes from lineage
>         result = step4.collect()
>         
>         print("Lineage-based recovery successful:")
>         for row in result:
>             print(f"  {row.status}: {row['count']}")
>         
>         # View lineage graph
>         print(f"\nExecution plan:\n{step4._jdf.queryExecution().toString()}")
>         
>     except Exception as e:
>         print(f"Error: {e}")
>         # Spark uses lineage to recover failed partitions
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     lineage_recovery_pattern()
> ```
> 
> ```python error_handling_patterns/checkpoint_recovery_pattern.py
> """
> Checkpoint Recovery Pattern
> Uses checkpointing for long lineage chains and streaming applications
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import os
> 
> def checkpoint_recovery_pattern():
>     spark = SparkSession.builder \
>         .appName("CheckpointRecoveryPattern") \
>         .getOrCreate()
>     
>     # Set checkpoint directory
>     checkpoint_dir = "/tmp/spark_checkpoints"
>     os.makedirs(checkpoint_dir, exist_ok=True)
>     spark.sparkContext.setCheckpointDir(checkpoint_dir)
>     
>     try:
>         # RDD checkpointing for long lineage chains
>         rdd = spark.sparkContext.parallelize(range(1, 1001), 10)
>         
>         # Build long lineage chain
>         for i in range(50):
>             rdd = rdd.map(lambda x: x + 1)
>         
>         # Checkpoint to break lineage and save state
>         rdd.checkpoint()
>         
>         # Force checkpoint by executing action
>         count = rdd.count()
>         print(f"RDD count after checkpoint: {count}")
>         
>         # DataFrame checkpointing
>         df = spark.range(1, 10001)
>         
>         for i in range(20):
>             df = df.withColumn(f"col_{i}", col("id") * (i + 1))
>         
>         # Eager checkpoint (truncates lineage immediately)
>         df_checkpointed = df.checkpoint(eager=True)
>         
>         result = df_checkpointed.count()
>         print(f"DataFrame count after checkpoint: {result}")
>         
>     except Exception as e:
>         print(f"Error: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     checkpoint_recovery_pattern()
> ```
> 
> ```python error_handling_patterns/task_retry_pattern.py
> """
> Task Retry Pattern
> Configures automatic task-level retries with backoff
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import udf
> from pyspark.sql.types import IntegerType
> import random
> 
> def task_retry_pattern():
>     spark = SparkSession.builder \
>         .appName("TaskRetryPattern") \
>         .config("spark.task.maxFailures", "4") \
>         .config("spark.task.reaper.enabled", "true") \
>         .config("spark.task.reaper.killTimeout", "30s") \
>         .getOrCreate()
>     
>     # UDF that randomly fails to simulate failures
>     attempt_counter = {"count": 0}
>     
>     @udf(returnType=IntegerType())
>     def unstable_operation(value):
>         attempt_counter["count"] += 1
>         # Fail 30% of the time initially
>         if random.random() < 0.3 and attempt_counter["count"] < 50:
>             raise Exception("Simulated task failure")
>         return value * 2
>     
>     try:
>         df = spark.range(1, 1001)
>         
>         # Apply unstable operation - tasks will retry on failure
>         result_df = df.withColumn("result", unstable_operation("id"))
>         
>         # Trigger execution - failed tasks will be retried
>         count = result_df.count()
>         print(f"Successfully processed {count} records")
>         print(f"Total attempts (including retries): {attempt_counter['count']}")
>         
>         # Show sample results
>         result_df.show(10)
>         
>     except Exception as e:
>         print(f"Task retry limit exceeded: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     task_retry_pattern()
> ```
> 
> ```python error_handling_patterns/stage_retry_pattern.py
> """
> Stage Retry Pattern
> Handles stage-level failures with automatic retry logic
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, sum as _sum
> import time
> 
> def stage_retry_pattern():
>     spark = SparkSession.builder \
>         .appName("StageRetryPattern") \
>         .config("spark.stage.maxConsecutiveAttempts", "4") \
>         .config("spark.blacklist.enabled", "true") \
>         .config("spark.blacklist.task.maxTaskAttemptsPerNode", "2") \
>         .getOrCreate()
>     
>     try:
>         # Create data that requires shuffle (triggers stage boundary)
>         df1 = spark.range(1, 10001).withColumn("key", col("id") % 100)
>         df2 = spark.range(1, 10001).withColumn("key", col("id") % 100)
>         
>         # Join creates multiple stages
>         joined = df1.alias("a").join(df2.alias("b"), "key")
>         
>         # Aggregation creates another stage
>         result = joined.groupBy("key") \
>                        .agg(_sum(col("a.id")).alias("sum_a"),
>                             _sum(col("b.id")).alias("sum_b"))
>         
>         # If any stage fails, Spark retries the entire stage
>         start_time = time.time()
>         final_count = result.count()
>         elapsed_time = time.time() - start_time
>         
>         print(f"Processed {final_count} groups in {elapsed_time:.2f} seconds")
>         print("Stage retry pattern completed successfully")
>         
>         # Show execution plan with stages
>         result.explain(extended=True)
>         
>     except Exception as e:
>         print(f"Stage retry failed: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     stage_retry_pattern()
> ```
> 
> ```python error_handling_patterns/dead_letter_queue_pattern.py
> """
> Dead Letter Queue Pattern
> Separates valid and invalid records for later processing
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, when, current_timestamp
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType
> import datetime
> 
> def dead_letter_queue_pattern():
>     spark = SparkSession.builder \
>         .appName("DeadLetterQueuePattern") \
>         .getOrCreate()
>     
>     try:
>         # Sample data with some invalid records
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("value", StringType(), True),
>             StructField("amount", IntegerType(), True)
>         ])
>         
>         data = [
>             (1, "valid", 100),
>             (2, "valid", 200),
>             (3, None, 300),  # Invalid: null value
>             (4, "valid", -50),  # Invalid: negative amount
>             (5, "valid", 500),
>             (6, "", 600),  # Invalid: empty string
>             (7, "valid", 700)
>         ]
>         
>         df = spark.createDataFrame(data, schema)
>         
>         # Define validation rules
>         df_validated = df.withColumn(
>             "is_valid",
>             when((col("value").isNotNull()) & 
>                  (col("value") != "") & 
>                  (col("amount") > 0), True)
>             .otherwise(False)
>         ).withColumn("validation_timestamp", current_timestamp())
>         
>         # Separate valid and invalid records
>         valid_records = df_validated.filter(col("is_valid") == True)
>         invalid_records = df_validated.filter(col("is_valid") == False) \
>             .withColumn("error_reason", 
>                        when(col("value").isNull(), "NULL_VALUE")
>                        .when(col("value") == "", "EMPTY_VALUE")
>                        .when(col("amount") <= 0, "INVALID_AMOUNT")
>                        .otherwise("UNKNOWN_ERROR"))
>         
>         # Process valid records
>         print("Valid Records:")
>         valid_records.show()
>         valid_records.write.mode("overwrite") \
>             .parquet("/tmp/dlq_valid_records")
>         
>         # Send invalid records to Dead Letter Queue
>         print("\nDead Letter Queue (Invalid Records):")
>         invalid_records.show()
>         invalid_records.write.mode("overwrite") \
>             .parquet("/tmp/dlq_invalid_records")
>         
>         # Summary statistics
>         total_count = df.count()
>         valid_count = valid_records.count()
>         invalid_count = invalid_records.count()
>         
>         print(f"\nSummary:")
>         print(f"  Total records: {total_count}")
>         print(f"  Valid records: {valid_count}")
>         print(f"  Invalid records: {invalid_count}")
>         print(f"  Success rate: {(valid_count/total_count)*100:.2f}%")
>         
>     except Exception as e:
>         print(f"Error: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     dead_letter_queue_pattern()
> ```
> 
> ```python error_handling_patterns/circuit_breaker_pattern.py
> """
> Circuit Breaker Pattern
> Prevents cascading failures by failing fast when error threshold is exceeded
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import udf, col
> from pyspark.sql.types import IntegerType, StringType
> import time
> from datetime import datetime, timedelta
> 
> class CircuitBreaker:
>     def __init__(self, failure_threshold=5, timeout=60):
>         self.failure_threshold = failure_threshold
>         self.timeout = timeout
>         self.failure_count = 0
>         self.last_failure_time = None
>         self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
>     
>     def call(self, func, *args, **kwargs):
>         if self.state == "OPEN":
>             if self._should_attempt_reset():
>                 self.state = "HALF_OPEN"
>             else:
>                 raise Exception("Circuit breaker is OPEN - failing fast")
>         
>         try:
>             result = func(*args, **kwargs)
>             self._on_success()
>             return result
>         except Exception as e:
>             self._on_failure()
>             raise e
>     
>     def _on_success(self):
>         self.failure_count = 0
>         self.state = "CLOSED"
>     
>     def _on_failure(self):
>         self.failure_count += 1
>         self.last_failure_time = datetime.now()
>         
>         if self.failure_count >= self.failure_threshold:
>             self.state = "OPEN"
>     
>     def _should_attempt_reset(self):
>         return (datetime.now() - self.last_failure_time).seconds >= self.timeout
> 
> def circuit_breaker_pattern():
>     spark = SparkSession.builder \
>         .appName("CircuitBreakerPattern") \
>         .getOrCreate()
>     
>     # Initialize circuit breaker
>     circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=10)
>     failure_counter = {"count": 0}
>     
>     def risky_operation(value):
>         """Simulates an operation that may fail"""
>         failure_counter["count"] += 1
>         
>         # Simulate failures for first 5 calls
>         if failure_counter["count"] <= 5:
>             raise Exception(f"External service failure #{failure_counter['count']}")
>         
>         return value * 2
>     
>     def safe_risky_operation(value):
>         """Wrapper with circuit breaker protection"""
>         try:
>             return circuit_breaker.call(risky_operation, value)
>         except Exception as e:
>             print(f"Circuit breaker caught error: {e}")
>             return -1  # Fallback value
>     
>     try:
>         # Register UDF with circuit breaker
>         safe_udf = udf(safe_risky_operation, IntegerType())
>         
>         df = spark.range(1, 20)
>         
>         # Process with circuit breaker protection
>         result_df = df.withColumn("result", safe_udf(col("id"))) \
>                       .withColumn("status",
>                                  when(col("result") == -1, "FAILED")
>                                  .otherwise("SUCCESS"))
>         
>         print("Processing with Circuit Breaker:")
>         result_df.show()
>         
>         # Show circuit breaker state
>         print(f"\nCircuit Breaker State: {circuit_breaker.state}")
>         print(f"Failure Count: {circuit_breaker.failure_count}")
>         
>         # Wait for timeout and retry
>         print("\nWaiting for circuit breaker timeout...")
>         time.sleep(11)
>         
>         # Reset counter for successful processing
>         failure_counter["count"] = 10
>         
>         result_df2 = df.withColumn("result", safe_udf(col("id"))) \
>                        .withColumn("status",
>                                   when(col("result") == -1, "FAILED")
>                                   .otherwise("SUCCESS"))
>         
>         print("\nAfter timeout - Circuit breaker reset:")
>         result_df2.show()
>         print(f"Circuit Breaker State: {circuit_breaker.state}")
>         
>     except Exception as e:
>         print(f"Error: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     circuit_breaker_pattern()
> ```
> 
> ```python error_handling_patterns/graceful_degradation_pattern.py
> """
> Graceful Degradation Pattern
> Provides fallback mechanisms when operations fail
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import udf, col, lit, when
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType
> import random
> 
> def graceful_degradation_pattern():
>     spark = SparkSession.builder \
>         .appName("GracefulDegradationPattern") \
>         .getOrCreate()
>     
>     # Simulate external service calls with fallbacks
>     def enrich_data_primary(id_value):
>         """Primary enrichment service (may fail)"""
>         if random.random() < 0.3:  # 30% failure rate
>             raise Exception("Primary service unavailable")
>         return f"PRIMARY_ENRICHED_{id_value}"
>     
>     def enrich_data_secondary(id_value):
>         """Secondary enrichment service (backup)"""
>         if random.random() < 0.1:  # 10% failure rate
>             raise Exception("Secondary service unavailable")
>         return f"SECONDARY_ENRICHED_{id_value}"
>     
>     def enrich_data_cached(id_value):
>         """Cached/default fallback"""
>         return f"CACHED_DEFAULT_{id_value}"
>     
>     @udf(returnType=StringType())
>     def enrich_with_fallback(id_value):
>         """Enrichment with graceful degradation"""
>         try:
>             # Try primary service
>             return enrich_data_primary(id_value)
>         except Exception as e1:
>             try:
>                 # Fallback to secondary service
>                 return enrich_data_secondary(id_value)
>             except Exception as e2:
>                 # Fallback to cached/default value
>                 return enrich_data_cached(id_value)
>     
>     try:
>         df = spark.range(1, 101)
>         
>         # Apply enrichment with graceful degradation
>         enriched_df = df.withColumn("enriched_value", enrich_with_fallback(col("id"))) \
>                         .withColumn("source",
>                                    when(col("enriched_value").startswith("PRIMARY"), "PRIMARY")
>                                    .when(col("enriched_value").startswith("SECONDARY"), "SECONDARY")
>                                    .otherwise("CACHED"))
>         
>         # Analyze degradation statistics
>         print("Graceful Degradation Results:")
>         enriched_df.show(20)
>         
>         print("\nService Usage Statistics:")
>         enriched_df.groupBy("source").count().show()
>         
>         # Calculate availability metrics
>         total_count = enriched_df.count()
>         source_stats = enriched_df.groupBy("source").count().collect()
>         
>         print("\nDegradation Metrics:")
>         for row in source_stats:
>             percentage = (row['count'] / total_count) * 100
>             print(f"  {row.source}: {row['count']} ({percentage:.2f}%)")
>         
>         # Write results with degradation metadata
>         enriched_df.write.mode("overwrite") \
>             .partitionBy("source") \
>             .parquet("/tmp/graceful_degradation_output")
>         
>     except Exception as e:
>         print(f"Error: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     graceful_degradation_pattern()
> ```
> 
> ```python error_handling_patterns/failover_pattern.py
> """
> Failover Pattern
> Automatically switches to backup resources when primary fails
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, lit
> import os
> 
> def failover_pattern():
>     spark = SparkSession.builder \
>         .appName("FailoverPattern") \
>         .getOrCreate()
>     
>     # Define primary and backup data sources
>     primary_path = "/tmp/primary_data"
>     backup_path = "/tmp/backup_data"
>     fallback_path = "/tmp/fallback_data"
>     
>     # Create sample data in different locations
>     try:
>         # Setup: Create backup and fallback data
>         backup_data = spark.range(1, 101).withColumn("source", lit("BACKUP"))
>         backup_data.write.mode("overwrite").parquet(backup_path)
>         
>         fallback_data = spark.range(1, 51).withColumn("source", lit("FALLBACK"))
>         fallback_data.write.mode("overwrite").parquet(fallback_path)
>         
>         # Simulate primary source failure by not creating it
>         # os.makedirs(primary_path, exist_ok=True)
>         
>     except Exception as e:
>         print(f"Setup error: {e}")
>     
>     def load_data_with_failover(primary, backup, fallback):
>         """Load data with automatic failover"""
>         data_sources = [
>             ("PRIMARY", primary),
>             ("BACKUP", backup),
>             ("FALLBACK", fallback)
>         ]
>         
>         for source_name, source_path in data_sources:
>             try:
>                 print(f"Attempting to load from {source_name}: {source_path}")
>                 df = spark.read.parquet(source_path)
>                 print(f"Successfully loaded from {source_name}")
>                 return df.withColumn("loaded_from", lit(source_name))
>             except Exception as e:
>                 print(f"Failed to load from {source_name}: {e}")
>                 continue
>         
>         raise Exception("All data sources failed")
>     
>     try:
>         # Attempt to load with automatic failover
>         df = load_data_with_failover(primary_path, backup_path, fallback_path)
>         
>         print("\nData loaded successfully:")
>         df.show(10)
>         
>         print("\nSource distribution:")
>         df.groupBy("loaded_from").count().show()
>         
>         # Process the data regardless of source
>         result = df.filter(col("id") > 10) \
>                    .selectExpr("id", "source", "loaded_from", "id * 2 as doubled")
>         
>         print("\nProcessed results:")
>         result.show(10)
>         
>         # Write with source tracking
>         result.write.mode("overwrite") \
>             .partitionBy("loaded_from") \
>             .parquet("/tmp/failover_output")
>         
>         print(f"\nTotal records processed: {result.count()}")
>         
>     except Exception as e:
>         print(f"Failover error: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     failover_pattern()
> ```
> 
> ```python error_handling_patterns/backup_pattern.py
> """
> Backup Pattern
> Creates periodic backups with versioning and recovery capabilities
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, current_timestamp, lit
> from datetime import datetime
> import os
> 
> def backup_pattern():
>     spark = SparkSession.builder \
>         .appName("BackupPattern") \
>         .getOrCreate()
>     
>     base_backup_path = "/tmp/backups"
>     current_data_path = "/tmp/current_data"
>     
>     def create_backup(df, backup_type="full"):
>         """Create versioned backup with timestamp"""
>         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
>         backup_path = f"{base_backup_path}/{backup_type}_{timestamp}"
>         
>         # Add backup metadata
>         backup_df = df.withColumn("backup_timestamp", current_timestamp()) \
>                       .withColumn("backup_type", lit(backup_type))
>         
>         # Write backup with version
>         backup_df.write.mode("overwrite").parquet(backup_path)
>         
>         print(f"Backup created: {backup_path}")
>         return backup_path
>     
>     def restore_backup(backup_path):
>         """Restore from specific backup version"""
>         print(f"Restoring from: {backup_path}")
>         return spark.read.parquet(backup_path)
>     
>     def list_backups():
>         """List all available backups"""
>         if os.path.exists(base_backup_path):
>             backups = [d for d in os.listdir(base_backup_path) 
>                       if os.path.isdir(os.path.join(base_backup_path, d))]
>             return sorted(backups, reverse=True)
>         return []
>     
>     try:
>         # Create initial dataset
>         df_v1 = spark.range(1, 101).withColumn("version", lit(1))
>         df_v1.write.mode("overwrite").parquet(current_data_path)
>         
>         # Create full backup
>         backup1 = create_backup(df_v1, "full")
>         
>         # Modify data (version 2)
>         df_v2 = spark.range(1, 201).withColumn("version", lit(2))
>         df_v2.write.mode("overwrite").parquet(current_data_path)
>         
>         # Create incremental backup
>         backup2 = create_backup(df_v2, "incremental")
>         
>         # Simulate data corruption
>         print("\nSimulating data corruption...")
>         corrupted_df = spark.range(1, 10).withColumn("version", lit(-1))
>         corrupted_df.write.mode("overwrite").parquet(current_data_path)
>         
>         # List available backups
>         print("\nAvailable backups:")
>         backups = list_backups()
>         for i, backup in enumerate(backups):
>             print(f"  {i+1}. {backup}")
>         
>         # Restore from latest backup
>         if backups:
>             latest_backup = os.path.join(base_backup_path, backups[0])
>             restored_df = restore_backup(latest_backup)
>             
>             print("\nRestored data:")
>             restored_df.select("id", "version", "backup_type").show(10)
>             
>             # Verify restoration
>             print(f"\nRestored record count: {restored_df.count()}")
>             
>             # Save restored data back to current
>             restored_df.drop("backup_timestamp", "backup_type") \
>                       .write.mode("overwrite").parquet(current_data_path)
>             
>             print("Data successfully restored from backup")
>         
>         # Backup rotation (keep only last N backups)
>         max_backups = 5
>         if len(backups) > max_backups:
>             print(f"\nRotating backups (keeping last {max_backups})...")
>             for old_backup in backups[max_backups:]:
>                 old_path = os.path.join(base_backup_path, old_backup)
>                 print(f"  Removing old backup: {old_backup}")
>                 # In production, actually remove the directory
>         
>     except Exception as e:
>         print(f"Error: {e}")
>     finally:
>         spark.stop()
> 
> if __name__ == "__main__":
>     backup_pattern()
> ```
> 
> ```python error_handling_patterns/validation_pattern.py
> """
> Validation Pattern
> Validates data integrity and business rules before processing
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, when, length, regexp_extract, size, split
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
> 
> def validation_pattern():
>     spark = SparkSession.builder \
>         .appName("ValidationPattern") \
>         .getOrCreate()
>     
>     # Define schema with constraints
>     schema = StructType([
>         StructField("id", IntegerType(), False),  # Not nullable
>         StructField("email", StringType(), True),
>         StructField("age", IntegerType(), True),
>         StructField("salary", DoubleType(), True),
>         StructField("status", StringType(), True)
>     ])
>     
>     # Sample data with validation issues
>     data = [
>         (1, "user1@example.com", 25, 50000.0, "ACTIVE"),
>         (2, "invalid-email", 30, 60000.0, "ACTIVE"),
>         (3, "user3@example.com", -5, 55000.0, "ACTIVE"),  # Invalid age
>         (4, "user4@example.com", 150, 70000.0, "ACTIVE"),  # Invalid age
>         (5, "user5@example.com", 35, -1000.0, "ACTIVE"),  # Invalid salary
>         (6, None, 40, 80000.0, "ACTIVE"),  # Missing email
>         (7, "user7@example.com", 28, 45000.0, "INVALID_STATUS"),
>         (8, "user8@example.com", 32, 90000.0, "ACTIVE"),
>     ]
>     
>     df = spark.createDataFrame(data, schema)
>     
>     # Define validation rules
>     validations = df.withColumn("valid_email",
>                                when(col("email").isNotNull() & 
>                                     col("email").rlike("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"),
>                                     True).otherwise(False)) \
>                     .withColumn("valid_age",
>                                when((col("age") >= 0) & (col("age") <= 120), True)
>                                .otherwise(False)) \
>                     .withColumn("valid_salary",
>                                when(col("salary") >= 0, True).otherwise(False)) \
>                     .withColumn("valid_status",
>                                when(col("status").isin(["ACTIVE", "INACTIVE", "PENDING"]), True)
>                                .otherwise(False))
>     
>     # Create overall validity flag
>     validated_df = validations.withColumn("is_valid",
>                                          col("valid_email") & 
>                                          col("valid_age") & 
>                                          col("valid_salary") & 
>                                          col("valid_status"))
>     
>     # Separate valid and invalid records
>     valid_records = validated_df.filter(col("is_valid") == True)
>     invalid_records = validated_df.filter(col("is_valid") == False)
>     
>     # Create detailed error messages
>     invalid_with_errors = invalid_records.withColumn("errors",
>         when(~col("valid_email"), "INVALID_EMAIL;").otherwise("") +
>         when(~col("valid_age"), "INVALID_AGE;").otherwise("") +
>         when(~col("valid_salary"), "INVALID_SALARY;").otherwise("") +
>         when(~col("valid_status"), "INVALID_STATUS;").otherwise("")
>     )
>     
>     print("=" * 80)
>     print("VALIDATION PATTERN RESULTS")
>     print("=" * 80)
>     
>     print("\nVALID RECORDS:")
>     valid_records.select("id", "email", "age", "salary", "status").show()
>     
>     print("\nINVALID RECORDS WITH ERRORS:")
>     invalid_with_errors.select("id", "email", "age", "salary", "status", "errors").show(truncate=False)
>     
>     # Validation summary
>     total_count = df.count()
>     valid_count = valid_records.count()
>     invalid_count = invalid_records.count()
>     
>     print("\nVALIDATION SUMMARY:")
>     print(f"  Total records: {total_count}")
>     print(f"  Valid records: {valid_count}")
>     print(f"  Invalid records: {invalid_count}")
>     print(f"  Validation rate: {(valid_count/total_count)*100:.2f}%")
>     
>     # Error breakdown
>     print("\nERROR BREAKDOWN:")
>     invalid_with_errors.

#### _User_

> continue exactly where you left off

#### _Assistant_

> ```python error_handling_patterns/validation_pattern.py
> """
> Validation Pattern
> Validates data integrity and business rules before processing
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, when, length, regexp_extract, size, split
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
> 
> def validation_pattern():
>     spark = SparkSession.builder \
>         .appName("ValidationPattern") \
>         .getOrCreate()
>     
>     # Define schema with constraints
>     schema = StructType([
>         StructField("id", IntegerType(), False),  # Not nullable
>         StructField("email", StringType(), True),
>         StructField("age", IntegerType(), True),
>         StructField("salary", DoubleType(), True),
>         StructField("status", StringType(), True)
>     ])
>     
>     # Sample data with validation issues
>     data = [
>         (1, "user1@example.com", 25, 50000.0, "ACTIVE"),
>         (2, "invalid-email", 30, 60000.0, "ACTIVE"),
>         (3, "user3@example.com", -5, 55000.0, "ACTIVE"),  # Invalid age
>         (4, "user4@example.com", 150, 70000.0, "ACTIVE"),  # Invalid age
>         (5, "user5@example.com", 35, -1000.0, "ACTIVE"),  # Invalid salary
>         (6, None, 40, 80000.0, "ACTIVE"),  # Missing email
>         (7, "user7@example.com", 28, 45000.0, "INVALID_STATUS"),
>         (8, "user8@example.com", 32, 90000.0, "ACTIVE"),
>     ]
>     
>     df = spark.createDataFrame(data, schema)
>     
>     # Define validation rules
>     validations = df.withColumn("valid_email",
>                                when(col("email").isNotNull() & 
>                                     col("email").rlike("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"),
>                                     True).otherwise(False)) \
>                     .withColumn("valid_age",
>                                when((col("age") >= 0) & (col("age") <= 120), True)
>                                .otherwise(False)) \
>                     .withColumn("valid_salary",
>                                when(col("salary") >= 0, True).otherwise(False)) \
>                     .withColumn("valid_status",
>                                when(col("status").isin(["ACTIVE", "INACTIVE", "PENDING"]), True)
>                                .otherwise(False))
>     
>     # Create overall validity flag
>     validated_df = validations.withColumn("is_valid",
>                                          col("valid_email") & 
>                                          col("valid_age") & 
>                                          col("valid_salary") & 
>                                          col("valid_status"))
>     
>     # Separate valid and invalid records
>     valid_records = validated_df.filter(col("is_valid") == True)
>     invalid_records = validated_df.filter(col("is_valid") == False)
>     
>     # Create detailed error messages
>     invalid_with_errors = invalid_records.withColumn("errors",
>         when(~col("valid_email"), "INVALID_EMAIL;").otherwise("") +
>         when(~col("valid_age"), "INVALID_AGE;").otherwise("") +
>         when(~col("valid_salary"), "INVALID_SALARY;").otherwise("") +
>         when(~col("valid_status"), "INVALID_STATUS;").otherwise("")
>     )
>     
>     print("=" * 80)
>     print("VALIDATION PATTERN RESULTS")
>     print("=" * 80)
>     
>     print("\nVALID RECORDS:")
>     valid_records.select("id", "email", "age", "salary", "status").show()
>     
>     print("\nINVALID RECORDS WITH ERRORS:")
>     invalid_with_errors.select("id", "email", "age", "salary", "status", "errors").show(truncate=False)
>     
>     # Validation summary
>     total_count = df.count()
>     valid_count = valid_records.count()
>     invalid_count = invalid_records.count()
>     
>     print("\nVALIDATION SUMMARY:")
>     print(f"  Total records: {total_count}")
>     print(f"  Valid records: {valid_count}")
>     print(f"  Invalid records: {invalid_count}")
>     print(f"  Validation rate: {(valid_count/total_count)*100:.2f}%")
>     
>     # Error breakdown
>     print("\nERROR BREAKDOWN:")
>     error_stats = invalid_with_errors.select(
>         col("valid_email").alias("email_valid"),
>         col("valid_age").alias("age_valid"),
>         col("valid_salary").alias("salary_valid"),
>         col("valid_status").alias("status_valid")
>     )
>     
>     print(f"  Email errors: {error_stats.filter(~col('email_valid')).count()}")
>     print(f"  Age errors: {error_stats.filter(~col('age_valid')).count()}")
>     print(f"  Salary errors: {error_stats.filter(~col('salary_valid')).count()}")
>     print(f"  Status errors: {error_stats.filter(~col('status_valid')).count()}")
>     
>     # Write validated data
>     valid_records.select("id", "email", "age", "salary", "status") \
>         .write.mode("overwrite").parquet("/tmp/validated_records")
>     
>     invalid_with_errors.select("id", "email", "age", "salary", "status", "errors") \
>         .write.mode("overwrite").parquet("/tmp/validation_errors")
>     
>     try:
>         spark.stop()
>     except:
>         pass
> 
> if __name__ == "__main__":
>     validation_pattern()
> ```
> 
> ```python error_handling_patterns/data_quality_check_pattern.py
> """
> Data Quality Check Pattern
> Comprehensive data quality checks with metrics and reporting
> """
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import (col, count, when, isnan, isnull, 
>                                    sum as _sum, avg, stddev, min as _min, 
>                                    max as _max, countDistinct, current_timestamp)
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType
> from datetime import datetime, date
> 
> def data_quality_check_pattern():
>     spark = SparkSession.builder \
>         .appName("DataQualityCheckPattern") \
>         .getOrCreate()
>     
>     # Sample dataset with quality issues
>     schema = StructType([
>         StructField("customer_id", IntegerType(), True),
>         StructField("name", StringType(), True),
>         StructField("email", StringType(), True),
>         StructField("age", IntegerType(), True),
>         StructField("purchase_amount", DoubleType(), True),
>         StructField("purchase_date", DateType(), True),
>         StructField("country", StringType(), True)
>     ])
>     
>     data = [
>         (1, "John Doe", "john@example.com", 25, 100.50, date(2024, 1, 15), "USA"),
>         (2, "Jane Smith", "jane@example.com", 30, 250.75, date(2024, 1, 16), "USA"),
>         (3, None, "unknown@example.com", 35, 180.00, date(2024, 1, 17), "Canada"),  # Missing name
>         (4, "Bob Johnson", None, 28, 320.25, date(2024, 1, 18), "USA"),  # Missing email
>         (5, "Alice Brown", "alice@example.com", None, 150.00, date(2024, 1, 19), "UK"),  # Missing age
>         (6, "Charlie Wilson", "charlie@example.com", -5, 200.00, date(2024, 1, 20), "USA"),  # Invalid age
>         (7, "Diana Davis", "diana@example.com", 999, 175.50, date(2024, 1, 21), "Canada"),  # Invalid age
>         (8, "Eve Martinez", "eve@example.com", 45, None, date(2024, 1, 22), "USA"),  # Missing amount
>         (9, "Frank Taylor", "frank@example.com", 38, -50.00, date(2024, 1, 23), "UK"),  # Invalid amount
>         (10, "Grace Lee", "grace@example.com", 42, 500.00, None, "USA"),  # Missing date
>         (1, "John Doe", "john@example.com", 25, 100.50, date(2024, 1, 15), "USA"),  # Duplicate
>         (11, "", "empty@example.com", 29, 220.00, date(2024, 1, 24), "USA"),  # Empty name
>         (12, "Henry Clark", "henry@example.com", 33, 190.00, date(2024, 1, 25), None),  # Missing country
>     ]
>     
>     df = spark.createDataFrame(data, schema)
>     
>     print("=" * 80)
>     print("DATA QUALITY CHECK PATTERN")
>     print("=" * 80)
>     
>     # 1. COMPLETENESS CHECKS
>     print("\n1. COMPLETENESS CHECKS:")
>     print("-" * 80)
>     
>     total_records = df.count()
>     
>     for column in df.columns:
>         null_count = df.filter(col(column).isNull()).count()
>         empty_count = df.filter((col(column) == "") | (col(column).isNull())).count() \
>                         if df.schema[column].dataType == StringType() else null_count
>         
>         completeness_rate = ((total_records - null_count) / total_records) * 100
>         
>         print(f"  {column}:")
>         print(f"    Null count: {null_count}")
>         print(f"    Empty/Null count: {empty_count}")
>         print(f"    Completeness: {completeness_rate:.2f}%")
>     
>     # 2. UNIQUENESS CHECKS
>     print("\n2. UNIQUENESS CHECKS:")
>     print("-" * 80)
>     
>     duplicate_count = df.count() - df.dropDuplicates(["customer_id"]).count()
>     print(f"  Duplicate customer_ids: {duplicate_count}")
>     
>     distinct_count = df.select(countDistinct("customer_id")).collect()[0][0]
>     print(f"  Distinct customer_ids: {distinct_count}")
>     print(f"  Uniqueness rate: {(distinct_count/total_records)*100:.2f}%")
>     
>     # 3. VALIDITY CHECKS
>     print("\n3. VALIDITY CHECKS:")
>     print("-" * 80)
>     
>     # Age validity
>     invalid_age = df.filter((col("age") < 0) | (col("age") > 120)).count()
>     valid_age = df.filter((col("age") >= 0) & (col("age") <= 120)).count()
>     print(f"  Age:")
>     print(f"    Invalid records: {invalid_age}")
>     print(f"    Valid records: {valid_age}")
>     
>     # Purchase amount validity
>     invalid_amount = df.filter((col("purchase_amount") < 0) | 
>                                (col("purchase_amount") > 10000)).count()
>     print(f"  Purchase Amount:")
>     print(f"    Invalid records (< 0 or > 10000): {invalid_amount}")
>     
>     # Email validity
>     email_pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
>     invalid_email = df.filter(col("email").isNotNull() & 
>                               ~col("email").rlike(email_pattern)).count()
>     print(f"  Email:")
>     print(f"    Invalid format: {invalid_email}")
>     
>     # 4. CONSISTENCY CHECKS
>     print("\n4. CONSISTENCY CHECKS:")
>     print("-" * 80)
>     
>     # Check for consistent country values
>     country_values = df.select("country").distinct().collect()
>     expected_countries = ["USA", "Canada", "UK", "Mexico"]
>     
>     print(f"  Country values found: {[row.country for row in country_values]}")
>     
>     # 5. ACCURACY CHECKS (Statistical outliers)
>     print("\n5. ACCURACY CHECKS (Statistical Outliers):")
>     print("-" * 80)
>     
>     stats = df.select(
>         avg("purchase_amount").alias("avg_amount"),
>         stddev("purchase_amount").alias("stddev_amount"),
>         _min("purchase_amount").alias("min_amount"),
>         _max("purchase_amount").alias("max_amount")
>     ).collect()[0]
>     
>     print(f"  Purchase Amount Statistics:")
>     print(f"    Average: ${stats.avg_amount:.2f}")
>     print(f"    Std Dev: ${stats.stddev_amount:.2f}")
>     print(f"    Min: ${stats.min_amount:.2f}")
>     print(f"    Max: ${stats.max_amount:.2f}")
>     
>     # Detect outliers (beyond 2 standard deviations)
>     if stats.stddev_amount:
>         lower_bound = stats.avg_amount - (2 * stats.stddev_amount)
>         upper_bound = stats.avg_amount + (2 * stats.stddev_amount)
>         
>         outliers = df.filter((col("purchase_amount") < lower_bound) | 
>                             (col("purchase_amount") > upper_bound)).count()
>         print(f"    Outliers (±2 std dev): {outliers}")
>     
>     # 6. TIMELINESS CHECKS
>     print("\n6. TIMELINESS CHECKS:")
>     print("-" * 80)
>     
>     current_date = date.today()
>     old_records = df.filter(col("purchase_date") < date(2024, 1, 1)).count()
>     future_records = df.filter(col("purchase_date") > current_date).count()
>     
>     print(f"  Records before 2024: {old_records}")
>     print(f"  Future-dated records: {future_records}")
>     
>     # 7. COMPREHENSIVE QUALITY SCORE
>     print("\n7. OVERALL DATA QUALITY SCORE:")
>     print("-" * 80)
>     
>     quality_df = df.withColumn("quality_score", 
>         (when(col("customer_id").isNotNull(), 1).otherwise(0) +
>          when((col("name").isNotNull()) & (col("name") != ""), 1).otherwise(0) +
>          when(col("email").isNotNull() & col("email").rlike(email_pattern), 1).otherwise(0) +
>          when((col("age") >= 0) & (col("age") <= 120), 1).otherwise(0) +
>          when((col("purchase_amount") >= 0) & (col("purchase_amount") <= 10000), 1).otherwise(0) +
>          when(col("purchase_date").isNotNull(), 1).otherwise(0) +
>          when(col("country").isNotNull(), 1).otherwise(0)) / 7 * 100
>     )
>     
>     avg_quality_score = quality_df.select(avg("quality_score")).collect()[0][0]
>     print(f"  Average Quality Score: {avg_quality_score:.2f}%")
>     
>     # Quality distribution
>     print("\n  Quality Score Distribution:")
>     quality_df.groupBy(
>         when(col("quality_score") == 100, "Perfect (100%)")
>         .when(col("quality_score") >= 80, "Good (80-99%)")
>         .when(col("quality_score") >= 60, "Fair (60-79%)")
>         .otherwise("Poor (<60%)").alias("quality_category")
>     ).count().orderBy("quality_category").show(truncate=False)
>     
>     # 8. DATA QUALITY REPORT
>     print("\n8. DETAILED DATA QUALITY REPORT:")
>     print("-" * 80)
>     
>     quality_report_df = quality_df.withColumn("issues",
>         when(col("customer_id").isNull(), "MISSING_ID;").otherwise("") +
>         when((col("name").isNull()) | (col("name") == ""), "MISSING_NAME;").otherwise("") +
>         when(col("email").isNull() | ~col("email").rlike(email_pattern), "INVALID_EMAIL;").otherwise("") +
>         when((col("age") < 0) | (col("age") > 120) | col("age").isNull(), "INVALID_AGE;").otherwise("") +
>         when((col("purchase_amount") < 0) | col("purchase_amount").isNull(), "INVALID_AMOUNT;").otherwise("") +
>         when(col("purchase_date").isNull(), "MISSING_DATE;").otherwise("") +
>         when(col("country").isNull(), "MISSING_COUNTRY;").otherwise("")
>     ).withColumn("check_timestamp", current_timestamp())
>     
>     # Show records with issues
>     records_with_issues = quality_report_df.filter(col("issues") != "")
>     
>     print(f"\nRecords with quality issues: {records_with_issues.count()}")
>     records_with_issues.select("customer_id", "name", "quality_score", "issues").show(truncate=False)
>     
>     # 9. SAVE QUALITY REPORTS
>     print("\n9. SAVING QUALITY REPORTS:")
>     print("-" * 80)
>     
>     # Save clean records
>     clean_records = quality_report_df.filter(col("quality_score") == 100)
>     clean_records.drop("quality_score", "issues", "check_timestamp") \
>         .write.mode("overwrite").parquet("/tmp/dq_clean_records")
>     print(f"  Clean records saved: {clean_records.count()}")
>     
>     # Save records needing review
>     records_with_issues.write.mode("overwrite").parquet("/tmp/dq_records_with_issues")
>     print(f"  Records with issues saved: {records_with_issues.count()}")
>     
>     # Save quality metrics
>     quality_metrics = spark.createDataFrame([{
>         "check_date": datetime.now(),
>         "total_records": total_records,
>         "clean_records": clean_records.count(),
>         "records_with_issues": records_with_issues.count(),
>         "avg_quality_score": avg_quality_score,
>         "duplicate_count": duplicate_count,
>         "completeness_rate": ((total_records - df.filter(
>             col("customer_id").isNull() |
>             col("name").isNull() |
>             col("email").isNull()
>         ).count()) / total_records) * 100
>     }])
>     
>     quality_metrics.write.mode("append").parquet("/tmp/dq_metrics_history")
>     print("  Quality metrics saved to history")
>     
>     print("\n" + "=" * 80)
>     print("DATA QUALITY CHECK COMPLETE")
>     print("=" * 80)
>     
>     spark.stop()
> 
> if __name__ == "__main__":
>     data_quality_check_pattern()
> ```
> 
> These comprehensive PySpark programs demonstrate all 12 error handling and recovery patterns with real-world scenarios, detailed logging, and production-ready implementations. Each pattern includes:
> 
> - **Practical examples** with realistic data
> - **Comprehensive error handling** mechanisms
> - **Detailed logging and metrics**
> - **Output persistence** for audit trails
> - **Configuration options** for tuning
> - **Comments explaining** the pattern implementation
> 
> You can run each pattern independently to see how different error handling strategies work in PySpark!