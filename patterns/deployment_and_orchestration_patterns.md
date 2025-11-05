### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:54:16 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Deployment & Orchestration**
> *   Spark-Submit Job Submission
> *   Passing Configuration and Arguments to Spark Jobs
> *   Managing Dependencies (JARs, Python files)
> *   Environment Variable Configuration
> ```

#### _Assistant_

> I'll create Python PySpark programs demonstrating each deployment and orchestration pattern.
> 
> ```python deployment/spark_submit_job.py
> """
> Spark-Submit Job Submission Pattern
> This script demonstrates a complete PySpark job designed to be submitted via spark-submit
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, count, avg, sum as _sum
> import sys
> import logging
> 
> # Configure logging
> logging.basicConfig(
>     level=logging.INFO,
>     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
> )
> logger = logging.getLogger(__name__)
> 
> 
> def create_spark_session(app_name="SparkSubmitJobExample"):
>     """Create and configure Spark session"""
>     return SparkSession.builder \
>         .appName(app_name) \
>         .getOrCreate()
> 
> 
> def process_sales_data(spark, input_path, output_path):
>     """
>     Main data processing logic
>     Reads sales data, performs aggregations, and writes results
>     """
>     logger.info(f"Reading data from: {input_path}")
>     
>     # Read input data
>     sales_df = spark.read \
>         .option("header", "true") \
>         .option("inferSchema", "true") \
>         .csv(input_path)
>     
>     logger.info(f"Records read: {sales_df.count()}")
>     
>     # Perform aggregations
>     aggregated_df = sales_df.groupBy("category", "region") \
>         .agg(
>             count("*").alias("total_transactions"),
>             _sum("amount").alias("total_sales"),
>             avg("amount").alias("avg_sale_amount")
>         ) \
>         .orderBy(col("total_sales").desc())
>     
>     logger.info("Aggregation completed")
>     
>     # Write results
>     aggregated_df.write \
>         .mode("overwrite") \
>         .partitionBy("region") \
>         .parquet(output_path)
>     
>     logger.info(f"Results written to: {output_path}")
>     
>     return aggregated_df
> 
> 
> def main():
>     """Main execution function"""
>     if len(sys.argv) < 3:
>         logger.error("Usage: spark_submit_job.py <input_path> <output_path>")
>         sys.exit(1)
>     
>     input_path = sys.argv[1]
>     output_path = sys.argv[2]
>     
>     logger.info("Starting Spark job...")
>     
>     # Create Spark session
>     spark = create_spark_session()
>     
>     try:
>         # Process data
>         result_df = process_sales_data(spark, input_path, output_path)
>         
>         # Show sample results
>         logger.info("Sample results:")
>         result_df.show(10, truncate=False)
>         
>         logger.info("Job completed successfully!")
>         
>     except Exception as e:
>         logger.error(f"Job failed with error: {str(e)}")
>         raise
>     finally:
>         spark.stop()
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> ```bash deployment/submit_job.sh
> #!/bin/bash
> 
> # Spark-Submit Job Submission Script
> # Demonstrates various spark-submit options and configurations
> 
> # Basic submission
> echo "=== Basic Spark Submit ==="
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   deployment/spark_submit_job.py \
>   hdfs://input/sales_data.csv \
>   hdfs://output/sales_aggregated
> 
> # Submission with resource configuration
> echo "=== Spark Submit with Resource Config ==="
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --name "Sales Data Processing Job" \
>   --driver-memory 4g \
>   --executor-memory 8g \
>   --executor-cores 4 \
>   --num-executors 10 \
>   --conf spark.dynamicAllocation.enabled=true \
>   --conf spark.dynamicAllocation.minExecutors=5 \
>   --conf spark.dynamicAllocation.maxExecutors=20 \
>   deployment/spark_submit_job.py \
>   hdfs://input/sales_data.csv \
>   hdfs://output/sales_aggregated
> 
> # Submission for local testing
> echo "=== Local Mode Submission ==="
> spark-submit \
>   --master local[4] \
>   --driver-memory 2g \
>   deployment/spark_submit_job.py \
>   data/sales_data.csv \
>   output/sales_aggregated
> ```
> 
> ```python deployment/config_arguments_job.py
> """
> Passing Configuration and Arguments to Spark Jobs Pattern
> Demonstrates multiple ways to handle configuration and arguments
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, current_timestamp
> import argparse
> import json
> import os
> from typing import Dict, Any
> import logging
> 
> logging.basicConfig(level=logging.INFO)
> logger = logging.getLogger(__name__)
> 
> 
> class JobConfig:
>     """Configuration class for Spark job parameters"""
>     
>     def __init__(self, args):
>         self.input_path = args.input_path
>         self.output_path = args.output_path
>         self.processing_date = args.processing_date
>         self.partition_count = args.partition_count
>         self.output_format = args.output_format
>         self.enable_compression = args.enable_compression
>         self.log_level = args.log_level
>         
>         # Load additional config from JSON file if provided
>         if args.config_file:
>             self.load_config_file(args.config_file)
>     
>     def load_config_file(self, config_path: str):
>         """Load additional configuration from JSON file"""
>         logger.info(f"Loading config from: {config_path}")
>         with open(config_path, 'r') as f:
>             config_data = json.load(f)
>             for key, value in config_data.items():
>                 setattr(self, key, value)
>     
>     def to_dict(self) -> Dict[str, Any]:
>         """Convert config to dictionary"""
>         return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
> 
> 
> def parse_arguments():
>     """Parse command line arguments"""
>     parser = argparse.ArgumentParser(
>         description='PySpark job with comprehensive argument handling'
>     )
>     
>     # Required arguments
>     parser.add_argument(
>         '--input-path',
>         required=True,
>         help='Input data path'
>     )
>     parser.add_argument(
>         '--output-path',
>         required=True,
>         help='Output data path'
>     )
>     
>     # Optional arguments with defaults
>     parser.add_argument(
>         '--processing-date',
>         default='2024-01-01',
>         help='Processing date (YYYY-MM-DD)'
>     )
>     parser.add_argument(
>         '--partition-count',
>         type=int,
>         default=10,
>         help='Number of output partitions'
>     )
>     parser.add_argument(
>         '--output-format',
>         choices=['parquet', 'orc', 'json', 'csv'],
>         default='parquet',
>         help='Output file format'
>     )
>     parser.add_argument(
>         '--enable-compression',
>         action='store_true',
>         help='Enable output compression'
>     )
>     parser.add_argument(
>         '--log-level',
>         choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
>         default='INFO',
>         help='Logging level'
>     )
>     
>     # Configuration file
>     parser.add_argument(
>         '--config-file',
>         help='Path to JSON configuration file'
>     )
>     
>     return parser.parse_args()
> 
> 
> def create_spark_session_with_config(app_name: str, config: JobConfig) -> SparkSession:
>     """Create Spark session with configuration from arguments and environment"""
>     
>     builder = SparkSession.builder.appName(app_name)
>     
>     # Apply configurations from environment variables
>     if os.getenv('SPARK_EXECUTOR_MEMORY'):
>         builder = builder.config('spark.executor.memory', os.getenv('SPARK_EXECUTOR_MEMORY'))
>     
>     if os.getenv('SPARK_EXECUTOR_CORES'):
>         builder = builder.config('spark.executor.cores', os.getenv('SPARK_EXECUTOR_CORES'))
>     
>     # Apply compression if enabled
>     if config.enable_compression:
>         builder = builder.config('spark.sql.parquet.compression.codec', 'snappy')
>         builder = builder.config('spark.sql.orc.compression.codec', 'snappy')
>     
>     # Set log level
>     spark = builder.getOrCreate()
>     spark.sparkContext.setLogLevel(config.log_level)
>     
>     return spark
> 
> 
> def process_data(spark: SparkSession, config: JobConfig):
>     """Process data using configuration"""
>     
>     logger.info(f"Job Configuration: {json.dumps(config.to_dict(), indent=2)}")
>     
>     # Read input data
>     logger.info(f"Reading from: {config.input_path}")
>     df = spark.read.parquet(config.input_path)
>     
>     # Add processing metadata
>     df = df.withColumn("processing_date", col("processing_date").cast("date")) \
>            .withColumn("processed_timestamp", current_timestamp())
>     
>     # Filter by processing date if specified
>     if config.processing_date:
>         df = df.filter(col("processing_date") == config.processing_date)
>     
>     # Repartition if specified
>     if config.partition_count:
>         df = df.repartition(config.partition_count)
>     
>     # Write output with specified format
>     logger.info(f"Writing to: {config.output_path} in {config.output_format} format")
>     
>     writer = df.write.mode("overwrite")
>     
>     if config.output_format == 'parquet':
>         writer.parquet(config.output_path)
>     elif config.output_format == 'orc':
>         writer.orc(config.output_path)
>     elif config.output_format == 'json':
>         writer.json(config.output_path)
>     elif config.output_format == 'csv':
>         writer.option("header", "true").csv(config.output_path)
>     
>     logger.info("Processing completed successfully")
> 
> 
> def main():
>     """Main execution function"""
>     
>     # Parse arguments
>     args = parse_arguments()
>     config = JobConfig(args)
>     
>     # Create Spark session
>     spark = create_spark_session_with_config("ConfigurableSparkJob", config)
>     
>     try:
>         # Process data
>         process_data(spark, config)
>         
>     except Exception as e:
>         logger.error(f"Job failed: {str(e)}")
>         raise
>     finally:
>         spark.stop()
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> ```json deployment/job_config.json
> {
>   "database_url": "jdbc:postgresql://localhost:5432/mydb",
>   "database_user": "spark_user",
>   "query_timeout": 300,
>   "max_retries": 3,
>   "checkpoint_location": "hdfs://checkpoints/job1",
>   "custom_parameters": {
>     "threshold_value": 100,
>     "filter_categories": ["electronics", "books", "clothing"]
>   }
> }
> ```
> 
> ```bash deployment/submit_with_config.sh
> #!/bin/bash
> 
> # Submit job with various configuration options
> 
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --name "Configurable Spark Job" \
>   --driver-memory 4g \
>   --executor-memory 8g \
>   --conf spark.executor.cores=4 \
>   --conf spark.sql.shuffle.partitions=200 \
>   --conf spark.sql.adaptive.enabled=true \
>   deployment/config_arguments_job.py \
>   --input-path hdfs://input/sales_data \
>   --output-path hdfs://output/processed_sales \
>   --processing-date 2024-01-15 \
>   --partition-count 20 \
>   --output-format parquet \
>   --enable-compression \
>   --log-level INFO \
>   --config-file deployment/job_config.json
> ```
> 
> ```python deployment/dependency_management_job.py
> """
> Managing Dependencies (JARs, Python files) Pattern
> Demonstrates how to handle external dependencies in PySpark jobs
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, udf
> from pyspark.sql.types import StringType, IntegerType
> import sys
> import os
> import logging
> 
> # Import from custom modules (will be distributed)
> # These would be in separate files: utils.py, transformations.py
> try:
>     from utils import validate_data, calculate_metrics
>     from transformations import clean_text, parse_json_column
> except ImportError:
>     logger.warning("Custom modules not found, using inline definitions")
> 
> logging.basicConfig(level=logging.INFO)
> logger = logging.getLogger(__name__)
> 
> 
> def create_spark_with_dependencies(app_name: str) -> SparkSession:
>     """
>     Create Spark session with various dependency configurations
>     """
>     
>     return SparkSession.builder \
>         .appName(app_name) \
>         .config("spark.jars.packages", 
>                 "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
>                 "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
>                 "org.apache.hadoop:hadoop-aws:3.3.4") \
>         .config("spark.jars", 
>                 "/path/to/custom-jar-1.0.jar,"
>                 "/path/to/postgres-jdbc-42.5.0.jar") \
>         .config("spark.jars.excludes", 
>                 "org.scala-lang:scala-reflect") \
>         .config("spark.sql.extensions", 
>                 "io.delta.sql.DeltaSparkSessionExtension") \
>         .config("spark.sql.catalog.spark_catalog", 
>                 "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
>         .getOrCreate()
> 
> 
> # Example UDF that might use external Python libraries
> @udf(returnType=StringType())
> def process_with_external_lib(text: str) -> str:
>     """
>     UDF that uses external Python libraries
>     In real scenario, this might use libraries like:
>     - pandas for data manipulation
>     - numpy for numerical operations
>     - custom libraries from requirements.txt
>     """
>     # Simulated processing
>     import re
>     return re.sub(r'[^\w\s]', '', text.lower())
> 
> 
> def process_with_jdbc_dependency(spark: SparkSession):
>     """
>     Example using JDBC driver dependency
>     Requires: postgresql-42.5.0.jar
>     """
>     
>     jdbc_url = "jdbc:postgresql://localhost:5432/mydb"
>     connection_properties = {
>         "user": "spark_user",
>         "password": "spark_password",
>         "driver": "org.postgresql.Driver"
>     }
>     
>     # Read from PostgreSQL
>     df = spark.read \
>         .jdbc(
>             url=jdbc_url,
>             table="sales_table",
>             properties=connection_properties
>         )
>     
>     return df
> 
> 
> def process_with_kafka_dependency(spark: SparkSession):
>     """
>     Example using Kafka dependency
>     Requires: spark-sql-kafka package
>     """
>     
>     df = spark.read \
>         .format("kafka") \
>         .option("kafka.bootstrap.servers", "localhost:9092") \
>         .option("subscribe", "sales-topic") \
>         .option("startingOffsets", "earliest") \
>         .load()
>     
>     return df
> 
> 
> def process_with_s3_dependency(spark: SparkSession, s3_path: str):
>     """
>     Example using AWS S3 dependencies
>     Requires: aws-java-sdk-bundle, hadoop-aws
>     """
>     
>     # S3 configuration (can also be set in spark-submit)
>     spark.sparkContext._jsc.hadoopConfiguration().set(
>         "fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", "")
>     )
>     spark.sparkContext._jsc.hadoopConfiguration().set(
>         "fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", "")
>     )
>     spark.sparkContext._jsc.hadoopConfiguration().set(
>         "fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
>     )
>     
>     # Read from S3
>     df = spark.read.parquet(s3_path)
>     
>     return df
> 
> 
> def main():
>     """Main execution with dependency management"""
>     
>     logger.info("Starting job with dependency management...")
>     
>     # Create Spark session with dependencies
>     spark = create_spark_with_dependencies("DependencyManagementJob")
>     
>     try:
>         # Example data processing
>         data = [
>             (1, "Product A!", "2024-01-01"),
>             (2, "Product B?", "2024-01-02"),
>             (3, "Product C#", "2024-01-03")
>         ]
>         df = spark.createDataFrame(data, ["id", "name", "date"])
>         
>         # Apply UDF that uses external libraries
>         df_processed = df.withColumn(
>             "cleaned_name",
>             process_with_external_lib(col("name"))
>         )
>         
>         df_processed.show()
>         
>         logger.info("Job completed successfully")
>         
>     except Exception as e:
>         logger.error(f"Job failed: {str(e)}")
>         raise
>     finally:
>         spark.stop()
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> ```python deployment/utils.py
> """
> Utility module to be distributed with Spark job
> This file would be passed using --py-files option
> """
> 
> from typing import Dict, Any
> import logging
> 
> logger = logging.getLogger(__name__)
> 
> 
> def validate_data(df) -> bool:
>     """Validate DataFrame structure and content"""
>     if df is None or df.count() == 0:
>         logger.warning("DataFrame is empty")
>         return False
>     return True
> 
> 
> def calculate_metrics(df) -> Dict[str, Any]:
>     """Calculate various metrics from DataFrame"""
>     return {
>         "total_records": df.count(),
>         "total_columns": len(df.columns),
>         "column_names": df.columns
>     }
> ```
> 
> ```python deployment/transformations.py
> """
> Transformation module to be distributed with Spark job
> This file would be passed using --py-files option
> """
> 
> import re
> import json
> 
> 
> def clean_text(text: str) -> str:
>     """Clean and normalize text"""
>     if not text:
>         return ""
>     return re.sub(r'[^\w\s]', '', text.lower().strip())
> 
> 
> def parse_json_column(json_str: str) -> dict:
>     """Parse JSON string to dictionary"""
>     try:
>         return json.loads(json_str)
>     except:
>         return {}
> ```
> 
> ```bash deployment/submit_with_dependencies.sh
> #!/bin/bash
> 
> # Comprehensive dependency management submission script
> 
> # Method 1: Submit with Python file dependencies
> echo "=== Submit with Python Dependencies ==="
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --name "Job with Python Dependencies" \
>   --py-files deployment/utils.py,deployment/transformations.py \
>   deployment/dependency_management_job.py
> 
> # Method 2: Submit with Python ZIP archive
> echo "=== Submit with Python ZIP Archive ==="
> # First create the archive
> cd deployment
> zip -r modules.zip utils.py transformations.py
> cd ..
> 
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --py-files deployment/modules.zip \
>   deployment/dependency_management_job.py
> 
> # Method 3: Submit with JAR dependencies
> echo "=== Submit with JAR Dependencies ==="
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --jars /path/to/postgresql-42.5.0.jar,/path/to/custom-lib.jar \
>   deployment/dependency_management_job.py
> 
> # Method 4: Submit with Maven packages
> echo "=== Submit with Maven Packages ==="
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,\
> com.amazonaws:aws-java-sdk-bundle:1.12.262,\
> org.apache.hadoop:hadoop-aws:3.3.4 \
>   deployment/dependency_management_job.py
> 
> # Method 5: Submit with requirements.txt (using virtual environment)
> echo "=== Submit with Python Requirements ==="
> # Create virtual environment archive
> pip install --target deployment/venv -r deployment/requirements.txt
> cd deployment
> zip -r venv.zip venv
> cd ..
> 
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --archives deployment/venv.zip#venv \
>   --conf spark.pyspark.python=./venv/bin/python \
>   deployment/dependency_management_job.py
> 
> # Method 6: Complete submission with all dependency types
> echo "=== Complete Submission with All Dependencies ==="
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --name "Complete Dependency Example" \
>   --driver-memory 4g \
>   --executor-memory 8g \
>   --executor-cores 4 \
>   --num-executors 10 \
>   --jars /path/to/postgresql-42.5.0.jar \
>   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
>   --py-files deployment/utils.py,deployment/transformations.py \
>   --files deployment/job_config.json \
>   --conf spark.sql.adaptive.enabled=true \
>   --conf spark.dynamicAllocation.enabled=true \
>   deployment/dependency_management_job.py \
>   --input-path hdfs://input/data \
>   --output-path hdfs://output/results
> ```
> 
> ```python deployment/environment_config_job.py
> """
> Environment Variable Configuration Pattern
> Demonstrates using environment variables for configuration
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, lit
> import os
> import logging
> from typing import Optional
> from dataclasses import dataclass
> 
> logging.basicConfig(level=logging.INFO)
> logger = logging.getLogger(__name__)
> 
> 
> @dataclass
> class EnvironmentConfig:
>     """Configuration loaded from environment variables"""
>     
>     # Application settings
>     app_name: str
>     environment: str  # dev, staging, prod
>     
>     # Spark settings
>     executor_memory: str
>     executor_cores: int
>     num_executors: int
>     driver_memory: str
>     
>     # Data paths
>     input_base_path: str
>     output_base_path: str
>     checkpoint_path: str
>     
>     # Database settings
>     db_host: str
>     db_port: int
>     db_name: str
>     db_user: str
>     db_password: str
>     
>     # AWS settings (optional)
>     aws_region: Optional[str] = None
>     aws_access_key: Optional[str] = None
>     aws_secret_key: Optional[str] = None
>     
>     # Processing settings
>     partition_count: int = 100
>     enable_adaptive_query: bool = True
>     enable_dynamic_allocation: bool = True
>     
>     @classmethod
>     def from_environment(cls):
>         """Create configuration from environment variables"""
>         
>         return cls(
>             # Application
>             app_name=os.getenv('APP_NAME', 'EnvironmentConfigJob'),
>             environment=os.getenv('ENVIRONMENT', 'dev'),
>             
>             # Spark resources
>             executor_memory=os.getenv('SPARK_EXECUTOR_MEMORY', '8g'),
>             executor_cores=int(os.getenv('SPARK_EXECUTOR_CORES', '4')),
>             num_executors=int(os.getenv('SPARK_NUM_EXECUTORS', '10')),
>             driver_memory=os.getenv('SPARK_DRIVER_MEMORY', '4g'),
>             
>             # Data paths
>             input_base_path=os.getenv('INPUT_BASE_PATH', '/data/input'),
>             output_base_path=os.getenv('OUTPUT_BASE_PATH', '/data/output'),
>             checkpoint_path=os.getenv('CHECKPOINT_PATH', '/data/checkpoints'),
>             
>             # Database
>             db_host=os.getenv('DB_HOST', 'localhost'),
>             db_port=int(os.getenv('DB_PORT', '5432')),
>             db_name=os.getenv('DB_NAME', 'analytics'),
>             db_user=os.getenv('DB_USER', 'spark_user'),
>             db_password=os.getenv('DB_PASSWORD', ''),
>             
>             # AWS (optional)
>             aws_region=os.getenv('AWS_REGION'),
>             aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
>             aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
>             
>             # Processing
>             partition_count=int(os.getenv('PARTITION_COUNT', '100')),
>             enable_adaptive_query=os.getenv('ENABLE_ADAPTIVE_QUERY', 'true').lower() == 'true',
>             enable_dynamic_allocation=os.getenv('ENABLE_DYNAMIC_ALLOCATION', 'true').lower() == 'true',
>         )
>     
>     def validate(self):
>         """Validate configuration"""
>         if not self.db_password and self.environment == 'prod':
>             raise ValueError("DB_PASSWORD must be set in production")
>         
>         if self.aws_access_key and not self.aws_secret_key:
>             raise ValueError("AWS_SECRET_ACCESS_KEY must be set if AWS_ACCESS_KEY_ID is provided")
>         
>         logger.info(f"Configuration validated for environment: {self.environment}")
> 
> 
> def create_spark_session_from_env(config: EnvironmentConfig) -> SparkSession:
>     """Create Spark session using environment-based configuration"""
>     
>     builder = SparkSession.builder \
>         .appName(f"{config.app_name}-{config.environment}")
>     
>     # Apply Spark configurations from environment
>     builder = builder.config('spark.executor.memory', config.executor_memory)
>     builder = builder.config('spark.executor.cores', str(config.executor_cores))
>     builder = builder.config('spark.driver.memory', config.driver_memory)
>     
>     # Adaptive query execution
>     if config.enable_adaptive_query:
>         builder = builder.config('spark.sql.adaptive.enabled', 'true')
>         builder = builder.config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
>     
>     # Dynamic allocation
>     if config.enable_dynamic_allocation:
>         builder = builder.config('spark.dynamicAllocation.enabled', 'true')
>         builder = builder.config('spark.dynamicAllocation.minExecutors', '5')
>         builder = builder.config('spark.dynamicAllocation.maxExecutors', str(config.num_executors))
>     
>     # AWS S3 configuration if credentials provided
>     if config.aws_access_key and config.aws_secret_key:
>         builder = builder.config('spark.hadoop.fs.s3a.access.key', config.aws_access_key)
>         builder = builder.config('spark.hadoop.fs.s3a.secret.key', config.aws_secret_key)
>         if config.aws_region:
>             builder = builder.config('spark.hadoop.fs.s3a.endpoint', 
>                                    f's3.{config.aws_region}.amazonaws.com')
>     
>     # Shuffle partitions
>     builder = builder.config('spark.sql.shuffle.partitions', str(config.partition_count))
>     
>     # Environment-specific configurations
>     if config.environment == 'prod':
>         builder = builder.config('spark.sql.autoBroadcastJoinThreshold', '100MB')
>         builder = builder.config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
>     elif config.environment == 'dev':
>         builder = builder.config('spark.ui.showConsoleProgress', 'true')
>     
>     return builder.getOrCreate()
> 
> 
> def get_jdbc_url(config: EnvironmentConfig) -> str:
>     """Construct JDBC URL from environment configuration"""
>     return f"jdbc:postgresql://{config.db_host}:{config.db_port}/{config.db_name}"
> 
> 
> def process_data_with_env_config(spark: SparkSession, config: EnvironmentConfig):
>     """Process data using environment-based configuration"""
>     
>     logger.info(f"Running in {config.environment} environment")
>     
>     # Construct input/output paths based on environment
>     input_path = f"{config.input_base_path}/sales_{config.environment}"
>     output_path = f"{config.output_base_path}/processed_{config.environment}"
>     
>     logger.info(f"Input path: {input_path}")
>     logger.info(f"Output path: {output_path}")
>     
>     # Read data
>     df = spark.read.parquet(input_path)
>     
>     # Add environment metadata
>     df = df.withColumn("environment", lit(config.environment)) \
>            .withColumn("processing_partition_count", lit(config.partition_count))
>     
>     # Repartition based on config
>     df = df.repartition(config.partition_count)
>     
>     # Write output
>     df.write \
>         .mode("overwrite") \
>         .option("compression", "snappy") \
>         .parquet(output_path)
>     
>     # Optionally write to database
>     if config.db_password:
>         logger.info("Writing results to database")
>         
>         jdbc_url = get_jdbc_url(config)
>         connection_properties = {
>             "user": config.db_user,
>             "password": config.db_password,
>             "driver": "org.postgresql.Driver"
>         }
>         
>         df.limit(1000).write \
>             .jdbc(
>                 url=jdbc_url,
>                 table=f"processed_sales_{config.environment}",
>                 mode="overwrite",
>                 properties=connection_properties
>             )
>     
>     logger.info("Data processing completed")
> 
> 
> def main():
>     """Main execution using environment configuration"""
>     
>     # Load configuration from environment
>     config = EnvironmentConfig.from_environment()
>     
>     # Validate configuration
>     config.validate()
>     
>     # Log configuration (mask sensitive data)
>     logger.info(f"App Name: {config.app_name}")
>     logger.info(f"Environment: {config.environment}")
>     logger.info(f"Executor Memory: {config.executor_memory}")
>     logger.info(f"Num Executors: {config.num_executors}")
>     logger.info(f"DB Host: {config.db_host}")
>     logger.info(f"DB Password: {'***' if config.db_password else 'Not set'}")
>     
>     # Create Spark session
>     spark = create_spark_session_from_env(config)
>     
>     try:
>         # Process data
>         process_data_with_env_config(spark, config)
>         
>         logger.info("Job completed successfully")
>         
>     except Exception as e:
>         logger.error(f"Job failed: {str(e)}")
>         raise
>     finally:
>         spark.stop()
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> ```bash deployment/env_config.sh
> #!/bin/bash
> 
> # Environment Configuration Script
> # Sets environment variables for different deployment environments
> 
> # Common settings
> export APP_NAME="SalesAnalyticsPipeline"
> export SPARK_EXECUTOR_CORES=4
> export PARTITION_COUNT=200
> 
> # Environment-specific configurations
> 
> if [ "$1" == "dev" ]; then
>     echo "=== Setting up DEVELOPMENT environment ==="
>     export ENVIRONMENT=dev
>     export SPARK_EXECUTOR_MEMORY=4g
>     export SPARK_DRIVER_MEMORY=2g
>     export SPARK_NUM_EXECUTORS=5
>     export INPUT_BASE_PATH=/data/dev/input
>     export OUTPUT_BASE_PATH=/data/dev/output
>     export CHECKPOINT_PATH=/data/dev/checkpoints
>     export DB_HOST=dev-db.example.com
>     export DB_PORT=5432
>     

#### _User_

> continue exactly where you left off

#### _Assistant_

> ```bash deployment/env_config.sh
>     export DB_NAME=analytics_dev
>     export DB_USER=dev_user
>     export DB_PASSWORD=dev_password
>     export ENABLE_ADAPTIVE_QUERY=true
>     export ENABLE_DYNAMIC_ALLOCATION=false
> 
> elif [ "$1" == "staging" ]; then
>     echo "=== Setting up STAGING environment ==="
>     export ENVIRONMENT=staging
>     export SPARK_EXECUTOR_MEMORY=8g
>     export SPARK_DRIVER_MEMORY=4g
>     export SPARK_NUM_EXECUTORS=10
>     export INPUT_BASE_PATH=/data/staging/input
>     export OUTPUT_BASE_PATH=/data/staging/output
>     export CHECKPOINT_PATH=/data/staging/checkpoints
>     export DB_HOST=staging-db.example.com
>     export DB_PORT=5432
>     export DB_NAME=analytics_staging
>     export DB_USER=staging_user
>     export DB_PASSWORD=staging_password
>     export ENABLE_ADAPTIVE_QUERY=true
>     export ENABLE_DYNAMIC_ALLOCATION=true
>     export AWS_REGION=us-east-1
> 
> elif [ "$1" == "prod" ]; then
>     echo "=== Setting up PRODUCTION environment ==="
>     export ENVIRONMENT=prod
>     export SPARK_EXECUTOR_MEMORY=16g
>     export SPARK_DRIVER_MEMORY=8g
>     export SPARK_NUM_EXECUTORS=20
>     export INPUT_BASE_PATH=s3a://prod-bucket/input
>     export OUTPUT_BASE_PATH=s3a://prod-bucket/output
>     export CHECKPOINT_PATH=s3a://prod-bucket/checkpoints
>     export DB_HOST=prod-db.example.com
>     export DB_PORT=5432
>     export DB_NAME=analytics_prod
>     export DB_USER=prod_user
>     # In production, password should come from secrets manager
>     export DB_PASSWORD=${PROD_DB_PASSWORD:-""}
>     export ENABLE_ADAPTIVE_QUERY=true
>     export ENABLE_DYNAMIC_ALLOCATION=true
>     export PARTITION_COUNT=500
>     # AWS credentials from environment or secrets manager
>     export AWS_REGION=us-east-1
>     export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-""}
>     export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-""}
> 
> else
>     echo "Usage: source env_config.sh [dev|staging|prod]"
>     exit 1
> fi
> 
> echo "Environment variables set for: $ENVIRONMENT"
> ```
> 
> ```bash deployment/submit_with_env.sh
> #!/bin/bash
> 
> # Submit Spark job with environment-based configuration
> 
> ENVIRONMENT=${1:-dev}
> 
> # Load environment variables
> source deployment/env_config.sh $ENVIRONMENT
> 
> echo "=== Submitting job for $ENVIRONMENT environment ==="
> 
> # Submit job with environment variables
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --name "$APP_NAME-$ENVIRONMENT" \
>   --driver-memory $SPARK_DRIVER_MEMORY \
>   --executor-memory $SPARK_EXECUTOR_MEMORY \
>   --executor-cores $SPARK_EXECUTOR_CORES \
>   --num-executors $SPARK_NUM_EXECUTORS \
>   --conf spark.sql.shuffle.partitions=$PARTITION_COUNT \
>   --conf spark.sql.adaptive.enabled=$ENABLE_ADAPTIVE_QUERY \
>   --conf spark.dynamicAllocation.enabled=$ENABLE_DYNAMIC_ALLOCATION \
>   --conf spark.yarn.appMasterEnv.ENVIRONMENT=$ENVIRONMENT \
>   --conf spark.yarn.appMasterEnv.INPUT_BASE_PATH=$INPUT_BASE_PATH \
>   --conf spark.yarn.appMasterEnv.OUTPUT_BASE_PATH=$OUTPUT_BASE_PATH \
>   --conf spark.yarn.appMasterEnv.DB_HOST=$DB_HOST \
>   --conf spark.yarn.appMasterEnv.DB_PORT=$DB_PORT \
>   --conf spark.yarn.appMasterEnv.DB_NAME=$DB_NAME \
>   --conf spark.yarn.appMasterEnv.DB_USER=$DB_USER \
>   --conf spark.yarn.appMasterEnv.DB_PASSWORD=$DB_PASSWORD \
>   --conf spark.executorEnv.ENVIRONMENT=$ENVIRONMENT \
>   --conf spark.executorEnv.INPUT_BASE_PATH=$INPUT_BASE_PATH \
>   --conf spark.executorEnv.OUTPUT_BASE_PATH=$OUTPUT_BASE_PATH \
>   deployment/environment_config_job.py
> 
> echo "Job submitted successfully for $ENVIRONMENT"
> ```
> 
> ```dockerfile deployment/Dockerfile
> # Dockerfile for containerized Spark job deployment
> # Demonstrates dependency management in containerized environments
> 
> FROM apache/spark:3.5.0-python3
> 
> # Set working directory
> WORKDIR /app
> 
> # Install system dependencies
> USER root
> RUN apt-get update && apt-get install -y \
>     gcc \
>     g++ \
>     python3-dev \
>     && rm -rf /var/lib/apt/lists/*
> 
> # Copy requirements file
> COPY deployment/requirements.txt .
> 
> # Install Python dependencies
> RUN pip install --no-cache-dir -r requirements.txt
> 
> # Copy application files
> COPY deployment/*.py ./
> COPY deployment/utils.py ./
> COPY deployment/transformations.py ./
> COPY deployment/job_config.json ./
> 
> # Copy JAR dependencies
> COPY deployment/jars/*.jar /opt/spark/jars/
> 
> # Set environment variables
> ENV SPARK_HOME=/opt/spark
> ENV PYTHONPATH=/app:$PYTHONPATH
> ENV PYSPARK_PYTHON=python3
> 
> # Create user for running Spark jobs
> RUN useradd -m -u 1000 spark && \
>     chown -R spark:spark /app
> 
> USER spark
> 
> # Default command
> CMD ["python", "spark_submit_job.py"]
> ```
> 
> ```txt deployment/requirements.txt
> # Python dependencies for PySpark jobs
> 
> # Data processing
> pandas==2.1.0
> numpy==1.24.3
> pyarrow==13.0.0
> 
> # Database connectivity
> psycopg2-binary==2.9.7
> pymongo==4.5.0
> redis==5.0.0
> 
> # AWS/Cloud
> boto3==1.28.0
> s3fs==2023.9.0
> 
> # Data validation
> pydantic==2.4.0
> cerberus==1.3.5
> 
> # Utilities
> python-dotenv==1.0.0
> PyYAML==6.0.1
> requests==2.31.0
> 
> # Testing (for local development)
> pytest==7.4.0
> pytest-spark==0.6.0
> ```
> 
> ```yaml deployment/kubernetes-job.yaml
> # Kubernetes Job definition for Spark application
> # Demonstrates Spark on Kubernetes deployment pattern
> 
> apiVersion: batch/v1
> kind: Job
> metadata:
>   name: spark-job-deployment-example
>   namespace: spark-jobs
> spec:
>   template:
>     metadata:
>       labels:
>         app: spark-job
>         version: v1
>     spec:
>       serviceAccountName: spark
>       restartPolicy: Never
>       containers:
>       - name: spark-submit
>         image: apache/spark:3.5.0
>         command:
>         - /opt/spark/bin/spark-submit
>         args:
>         - --master
>         - k8s://https://kubernetes.default.svc:443
>         - --deploy-mode
>         - cluster
>         - --name
>         - spark-k8s-deployment-job
>         - --conf
>         - spark.executor.instances=5
>         - --conf
>         - spark.executor.memory=8g
>         - --conf
>         - spark.executor.cores=4
>         - --conf
>         - spark.driver.memory=4g
>         - --conf
>         - spark.kubernetes.container.image=spark-job-image:latest
>         - --conf
>         - spark.kubernetes.namespace=spark-jobs
>         - --conf
>         - spark.kubernetes.authenticate.driver.serviceAccountName=spark
>         - --conf
>         - spark.kubernetes.driver.pod.name=spark-driver-pod
>         - --conf
>         - spark.kubernetes.executor.request.cores=3
>         - --conf
>         - spark.kubernetes.executor.limit.cores=4
>         - --py-files
>         - local:///app/utils.py,local:///app/transformations.py
>         - local:///app/environment_config_job.py
>         env:
>         - name: ENVIRONMENT
>           value: "prod"
>         - name: INPUT_BASE_PATH
>           value: "s3a://prod-bucket/input"
>         - name: OUTPUT_BASE_PATH
>           value: "s3a://prod-bucket/output"
>         - name: SPARK_EXECUTOR_MEMORY
>           value: "8g"
>         - name: SPARK_EXECUTOR_CORES
>           value: "4"
>         - name: DB_HOST
>           valueFrom:
>             configMapKeyRef:
>               name: spark-config
>               key: db_host
>         - name: DB_USER
>           valueFrom:
>             secretKeyRef:
>               name: spark-secrets
>               key: db_user
>         - name: DB_PASSWORD
>           valueFrom:
>             secretKeyRef:
>               name: spark-secrets
>               key: db_password
>         - name: AWS_ACCESS_KEY_ID
>           valueFrom:
>             secretKeyRef:
>               name: aws-credentials
>               key: access_key_id
>         - name: AWS_SECRET_ACCESS_KEY
>           valueFrom:
>             secretKeyRef:
>               name: aws-credentials
>               key: secret_access_key
>         resources:
>           requests:
>             memory: "2Gi"
>             cpu: "1"
>           limits:
>             memory: "4Gi"
>             cpu: "2"
> ---
> apiVersion: v1
> kind: ConfigMap
> metadata:
>   name: spark-config
>   namespace: spark-jobs
> data:
>   db_host: "prod-db.example.com"
>   db_port: "5432"
>   db_name: "analytics_prod"
>   partition_count: "500"
> ---
> apiVersion: v1
> kind: Secret
> metadata:
>   name: spark-secrets
>   namespace: spark-jobs
> type: Opaque
> stringData:
>   db_user: "prod_user"
>   db_password: "secure_password_here"
> ---
> apiVersion: v1
> kind: Secret
> metadata:
>   name: aws-credentials
>   namespace: spark-jobs
> type: Opaque
> stringData:
>   access_key_id: "your_access_key"
>   secret_access_key: "your_secret_key"
> ```
> 
> ```python deployment/advanced_deployment_patterns.py
> """
> Advanced Deployment Patterns
> Demonstrates sophisticated deployment techniques including:
> - Job chaining and orchestration
> - Failure handling and retries
> - Dynamic resource allocation
> - Multi-stage pipelines
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, current_timestamp, lit
> import sys
> import time
> import logging
> from typing import List, Dict, Any, Callable
> from enum import Enum
> from dataclasses import dataclass
> import traceback
> 
> logging.basicConfig(
>     level=logging.INFO,
>     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
> )
> logger = logging.getLogger(__name__)
> 
> 
> class JobStatus(Enum):
>     """Job execution status"""
>     PENDING = "PENDING"
>     RUNNING = "RUNNING"
>     SUCCESS = "SUCCESS"
>     FAILED = "FAILED"
>     RETRYING = "RETRYING"
> 
> 
> @dataclass
> class JobStage:
>     """Represents a stage in a multi-stage pipeline"""
>     name: str
>     function: Callable
>     max_retries: int = 3
>     retry_delay: int = 60  # seconds
>     depends_on: List[str] = None
>     
>     def __post_init__(self):
>         if self.depends_on is None:
>             self.depends_on = []
> 
> 
> class PipelineOrchestrator:
>     """
>     Orchestrates multi-stage Spark job pipeline with:
>     - Dependency management
>     - Retry logic
>     - Resource optimization
>     - Failure recovery
>     """
>     
>     def __init__(self, spark: SparkSession, pipeline_name: str):
>         self.spark = spark
>         self.pipeline_name = pipeline_name
>         self.stages: Dict[str, JobStage] = {}
>         self.stage_results: Dict[str, Any] = {}
>         self.stage_status: Dict[str, JobStatus] = {}
>         
>     def add_stage(self, stage: JobStage):
>         """Add a stage to the pipeline"""
>         self.stages[stage.name] = stage
>         self.stage_status[stage.name] = JobStatus.PENDING
>         logger.info(f"Added stage: {stage.name}")
>     
>     def can_execute_stage(self, stage_name: str) -> bool:
>         """Check if a stage's dependencies are satisfied"""
>         stage = self.stages[stage_name]
>         
>         for dependency in stage.depends_on:
>             if dependency not in self.stage_status:
>                 logger.error(f"Dependency {dependency} not found for stage {stage_name}")
>                 return False
>             if self.stage_status[dependency] != JobStatus.SUCCESS:
>                 return False
>         
>         return True
>     
>     def execute_stage_with_retry(self, stage: JobStage) -> bool:
>         """Execute a stage with retry logic"""
>         attempt = 0
>         
>         while attempt < stage.max_retries:
>             try:
>                 logger.info(f"Executing stage '{stage.name}' (attempt {attempt + 1}/{stage.max_retries})")
>                 self.stage_status[stage.name] = JobStatus.RUNNING
>                 
>                 # Execute the stage function
>                 result = stage.function(self.spark, self.stage_results)
>                 
>                 # Store result
>                 self.stage_results[stage.name] = result
>                 self.stage_status[stage.name] = JobStatus.SUCCESS
>                 
>                 logger.info(f"Stage '{stage.name}' completed successfully")
>                 return True
>                 
>             except Exception as e:
>                 attempt += 1
>                 logger.error(f"Stage '{stage.name}' failed (attempt {attempt}): {str(e)}")
>                 logger.error(traceback.format_exc())
>                 
>                 if attempt < stage.max_retries:
>                     self.stage_status[stage.name] = JobStatus.RETRYING
>                     logger.info(f"Retrying in {stage.retry_delay} seconds...")
>                     time.sleep(stage.retry_delay)
>                 else:
>                     self.stage_status[stage.name] = JobStatus.FAILED
>                     logger.error(f"Stage '{stage.name}' failed after {stage.max_retries} attempts")
>                     return False
>         
>         return False
>     
>     def execute_pipeline(self) -> bool:
>         """Execute the entire pipeline in dependency order"""
>         logger.info(f"Starting pipeline: {self.pipeline_name}")
>         
>         executed_stages = set()
>         total_stages = len(self.stages)
>         
>         while len(executed_stages) < total_stages:
>             progress_made = False
>             
>             for stage_name, stage in self.stages.items():
>                 if stage_name in executed_stages:
>                     continue
>                 
>                 if self.stage_status[stage_name] == JobStatus.FAILED:
>                     continue
>                 
>                 if self.can_execute_stage(stage_name):
>                     success = self.execute_stage_with_retry(stage)
>                     executed_stages.add(stage_name)
>                     progress_made = True
>                     
>                     if not success:
>                         logger.error(f"Pipeline failed at stage: {stage_name}")
>                         return False
>             
>             if not progress_made:
>                 logger.error("Pipeline deadlock: No stages can be executed")
>                 return False
>         
>         logger.info(f"Pipeline '{self.pipeline_name}' completed successfully")
>         return True
>     
>     def get_execution_summary(self) -> Dict[str, Any]:
>         """Get summary of pipeline execution"""
>         return {
>             "pipeline_name": self.pipeline_name,
>             "total_stages": len(self.stages),
>             "successful_stages": sum(1 for s in self.stage_status.values() if s == JobStatus.SUCCESS),
>             "failed_stages": sum(1 for s in self.stage_status.values() if s == JobStatus.FAILED),
>             "stage_status": {name: status.value for name, status in self.stage_status.items()}
>         }
> 
> 
> # Example pipeline stage functions
> 
> def stage_extract_data(spark: SparkSession, previous_results: Dict) -> Any:
>     """Stage 1: Extract data from source"""
>     logger.info("Extracting data from source...")
>     
>     # Simulate data extraction
>     data = [
>         (1, "Product A", 100.0, "2024-01-01"),
>         (2, "Product B", 200.0, "2024-01-02"),
>         (3, "Product C", 150.0, "2024-01-03")
>     ]
>     
>     df = spark.createDataFrame(data, ["id", "product", "amount", "date"])
>     
>     # Cache for downstream stages
>     df.cache()
>     
>     logger.info(f"Extracted {df.count()} records")
>     return df
> 
> 
> def stage_transform_data(spark: SparkSession, previous_results: Dict) -> Any:
>     """Stage 2: Transform extracted data"""
>     logger.info("Transforming data...")
>     
>     # Get data from previous stage
>     input_df = previous_results.get("extract_data")
>     
>     if input_df is None:
>         raise ValueError("No input data from extract stage")
>     
>     # Apply transformations
>     transformed_df = input_df \
>         .withColumn("amount_usd", col("amount") * 1.0) \
>         .withColumn("processed_at", current_timestamp()) \
>         .withColumn("data_quality", lit("VALIDATED"))
>     
>     logger.info("Transformation completed")
>     return transformed_df
> 
> 
> def stage_aggregate_data(spark: SparkSession, previous_results: Dict) -> Any:
>     """Stage 3: Aggregate transformed data"""
>     logger.info("Aggregating data...")
>     
>     # Get data from previous stage
>     input_df = previous_results.get("transform_data")
>     
>     if input_df is None:
>         raise ValueError("No input data from transform stage")
>     
>     # Perform aggregations
>     aggregated_df = input_df.groupBy("product") \
>         .agg({"amount": "sum", "id": "count"}) \
>         .withColumnRenamed("sum(amount)", "total_amount") \
>         .withColumnRenamed("count(id)", "transaction_count")
>     
>     logger.info("Aggregation completed")
>     return aggregated_df
> 
> 
> def stage_load_data(spark: SparkSession, previous_results: Dict) -> Any:
>     """Stage 4: Load final results"""
>     logger.info("Loading data to destination...")
>     
>     # Get data from previous stage
>     final_df = previous_results.get("aggregate_data")
>     
>     if final_df is None:
>         raise ValueError("No input data from aggregate stage")
>     
>     # Write to destination (simulated)
>     output_path = "/tmp/spark_output/final_results"
>     final_df.write.mode("overwrite").parquet(output_path)
>     
>     logger.info(f"Data loaded to {output_path}")
>     return {"output_path": output_path, "record_count": final_df.count()}
> 
> 
> def main():
>     """Main execution with advanced deployment patterns"""
>     
>     # Create Spark session with optimized configuration
>     spark = SparkSession.builder \
>         .appName("AdvancedDeploymentPipeline") \
>         .config("spark.sql.adaptive.enabled", "true") \
>         .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
>         .config("spark.dynamicAllocation.enabled", "true") \
>         .getOrCreate()
>     
>     try:
>         # Create pipeline orchestrator
>         orchestrator = PipelineOrchestrator(spark, "ETL_Pipeline")
>         
>         # Define pipeline stages
>         stage1 = JobStage(
>             name="extract_data",
>             function=stage_extract_data,
>             max_retries=3
>         )
>         
>         stage2 = JobStage(
>             name="transform_data",
>             function=stage_transform_data,
>             depends_on=["extract_data"],
>             max_retries=2
>         )
>         
>         stage3 = JobStage(
>             name="aggregate_data",
>             function=stage_aggregate_data,
>             depends_on=["transform_data"],
>             max_retries=2
>         )
>         
>         stage4 = JobStage(
>             name="load_data",
>             function=stage_load_data,
>             depends_on=["aggregate_data"],
>             max_retries=3
>         )
>         
>         # Add stages to orchestrator
>         orchestrator.add_stage(stage1)
>         orchestrator.add_stage(stage2)
>         orchestrator.add_stage(stage3)
>         orchestrator.add_stage(stage4)
>         
>         # Execute pipeline
>         success = orchestrator.execute_pipeline()
>         
>         # Print summary
>         summary = orchestrator.get_execution_summary()
>         logger.info(f"Execution Summary: {summary}")
>         
>         if not success:
>             sys.exit(1)
>             
>     except Exception as e:
>         logger.error(f"Pipeline execution failed: {str(e)}")
>         logger.error(traceback.format_exc())
>         sys.exit(1)
>     finally:
>         spark.stop()
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> ```bash deployment/submit_advanced_pipeline.sh
> #!/bin/bash
> 
> # Submit advanced multi-stage pipeline with comprehensive configuration
> 
> spark-submit \
>   --master yarn \
>   --deploy-mode cluster \
>   --name "Advanced-ETL-Pipeline" \
>   --driver-memory 4g \
>   --executor-memory 8g \
>   --executor-cores 4 \
>   --num-executors 10 \
>   --conf spark.sql.adaptive.enabled=true \
>   --conf spark.sql.adaptive.coalescePartitions.enabled=true \
>   --conf spark.sql.adaptive.skewJoin.enabled=true \
>   --conf spark.dynamicAllocation.enabled=true \
>   --conf spark.dynamicAllocation.minExecutors=5 \
>   --conf spark.dynamicAllocation.maxExecutors=20 \
>   --conf spark.dynamicAllocation.initialExecutors=10 \
>   --conf spark.speculation=true \
>   --conf spark.task.maxFailures=4 \
>   --conf spark.stage.maxConsecutiveAttempts=4 \
>   --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
>   --conf spark.sql.shuffle.partitions=200 \
>   --conf spark.default.parallelism=200 \
>   --conf spark.network.timeout=600s \
>   --conf spark.executor.heartbeatInterval=60s \
>   --files deployment/job_config.json \
>   --py-files deployment/utils.py,deployment/transformations.py \
>   deployment/advanced_deployment_patterns.py
> 
> echo "Advanced pipeline submitted successfully"
> ```
> 
> These comprehensive examples demonstrate all four deployment and orchestration patterns with production-ready code including error handling, logging, retry logic, and multiple deployment options (local, YARN, Kubernetes).