### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:56:00 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Testing**
> *   Unit Testing Spark Code (e.g., using `spark-testing-base` or creating local `SparkSession` for tests)
> *   Mocking External Dependencies for Tests
> ```

#### _Assistant_

> I'll create comprehensive PySpark examples for both unit testing patterns.
> 
> ## 1. Unit Testing Spark Code
> 
> ```python test_spark_unit_tests.py
> import pytest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> from pyspark.sql.functions import col, when, avg
> from chispa.dataframe_comparer import assert_df_equality
> from decimal import Decimal
> 
> 
> # Fixture to create a local SparkSession for testing
> @pytest.fixture(scope="session")
> def spark():
>     """Create a SparkSession for testing."""
>     spark = SparkSession.builder \
>         .appName("unit-tests") \
>         .master("local[2]") \
>         .config("spark.sql.shuffle.partitions", "2") \
>         .getOrCreate()
>     
>     yield spark
>     
>     spark.stop()
> 
> 
> # Sample business logic functions to test
> def transform_customer_data(df):
>     """
>     Transform customer data:
>     - Categorize customers by age group
>     - Calculate discount based on purchase amount
>     """
>     return df.withColumn(
>         "age_group",
>         when(col("age") < 18, "minor")
>         .when((col("age") >= 18) & (col("age") < 65), "adult")
>         .otherwise("senior")
>     ).withColumn(
>         "discount",
>         when(col("purchase_amount") >= 1000, 0.20)
>         .when(col("purchase_amount") >= 500, 0.10)
>         .otherwise(0.05)
>     )
> 
> 
> def calculate_average_by_category(df, group_col, agg_col):
>     """Calculate average of a column grouped by another column."""
>     return df.groupBy(group_col).agg(
>         avg(agg_col).alias(f"avg_{agg_col}")
>     )
> 
> 
> # Test cases using local SparkSession
> class TestCustomerTransformation:
>     """Test suite for customer data transformation."""
>     
>     def test_age_group_categorization(self, spark):
>         """Test age group categorization logic."""
>         # Arrange - Create test data
>         input_data = [
>             ("Alice", 15, 100),
>             ("Bob", 25, 200),
>             ("Charlie", 70, 300),
>         ]
>         input_schema = StructType([
>             StructField("name", StringType(), True),
>             StructField("age", IntegerType(), True),
>             StructField("purchase_amount", IntegerType(), True),
>         ])
>         
>         input_df = spark.createDataFrame(input_data, input_schema)
>         
>         # Act - Apply transformation
>         result_df = transform_customer_data(input_df)
>         
>         # Assert - Verify results
>         expected_data = [
>             ("Alice", 15, 100, "minor", 0.05),
>             ("Bob", 25, 200, "adult", 0.05),
>             ("Charlie", 70, 300, "senior", 0.05),
>         ]
>         expected_schema = StructType([
>             StructField("name", StringType(), True),
>             StructField("age", IntegerType(), True),
>             StructField("purchase_amount", IntegerType(), True),
>             StructField("age_group", StringType(), True),
>             StructField("discount", DoubleType(), True),
>         ])
>         
>         expected_df = spark.createDataFrame(expected_data, expected_schema)
>         
>         assert_df_equality(result_df, expected_df, ignore_nullable=True)
>     
>     def test_discount_calculation(self, spark):
>         """Test discount calculation based on purchase amount."""
>         # Arrange
>         input_data = [
>             ("Customer1", 30, 400),   # 5% discount
>             ("Customer2", 30, 600),   # 10% discount
>             ("Customer3", 30, 1200),  # 20% discount
>         ]
>         input_schema = StructType([
>             StructField("name", StringType(), True),
>             StructField("age", IntegerType(), True),
>             StructField("purchase_amount", IntegerType(), True),
>         ])
>         
>         input_df = spark.createDataFrame(input_data, input_schema)
>         
>         # Act
>         result_df = transform_customer_data(input_df)
>         
>         # Assert
>         discounts = [row.discount for row in result_df.select("discount").collect()]
>         assert discounts == [0.05, 0.10, 0.20]
>     
>     def test_empty_dataframe(self, spark):
>         """Test handling of empty DataFrames."""
>         # Arrange
>         schema = StructType([
>             StructField("name", StringType(), True),
>             StructField("age", IntegerType(), True),
>             StructField("purchase_amount", IntegerType(), True),
>         ])
>         
>         empty_df = spark.createDataFrame([], schema)
>         
>         # Act
>         result_df = transform_customer_data(empty_df)
>         
>         # Assert
>         assert result_df.count() == 0
>         assert "age_group" in result_df.columns
>         assert "discount" in result_df.columns
> 
> 
> class TestAggregations:
>     """Test suite for aggregation functions."""
>     
>     def test_average_calculation(self, spark):
>         """Test average calculation by category."""
>         # Arrange
>         input_data = [
>             ("Electronics", 100),
>             ("Electronics", 200),
>             ("Clothing", 50),
>             ("Clothing", 150),
>         ]
>         input_schema = StructType([
>             StructField("category", StringType(), True),
>             StructField("price", IntegerType(), True),
>         ])
>         
>         input_df = spark.createDataFrame(input_data, input_schema)
>         
>         # Act
>         result_df = calculate_average_by_category(input_df, "category", "price")
>         
>         # Assert
>         result_dict = {row.category: row.avg_price for row in result_df.collect()}
>         assert result_dict["Electronics"] == 150.0
>         assert result_dict["Clothing"] == 100.0
>     
>     def test_single_group(self, spark):
>         """Test aggregation with single group."""
>         # Arrange
>         input_data = [
>             ("A", 10),
>             ("A", 20),
>             ("A", 30),
>         ]
>         input_schema = StructType([
>             StructField("group", StringType(), True),
>             StructField("value", IntegerType(), True),
>         ])
>         
>         input_df = spark.createDataFrame(input_data, input_schema)
>         
>         # Act
>         result_df = calculate_average_by_category(input_df, "group", "value")
>         
>         # Assert
>         assert result_df.count() == 1
>         assert result_df.first().avg_value == 20.0
> 
> 
> # Alternative: Using spark-testing-base (if installed)
> # pip install spark-testing-base
> try:
>     from sparktestingbase.sqltestcase import SQLTestCase
>     
>     class TestWithSparkTestingBase(SQLTestCase):
>         """Test using spark-testing-base library."""
>         
>         def test_data_transformation(self):
>             """Test transformation using spark-testing-base."""
>             # Create input DataFrame
>             input_data = [
>                 ("Alice", 25, 500),
>                 ("Bob", 65, 1000),
>             ]
>             input_df = self.spark.createDataFrame(
>                 input_data, 
>                 ["name", "age", "purchase_amount"]
>             )
>             
>             # Apply transformation
>             result_df = transform_customer_data(input_df)
>             
>             # Expected output
>             expected_data = [
>                 ("Alice", 25, 500, "adult", 0.10),
>                 ("Bob", 65, 1000, "senior", 0.20),
>             ]
>             expected_df = self.spark.createDataFrame(
>                 expected_data,
>                 ["name", "age", "purchase_amount", "age_group", "discount"]
>             )
>             
>             # Assert DataFrames are equal
>             self.assertDataFrameEqual(result_df, expected_df)
> 
> except ImportError:
>     print("spark-testing-base not installed. Skipping those tests.")
> 
> 
> if __name__ == "__main__":
>     pytest.main([__file__, "-v"])
> ```
> 
> ## 2. Mocking External Dependencies for Tests
> 
> ```python test_mocking_dependencies.py
> import pytest
> from unittest.mock import Mock, patch, MagicMock
> from pyspark.sql import SparkSession, DataFrame
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> from pyspark.sql.functions import col
> import requests
> 
> 
> # Fixture for SparkSession
> @pytest.fixture(scope="session")
> def spark():
>     """Create a SparkSession for testing."""
>     spark = SparkSession.builder \
>         .appName("mock-tests") \
>         .master("local[2]") \
>         .getOrCreate()
>     
>     yield spark
>     spark.stop()
> 
> 
> # ============================================
> # Example 1: Mocking External API Calls
> # ============================================
> 
> class ExternalAPIService:
>     """Service that calls external REST API."""
>     
>     def __init__(self, base_url):
>         self.base_url = base_url
>     
>     def get_enrichment_data(self, customer_id):
>         """Fetch customer enrichment data from external API."""
>         response = requests.get(f"{self.base_url}/customers/{customer_id}")
>         response.raise_for_status()
>         return response.json()
>     
>     def enrich_dataframe(self, df, spark):
>         """Enrich DataFrame with external API data."""
>         enriched_rows = []
>         
>         for row in df.collect():
>             customer_id = row.customer_id
>             try:
>                 api_data = self.get_enrichment_data(customer_id)
>                 enriched_rows.append({
>                     "customer_id": customer_id,
>                     "name": row.name,
>                     "credit_score": api_data.get("credit_score"),
>                     "risk_level": api_data.get("risk_level")
>                 })
>             except Exception as e:
>                 enriched_rows.append({
>                     "customer_id": customer_id,
>                     "name": row.name,
>                     "credit_score": None,
>                     "risk_level": "unknown"
>                 })
>         
>         schema = StructType([
>             StructField("customer_id", StringType(), True),
>             StructField("name", StringType(), True),
>             StructField("credit_score", IntegerType(), True),
>             StructField("risk_level", StringType(), True),
>         ])
>         
>         return spark.createDataFrame(enriched_rows, schema)
> 
> 
> class TestExternalAPIServiceMocking:
>     """Test suite with mocked external API calls."""
>     
>     @patch('requests.get')
>     def test_enrich_dataframe_success(self, mock_get, spark):
>         """Test enrichment with mocked successful API response."""
>         # Arrange - Mock API response
>         mock_response = Mock()
>         mock_response.status_code = 200
>         mock_response.json.return_value = {
>             "credit_score": 750,
>             "risk_level": "low"
>         }
>         mock_get.return_value = mock_response
>         
>         # Create input DataFrame
>         input_data = [("C001", "Alice")]
>         input_df = spark.createDataFrame(input_data, ["customer_id", "name"])
>         
>         # Create service and enrich
>         service = ExternalAPIService("https://api.example.com")
>         result_df = service.enrich_dataframe(input_df, spark)
>         
>         # Assert
>         result = result_df.collect()[0]
>         assert result.customer_id == "C001"
>         assert result.credit_score == 750
>         assert result.risk_level == "low"
>         
>         # Verify API was called correctly
>         mock_get.assert_called_once_with("https://api.example.com/customers/C001")
>     
>     @patch('requests.get')
>     def test_enrich_dataframe_api_failure(self, mock_get, spark):
>         """Test enrichment when API call fails."""
>         # Arrange - Mock API failure
>         mock_get.side_effect = requests.exceptions.RequestException("API Error")
>         
>         input_data = [("C002", "Bob")]
>         input_df = spark.createDataFrame(input_data, ["customer_id", "name"])
>         
>         # Act
>         service = ExternalAPIService("https://api.example.com")
>         result_df = service.enrich_dataframe(input_df, spark)
>         
>         # Assert - Should handle error gracefully
>         result = result_df.collect()[0]
>         assert result.customer_id == "C002"
>         assert result.credit_score is None
>         assert result.risk_level == "unknown"
> 
> 
> # ============================================
> # Example 2: Mocking Database Connections
> # ============================================
> 
> class DatabaseService:
>     """Service that reads from and writes to database."""
>     
>     def __init__(self, jdbc_url, properties):
>         self.jdbc_url = jdbc_url
>         self.properties = properties
>     
>     def read_from_database(self, spark, table_name):
>         """Read data from database table."""
>         return spark.read.jdbc(
>             url=self.jdbc_url,
>             table=table_name,
>             properties=self.properties
>         )
>     
>     def write_to_database(self, df, table_name, mode="append"):
>         """Write DataFrame to database table."""
>         df.write.jdbc(
>             url=self.jdbc_url,
>             table=table_name,
>             mode=mode,
>             properties=self.properties
>         )
> 
> 
> class TestDatabaseServiceMocking:
>     """Test suite with mocked database operations."""
>     
>     def test_read_from_database_mock(self, spark):
>         """Test reading from database with mocked connection."""
>         # Create mock DataFrame that would come from database
>         mock_data = [
>             ("P001", "Product A", 100),
>             ("P002", "Product B", 200),
>         ]
>         mock_df = spark.createDataFrame(
>             mock_data,
>             ["product_id", "product_name", "price"]
>         )
>         
>         # Mock the spark.read.jdbc method
>         with patch.object(spark.read, 'jdbc', return_value=mock_df) as mock_jdbc:
>             # Create service
>             db_service = DatabaseService(
>                 jdbc_url="jdbc:postgresql://localhost:5432/testdb",
>                 properties={"user": "test", "password": "test"}
>             )
>             
>             # Act
>             result_df = db_service.read_from_database(spark, "products")
>             
>             # Assert
>             assert result_df.count() == 2
>             assert "product_id" in result_df.columns
>             
>             # Verify jdbc was called with correct parameters
>             mock_jdbc.assert_called_once()
>     
>     def test_write_to_database_mock(self, spark):
>         """Test writing to database with mocked connection."""
>         # Arrange - Create test DataFrame
>         test_data = [("P003", "Product C", 300)]
>         test_df = spark.createDataFrame(
>             test_data,
>             ["product_id", "product_name", "price"]
>         )
>         
>         # Mock the write.jdbc method
>         mock_writer = MagicMock()
>         
>         with patch.object(test_df.write, 'jdbc', mock_writer):
>             # Create service
>             db_service = DatabaseService(
>                 jdbc_url="jdbc:postgresql://localhost:5432/testdb",
>                 properties={"user": "test", "password": "test"}
>             )
>             
>             # Act
>             db_service.write_to_database(test_df, "products", mode="overwrite")
>             
>             # Assert - Verify write was called
>             mock_writer.assert_called_once()
> 
> 
> # ============================================
> # Example 3: Mocking File System Operations
> # ============================================
> 
> class FileProcessor:
>     """Process files from various sources."""
>     
>     def read_csv_file(self, spark, file_path):
>         """Read CSV file into DataFrame."""
>         return spark.read.csv(file_path, header=True, inferSchema=True)
>     
>     def read_parquet_file(self, spark, file_path):
>         """Read Parquet file into DataFrame."""
>         return spark.read.parquet(file_path)
>     
>     def write_output(self, df, output_path, format_type="parquet"):
>         """Write DataFrame to file."""
>         if format_type == "parquet":
>             df.write.parquet(output_path, mode="overwrite")
>         elif format_type == "csv":
>             df.write.csv(output_path, header=True, mode="overwrite")
> 
> 
> class TestFileProcessorMocking:
>     """Test suite with mocked file operations."""
>     
>     def test_read_csv_file_mock(self, spark):
>         """Test CSV reading with mocked file system."""
>         # Create mock DataFrame that would come from CSV
>         mock_data = [("1", "Alice", "30"), ("2", "Bob", "25")]
>         mock_df = spark.createDataFrame(mock_data, ["id", "name", "age"])
>         
>         # Mock the spark.read.csv method
>         with patch.object(spark.read, 'csv', return_value=mock_df) as mock_csv:
>             processor = FileProcessor()
>             result_df = processor.read_csv_file(spark, "/fake/path/data.csv")
>             
>             # Assert
>             assert result_df.count() == 2
>             mock_csv.assert_called_once_with(
>                 "/fake/path/data.csv",
>                 header=True,
>                 inferSchema=True
>             )
>     
>     def test_write_output_mock(self, spark):
>         """Test file writing with mocked file system."""
>         # Create test DataFrame
>         test_data = [("1", "Test")]
>         test_df = spark.createDataFrame(test_data, ["id", "value"])
>         
>         # Mock the write operations
>         mock_parquet_writer = MagicMock()
>         
>         with patch.object(test_df.write, 'parquet', mock_parquet_writer):
>             processor = FileProcessor()
>             processor.write_output(test_df, "/fake/output/path", "parquet")
>             
>             # Assert
>             mock_parquet_writer.assert_called_once_with(
>                 "/fake/output/path",
>                 mode="overwrite"
>             )
> 
> 
> # ============================================
> # Example 4: Mocking Spark Configuration
> # ============================================
> 
> class ConfigurableProcessor:
>     """Processor that uses Spark configuration."""
>     
>     def process_with_config(self, spark, df):
>         """Process DataFrame using Spark configuration."""
>         # Get configuration value
>         max_records = int(spark.conf.get("spark.custom.maxRecords", "1000"))
>         
>         # Limit DataFrame based on configuration
>         return df.limit(max_records)
> 
> 
> class TestConfigMocking:
>     """Test suite with mocked Spark configuration."""
>     
>     def test_process_with_custom_config(self, spark):
>         """Test processing with mocked configuration."""
>         # Create test DataFrame
>         test_data = [(i, f"value_{i}") for i in range(100)]
>         test_df = spark.createDataFrame(test_data, ["id", "value"])
>         
>         # Mock the configuration
>         with patch.object(spark.conf, 'get', return_value="50"):
>             processor = ConfigurableProcessor()
>             result_df = processor.process_with_config(spark, test_df)
>             
>             # Assert
>             assert result_df.count() == 50
>     
>     def test_process_with_default_config(self, spark):
>         """Test processing with default configuration."""
>         # Create large test DataFrame
>         test_data = [(i, f"value_{i}") for i in range(2000)]
>         test_df = spark.createDataFrame(test_data, ["id", "value"])
>         
>         processor = ConfigurableProcessor()
>         result_df = processor.process_with_config(spark, test_df)
>         
>         # Should use default limit of 1000
>         assert result_df.count() == 1000
> 
> 
> if __name__ == "__main__":
>     pytest.main([__file__, "-v"])
> ```
> 
> ## Requirements File
> 
> ```text requirements.txt
> pyspark>=3.0.0
> pytest>=7.0.0
> chispa>=0.9.0
> requests>=2.28.0
> 
> # Optional: for spark-testing-base
> # spark-testing-base==0.14.0
> ```
> 
> ## Usage Instructions
> 
> ```bash usage.sh
> # Install dependencies
> pip install -r requirements.txt
> 
> # Run all tests
> pytest test_spark_unit_tests.py test_mocking_dependencies.py -v
> 
> # Run specific test class
> pytest test_spark_unit_tests.py::TestCustomerTransformation -v
> 
> # Run tests with coverage
> pytest --cov=. --cov-report=html
> 
> # Run only mocking tests
> pytest test_mocking_dependencies.py -v
> 
> # Run tests in parallel (requires pytest-xdist)
> pytest -n auto
> ```
> 
> These examples demonstrate:
> 
> 1. **Unit Testing Spark Code**: Creating local SparkSession, testing transformations, aggregations, and edge cases using pytest and chispa
> 2. **Mocking External Dependencies**: Mocking API calls, database connections, file systems, and Spark configurations to isolate unit tests
> 
> Both patterns are essential for maintaining reliable PySpark applications with comprehensive test coverage!