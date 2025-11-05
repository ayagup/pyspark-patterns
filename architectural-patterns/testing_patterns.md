### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 8:09:13 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> Testing Patterns
> - Unit Testing Pattern
> - Integration Testing Pattern
> - Data Validation Testing Pattern
> - Schema Testing Pattern
> - Mock Data Pattern
> - Fixture Pattern
> - Test Data Generation Pattern
> - Property-based Testing Pattern
> - Performance Testing Pattern
> - Regression Testing Pattern
> ```

#### _Assistant_

> I'll create comprehensive PySpark testing pattern examples for each pattern. Let me generate these one by one:
> 
> ## 1. Unit Testing Pattern
> 
> ```python test_patterns/unit_testing_pattern.py
> """
> Unit Testing Pattern - Tests individual PySpark functions in isolation
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType
> from pyspark.sql.functions import col, upper, length
> 
> 
> class DataTransformations:
>     """Example transformations to test"""
>     
>     @staticmethod
>     def clean_name(df):
>         """Convert names to uppercase and filter out empty names"""
>         return df.filter(col("name").isNotNull()) \
>                  .withColumn("name", upper(col("name")))
>     
>     @staticmethod
>     def add_name_length(df):
>         """Add column with name length"""
>         return df.withColumn("name_length", length(col("name")))
> 
> 
> class TestDataTransformations(unittest.TestCase):
>     """Unit tests for data transformations"""
>     
>     @classmethod
>     def setUpClass(cls):
>         """Create Spark session once for all tests"""
>         cls.spark = SparkSession.builder \
>             .appName("UnitTestingPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         """Stop Spark session after all tests"""
>         cls.spark.stop()
>     
>     def test_clean_name_converts_to_uppercase(self):
>         """Test that clean_name converts names to uppercase"""
>         # Arrange
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("name", StringType(), True)
>         ])
>         data = [(1, "john"), (2, "jane")]
>         df = self.spark.createDataFrame(data, schema)
>         
>         # Act
>         result_df = DataTransformations.clean_name(df)
>         
>         # Assert
>         names = [row.name for row in result_df.collect()]
>         self.assertEqual(names, ["JOHN", "JANE"])
>     
>     def test_clean_name_filters_nulls(self):
>         """Test that clean_name removes null names"""
>         # Arrange
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("name", StringType(), True)
>         ])
>         data = [(1, "john"), (2, None), (3, "jane")]
>         df = self.spark.createDataFrame(data, schema)
>         
>         # Act
>         result_df = DataTransformations.clean_name(df)
>         
>         # Assert
>         self.assertEqual(result_df.count(), 2)
>     
>     def test_add_name_length(self):
>         """Test that add_name_length calculates correct length"""
>         # Arrange
>         schema = StructType([
>             StructField("name", StringType(), True)
>         ])
>         data = [("John",), ("Alexander",)]
>         df = self.spark.createDataFrame(data, schema)
>         
>         # Act
>         result_df = DataTransformations.add_name_length(df)
>         
>         # Assert
>         lengths = [row.name_length for row in result_df.collect()]
>         self.assertEqual(lengths, [4, 9])
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 2. Integration Testing Pattern
> 
> ```python test_patterns/integration_testing_pattern.py
> """
> Integration Testing Pattern - Tests multiple components working together
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> from pyspark.sql.functions import col, avg, count
> import tempfile
> import shutil
> 
> 
> class DataPipeline:
>     """Example data pipeline with multiple stages"""
>     
>     def __init__(self, spark):
>         self.spark = spark
>     
>     def extract(self, path):
>         """Extract data from CSV"""
>         return self.spark.read.csv(path, header=True, inferSchema=True)
>     
>     def transform(self, df):
>         """Transform data - clean and aggregate"""
>         cleaned_df = df.filter(col("amount").isNotNull()) \
>                        .filter(col("amount") > 0)
>         
>         aggregated_df = cleaned_df.groupBy("category") \
>             .agg(
>                 avg("amount").alias("avg_amount"),
>                 count("*").alias("transaction_count")
>             )
>         
>         return aggregated_df
>     
>     def load(self, df, path):
>         """Load data to parquet"""
>         df.write.mode("overwrite").parquet(path)
> 
> 
> class TestDataPipeline(unittest.TestCase):
>     """Integration tests for complete data pipeline"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("IntegrationTestingPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>         
>         # Create temporary directories
>         cls.temp_dir = tempfile.mkdtemp()
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>         shutil.rmtree(cls.temp_dir)
>     
>     def setUp(self):
>         """Create test data for each test"""
>         self.input_path = f"{self.temp_dir}/input.csv"
>         self.output_path = f"{self.temp_dir}/output"
>         
>         # Create test input data
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("category", StringType(), True),
>             StructField("amount", DoubleType(), True)
>         ])
>         data = [
>             (1, "food", 50.0),
>             (2, "food", 75.0),
>             (3, "transport", 30.0),
>             (4, "transport", None),
>             (5, "entertainment", -10.0)
>         ]
>         df = self.spark.createDataFrame(data, schema)
>         df.coalesce(1).write.mode("overwrite").csv(self.input_path, header=True)
>     
>     def test_complete_pipeline_flow(self):
>         """Test the entire ETL pipeline integration"""
>         # Arrange
>         pipeline = DataPipeline(self.spark)
>         
>         # Act - Run complete pipeline
>         extracted_df = pipeline.extract(self.input_path)
>         transformed_df = pipeline.transform(extracted_df)
>         pipeline.load(transformed_df, self.output_path)
>         
>         # Assert - Read output and verify
>         result_df = self.spark.read.parquet(self.output_path)
>         
>         self.assertEqual(result_df.count(), 2)  # Only food and transport
>         
>         food_row = result_df.filter(col("category") == "food").first()
>         self.assertEqual(food_row.transaction_count, 2)
>         self.assertAlmostEqual(food_row.avg_amount, 62.5, places=1)
>     
>     def test_pipeline_handles_corrupted_data(self):
>         """Test pipeline filters out invalid records"""
>         # Arrange
>         pipeline = DataPipeline(self.spark)
>         
>         # Act
>         extracted_df = pipeline.extract(self.input_path)
>         original_count = extracted_df.count()
>         transformed_df = pipeline.transform(extracted_df)
>         final_count = transformed_df.agg({"transaction_count": "sum"}).first()[0]
>         
>         # Assert - Verify filtering occurred
>         self.assertEqual(original_count, 5)
>         self.assertEqual(final_count, 3)  # Removed null and negative values
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 3. Data Validation Testing Pattern
> 
> ```python test_patterns/data_validation_testing_pattern.py
> """
> Data Validation Testing Pattern - Validates data quality and constraints
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, DoubleType
> from pyspark.sql.functions import col, count, when, isnan, isnull
> from datetime import date
> 
> 
> class DataValidator:
>     """Data validation rules"""
>     
>     @staticmethod
>     def validate_not_null(df, column_name):
>         """Check for null values in a column"""
>         null_count = df.filter(col(column_name).isNull()).count()
>         return null_count == 0, null_count
>     
>     @staticmethod
>     def validate_range(df, column_name, min_val, max_val):
>         """Validate values are within range"""
>         out_of_range = df.filter(
>             (col(column_name) < min_val) | (col(column_name) > max_val)
>         ).count()
>         return out_of_range == 0, out_of_range
>     
>     @staticmethod
>     def validate_unique(df, column_name):
>         """Check if column values are unique"""
>         total_count = df.count()
>         unique_count = df.select(column_name).distinct().count()
>         return total_count == unique_count, total_count - unique_count
>     
>     @staticmethod
>     def validate_format(df, column_name, pattern):
>         """Validate string format using regex"""
>         invalid_count = df.filter(~col(column_name).rlike(pattern)).count()
>         return invalid_count == 0, invalid_count
>     
>     @staticmethod
>     def validate_referential_integrity(df1, df2, key_column):
>         """Check referential integrity between two dataframes"""
>         orphan_count = df1.join(
>             df2, 
>             df1[key_column] == df2[key_column], 
>             "left_anti"
>         ).count()
>         return orphan_count == 0, orphan_count
> 
> 
> class TestDataValidation(unittest.TestCase):
>     """Tests for data validation"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("DataValidationTestingPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>     
>     def test_validate_not_null_passes(self):
>         """Test not null validation with valid data"""
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("name", StringType(), True)
>         ])
>         data = [(1, "John"), (2, "Jane")]
>         df = self.spark.createDataFrame(data, schema)
>         
>         is_valid, null_count = DataValidator.validate_not_null(df, "name")
>         
>         self.assertTrue(is_valid)
>         self.assertEqual(null_count, 0)
>     
>     def test_validate_not_null_fails(self):
>         """Test not null validation with null values"""
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("name", StringType(), True)
>         ])
>         data = [(1, "John"), (2, None), (3, "Jane")]
>         df = self.spark.createDataFrame(data, schema)
>         
>         is_valid, null_count = DataValidator.validate_not_null(df, "name")
>         
>         self.assertFalse(is_valid)
>         self.assertEqual(null_count, 1)
>     
>     def test_validate_range(self):
>         """Test range validation"""
>         schema = StructType([
>             StructField("age", IntegerType(), True)
>         ])
>         data = [(25,), (30,), (150,), (-5,)]
>         df = self.spark.createDataFrame(data, schema)
>         
>         is_valid, out_of_range_count = DataValidator.validate_range(df, "age", 0, 120)
>         
>         self.assertFalse(is_valid)
>         self.assertEqual(out_of_range_count, 2)
>     
>     def test_validate_unique(self):
>         """Test uniqueness validation"""
>         schema = StructType([
>             StructField("email", StringType(), True)
>         ])
>         data = [("a@test.com",), ("b@test.com",), ("a@test.com",)]
>         df = self.spark.createDataFrame(data, schema)
>         
>         is_valid, duplicate_count = DataValidator.validate_unique(df, "email")
>         
>         self.assertFalse(is_valid)
>         self.assertEqual(duplicate_count, 1)
>     
>     def test_validate_format(self):
>         """Test format validation with regex"""
>         schema = StructType([
>             StructField("email", StringType(), True)
>         ])
>         data = [
>             ("valid@email.com",),
>             ("also.valid@test.co.uk",),
>             ("invalid.email",),
>             ("@missing.user",)
>         ]
>         df = self.spark.createDataFrame(data, schema)
>         
>         email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
>         is_valid, invalid_count = DataValidator.validate_format(df, "email", email_pattern)
>         
>         self.assertFalse(is_valid)
>         self.assertEqual(invalid_count, 2)
>     
>     def test_validate_referential_integrity(self):
>         """Test referential integrity between tables"""
>         # Parent table
>         parent_schema = StructType([
>             StructField("dept_id", IntegerType(), True),
>             StructField("dept_name", StringType(), True)
>         ])
>         parent_data = [(1, "Sales"), (2, "IT")]
>         parent_df = self.spark.createDataFrame(parent_data, parent_schema)
>         
>         # Child table with orphan record
>         child_schema = StructType([
>             StructField("emp_id", IntegerType(), True),
>             StructField("dept_id", IntegerType(), True)
>         ])
>         child_data = [(101, 1), (102, 2), (103, 99)]  # 99 doesn't exist
>         child_df = self.spark.createDataFrame(child_data, child_schema)
>         
>         is_valid, orphan_count = DataValidator.validate_referential_integrity(
>             child_df, parent_df, "dept_id"
>         )
>         
>         self.assertFalse(is_valid)
>         self.assertEqual(orphan_count, 1)
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 4. Schema Testing Pattern
> 
> ```python test_patterns/schema_testing_pattern.py
> """
> Schema Testing Pattern - Validates DataFrame schemas
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import (
>     StructType, StructField, StringType, IntegerType, 
>     DoubleType, DateType, BooleanType, ArrayType
> )
> from datetime import date
> 
> 
> class SchemaValidator:
>     """Schema validation utilities"""
>     
>     @staticmethod
>     def validate_schema_match(df, expected_schema):
>         """Check if DataFrame schema matches expected schema exactly"""
>         return df.schema == expected_schema
>     
>     @staticmethod
>     def validate_column_exists(df, column_name):
>         """Check if column exists in DataFrame"""
>         return column_name in df.columns
>     
>     @staticmethod
>     def validate_column_type(df, column_name, expected_type):
>         """Check if column has expected data type"""
>         for field in df.schema.fields:
>             if field.name == column_name:
>                 return isinstance(field.dataType, type(expected_type))
>         return False
>     
>     @staticmethod
>     def validate_nullable(df, column_name, expected_nullable):
>         """Check if column nullable property matches expected"""
>         for field in df.schema.fields:
>             if field.name == column_name:
>                 return field.nullable == expected_nullable
>         return False
>     
>     @staticmethod
>     def get_schema_diff(actual_schema, expected_schema):
>         """Get differences between schemas"""
>         actual_fields = {f.name: f for f in actual_schema.fields}
>         expected_fields = {f.name: f for f in expected_schema.fields}
>         
>         missing = set(expected_fields.keys()) - set(actual_fields.keys())
>         extra = set(actual_fields.keys()) - set(expected_fields.keys())
>         type_mismatches = []
>         
>         for name in set(actual_fields.keys()) & set(expected_fields.keys()):
>             if actual_fields[name].dataType != expected_fields[name].dataType:
>                 type_mismatches.append({
>                     'column': name,
>                     'actual': str(actual_fields[name].dataType),
>                     'expected': str(expected_fields[name].dataType)
>                 })
>         
>         return {
>             'missing_columns': list(missing),
>             'extra_columns': list(extra),
>             'type_mismatches': type_mismatches
>         }
> 
> 
> class TestSchemaValidation(unittest.TestCase):
>     """Tests for schema validation"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("SchemaTestingPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>     
>     def test_exact_schema_match(self):
>         """Test that schema matches expected schema exactly"""
>         expected_schema = StructType([
>             StructField("id", IntegerType(), False),
>             StructField("name", StringType(), True),
>             StructField("salary", DoubleType(), True)
>         ])
>         
>         data = [(1, "John", 50000.0), (2, "Jane", 60000.0)]
>         df = self.spark.createDataFrame(data, expected_schema)
>         
>         self.assertTrue(SchemaValidator.validate_schema_match(df, expected_schema))
>     
>     def test_schema_mismatch_detected(self):
>         """Test that schema mismatch is detected"""
>         actual_schema = StructType([
>             StructField("id", IntegerType(), False),
>             StructField("name", StringType(), True)
>         ])
>         
>         expected_schema = StructType([
>             StructField("id", StringType(), False),  # Different type
>             StructField("name", StringType(), True)
>         ])
>         
>         data = [(1, "John"), (2, "Jane")]
>         df = self.spark.createDataFrame(data, actual_schema)
>         
>         self.assertFalse(SchemaValidator.validate_schema_match(df, expected_schema))
>     
>     def test_column_exists(self):
>         """Test column existence validation"""
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("name", StringType(), True)
>         ])
>         data = [(1, "John")]
>         df = self.spark.createDataFrame(data, schema)
>         
>         self.assertTrue(SchemaValidator.validate_column_exists(df, "name"))
>         self.assertFalse(SchemaValidator.validate_column_exists(df, "age"))
>     
>     def test_column_type_validation(self):
>         """Test column type validation"""
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("amount", DoubleType(), True),
>             StructField("active", BooleanType(), True)
>         ])
>         data = [(1, 100.5, True)]
>         df = self.spark.createDataFrame(data, schema)
>         
>         self.assertTrue(SchemaValidator.validate_column_type(df, "id", IntegerType()))
>         self.assertTrue(SchemaValidator.validate_column_type(df, "amount", DoubleType()))
>         self.assertFalse(SchemaValidator.validate_column_type(df, "id", StringType()))
>     
>     def test_nullable_validation(self):
>         """Test nullable property validation"""
>         schema = StructType([
>             StructField("id", IntegerType(), False),
>             StructField("name", StringType(), True)
>         ])
>         data = [(1, "John")]
>         df = self.spark.createDataFrame(data, schema)
>         
>         self.assertTrue(SchemaValidator.validate_nullable(df, "id", False))
>         self.assertTrue(SchemaValidator.validate_nullable(df, "name", True))
>         self.assertFalse(SchemaValidator.validate_nullable(df, "id", True))
>     
>     def test_complex_schema_with_nested_types(self):
>         """Test schema validation with complex nested types"""
>         expected_schema = StructType([
>             StructField("id", IntegerType(), False),
>             StructField("tags", ArrayType(StringType()), True),
>             StructField("metadata", StructType([
>                 StructField("created_date", DateType(), True),
>                 StructField("version", IntegerType(), True)
>             ]), True)
>         ])
>         
>         data = [(1, ["tag1", "tag2"], (date(2024, 1, 1), 1))]
>         df = self.spark.createDataFrame(data, expected_schema)
>         
>         self.assertTrue(SchemaValidator.validate_schema_match(df, expected_schema))
>     
>     def test_schema_diff_report(self):
>         """Test schema difference reporting"""
>         actual_schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("name", StringType(), True),
>             StructField("extra_col", StringType(), True)
>         ])
>         
>         expected_schema = StructType([
>             StructField("id", StringType(), True),  # Wrong type
>             StructField("name", StringType(), True),
>             StructField("email", StringType(), True)  # Missing in actual
>         ])
>         
>         data = [(1, "John", "extra")]
>         df = self.spark.createDataFrame(data, actual_schema)
>         
>         diff = SchemaValidator.get_schema_diff(df.schema, expected_schema)
>         
>         self.assertIn("email", diff['missing_columns'])
>         self.assertIn("extra_col", diff['extra_columns'])
>         self.assertEqual(len(diff['type_mismatches']), 1)
>         self.assertEqual(diff['type_mismatches'][0]['column'], 'id')
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 5. Mock Data Pattern
> 
> ```python test_patterns/mock_data_pattern.py
> """
> Mock Data Pattern - Creating and using mock data for testing
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
> from pyspark.sql.functions import col
> from unittest.mock import Mock, patch, MagicMock
> 
> 
> class ExternalDataSource:
>     """Simulates an external data source (database, API, etc.)"""
>     
>     def __init__(self, spark, connection_string):
>         self.spark = spark
>         self.connection_string = connection_string
>     
>     def fetch_customers(self):
>         """Fetch customer data from external source"""
>         # In real scenario, this would connect to a database
>         return self.spark.read.jdbc(
>             url=self.connection_string,
>             table="customers",
>             properties={"driver": "org.postgresql.Driver"}
>         )
> 
> 
> class DataProcessor:
>     """Processes data from external sources"""
>     
>     def __init__(self, data_source):
>         self.data_source = data_source
>     
>     def get_premium_customers(self):
>         """Get customers with total purchases > 1000"""
>         customers_df = self.data_source.fetch_customers()
>         return customers_df.filter(col("total_purchases") > 1000)
> 
> 
> class MockDataFactory:
>     """Factory for creating mock DataFrames"""
>     
>     @staticmethod
>     def create_customer_mock(spark, customer_data=None):
>         """Create mock customer DataFrame"""
>         schema = StructType([
>             StructField("customer_id", IntegerType(), False),
>             StructField("name", StringType(), True),
>             StructField("total_purchases", DoubleType(), True),
>             StructField("status", StringType(), True)
>         ])
>         
>         if customer_data is None:
>             customer_data = [
>                 (1, "John Doe", 1500.0, "active"),
>                 (2, "Jane Smith", 500.0, "active"),
>                 (3, "Bob Wilson", 2000.0, "premium"),
>                 (4, "Alice Brown", 800.0, "active")
>             ]
>         
>         return spark.createDataFrame(customer_data, schema)
>     
>     @staticmethod
>     def create_transaction_mock(spark, transaction_data=None):
>         """Create mock transaction DataFrame"""
>         schema = StructType([
>             StructField("transaction_id", IntegerType(), False),
>             StructField("customer_id", IntegerType(), True),
>             StructField("amount", DoubleType(), True),
>             StructField("product", StringType(), True)
>         ])
>         
>         if transaction_data is None:
>             transaction_data = [
>                 (1, 1, 100.0, "Widget"),
>                 (2, 1, 200.0, "Gadget"),
>                 (3, 2, 50.0, "Widget"),
>                 (4, 3, 500.0, "Premium Item")
>             ]
>         
>         return spark.createDataFrame(transaction_data, schema)
> 
> 
> class TestMockDataPattern(unittest.TestCase):
>     """Tests using mock data"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("MockDataPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>     
>     def test_using_mock_dataframe(self):
>         """Test using mock DataFrame instead of real data source"""
>         # Arrange - Create mock data
>         mock_customers = MockDataFactory.create_customer_mock(self.spark)
>         
>         # Create a mock data source
>         mock_source = Mock(spec=ExternalDataSource)
>         mock_source.fetch_customers.return_value = mock_customers
>         
>         # Act
>         processor = DataProcessor(mock_source)
>         premium_customers = processor.get_premium_customers()
>         
>         # Assert
>         self.assertEqual(premium_customers.count(), 2)
>         mock_source.fetch_customers.assert_called_once()
>     
>     def test_custom_mock_data(self):
>         """Test with custom mock data for edge cases"""
>         # Arrange - Create custom mock data with edge cases
>         custom_data = [
>             (1, "Edge Case 1", 1000.0, "active"),  # Exactly 1000
>             (2, "Edge Case 2", 1000.01, "active"),  # Just over 1000
>             (3, "Edge Case 3", 999.99, "active")   # Just under 1000
>         ]
>         mock_customers = MockDataFactory.create_customer_mock(self.spark, custom_data)
>         
>         mock_source = Mock(spec=ExternalDataSource)
>         mock_source.fetch_customers.return_value = mock_customers
>         
>         # Act
>         processor = DataProcessor(mock_source)
>         premium_customers = processor.get_premium_customers()
>         
>         # Assert - Only customer with 1000.01 should be included
>         self.assertEqual(premium_customers.count(), 1)
>         result = premium_customers.first()
>         self.assertEqual(result.customer_id, 2)
>     
>     def test_mock_empty_dataframe(self):
>         """Test behavior with empty mock data"""
>         # Arrange
>         empty_mock = MockDataFactory.create_customer_mock(self.spark, [])
>         
>         mock_source = Mock(spec=ExternalDataSource)
>         mock_source.fetch_customers.return_value = empty_mock
>         
>         # Act
>         processor = DataProcessor(mock_source)
>         premium_customers = processor.get_premium_customers()
>         
>         # Assert
>         self.assertEqual(premium_customers.count(), 0)
>     
>     @patch('test_patterns.mock_data_pattern.ExternalDataSource')
>     def test_with_patch_decorator(self, MockDataSource):
>         """Test using patch decorator for mocking"""
>         # Arrange
>         mock_instance = MockDataSource.return_value
>         mock_df = MockDataFactory.create_customer_mock(self.spark)
>         mock_instance.fetch_customers.return_value = mock_df
>         
>         # Act
>         data_source = ExternalDataSource(self.spark, "mock://connection")
>         processor = DataProcessor(data_source)
>         result = processor.get_premium_customers()
>         
>         # Assert
>         self.assertEqual(result.count(), 2)
>     
>     def test_mock_with_side_effect(self):
>         """Test mock with side effects for complex scenarios"""
>         # Arrange
>         call_count = [0]
>         
>         def side_effect():
>             call_count[0] += 1
>             if call_count[0] == 1:
>                 # First call returns some data
>                 return MockDataFactory.create_customer_mock(self.spark, [
>                     (1, "First Call", 1500.0, "active")
>                 ])
>             else:
>                 # Subsequent calls return different data
>                 return MockDataFactory.create_customer_mock(self.spark, [
>                     (2, "Second Call", 2000.0, "premium")
>                 ])
>         
>         mock_source = Mock(spec=ExternalDataSource)
>         mock_source.fetch_customers.side_effect = side_effect
>         
>         # Act
>         processor = DataProcessor(mock_source)
>         result1 = processor.get_premium_customers()
>         result2 = processor.get_premium_customers()
>         
>         # Assert
>         self.assertEqual(result1.first().name, "First Call")
>         self.assertEqual(result2.first().name, "Second Call")
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 6. Fixture Pattern
> 
> ```python test_patterns/fixture_pattern.py
> """
> Fixture Pattern - Reusable test data setup and teardown
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
> from datetime import date, timedelta
> import tempfile
> import shutil
> 
> 
> class TestFixtures:
>     """Reusable test fixtures"""
>     
>     @staticmethod
>     def create_sample_customers(spark):
>         """Fixture: Create sample customer DataFrame"""
>         schema = StructType([
>             StructField("customer_id", IntegerType(), False),
>             StructField("name", StringType(), True),
>             StructField("email", StringType(), True),
>             StructField("signup_date", DateType(), True),
>             StructField("status", StringType(), True)
>         ])
>         
>         data = [
>             (1, "John Doe", "john@example.com", date(2023, 1, 15), "active"),
>             (2, "Jane Smith", "jane@example.com", date(2023, 2, 20), "active"),
>             (3, "Bob Wilson", "bob@example.com", date(2022, 12, 10), "inactive"),
>             (4, "Alice Brown", "alice@example.com", date(2023, 3, 5), "active"),
>             (5, "Charlie Davis", "charlie@example.com", date(2023, 1, 25), "suspended")
>         ]
>         
>         return spark.createData

#### _User_

> Continue your response exactly where you left off:

#### _Assistant_

> ```python test_patterns/fixture_pattern.py
> """
> Fixture Pattern - Reusable test data setup and teardown
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
> from datetime import date, timedelta
> import tempfile
> import shutil
> 
> 
> class TestFixtures:
>     """Reusable test fixtures"""
>     
>     @staticmethod
>     def create_sample_customers(spark):
>         """Fixture: Create sample customer DataFrame"""
>         schema = StructType([
>             StructField("customer_id", IntegerType(), False),
>             StructField("name", StringType(), True),
>             StructField("email", StringType(), True),
>             StructField("signup_date", DateType(), True),
>             StructField("status", StringType(), True)
>         ])
>         
>         data = [
>             (1, "John Doe", "john@example.com", date(2023, 1, 15), "active"),
>             (2, "Jane Smith", "jane@example.com", date(2023, 2, 20), "active"),
>             (3, "Bob Wilson", "bob@example.com", date(2022, 12, 10), "inactive"),
>             (4, "Alice Brown", "alice@example.com", date(2023, 3, 5), "active"),
>             (5, "Charlie Davis", "charlie@example.com", date(2023, 1, 25), "suspended")
>         ]
>         
>         return spark.createDataFrame(data, schema)
>     
>     @staticmethod
>     def create_sample_orders(spark):
>         """Fixture: Create sample orders DataFrame"""
>         schema = StructType([
>             StructField("order_id", IntegerType(), False),
>             StructField("customer_id", IntegerType(), True),
>             StructField("order_date", DateType(), True),
>             StructField("amount", DoubleType(), True),
>             StructField("product", StringType(), True)
>         ])
>         
>         data = [
>             (101, 1, date(2023, 1, 20), 150.0, "Widget A"),
>             (102, 1, date(2023, 2, 15), 200.0, "Widget B"),
>             (103, 2, date(2023, 2, 25), 75.0, "Gadget X"),
>             (104, 3, date(2023, 1, 10), 300.0, "Widget A"),
>             (105, 4, date(2023, 3, 10), 120.0, "Gadget Y"),
>             (106, 1, date(2023, 3, 15), 180.0, "Widget C")
>         ]
>         
>         return spark.createDataFrame(data, schema)
>     
>     @staticmethod
>     def create_large_dataset(spark, num_records=1000):
>         """Fixture: Create large dataset for performance testing"""
>         from pyspark.sql.functions import monotonically_increasing_id, rand
>         
>         df = spark.range(num_records).toDF("id")
>         df = df.withColumn("value", (rand() * 1000).cast("double"))
>         df = df.withColumn("category", (rand() * 10).cast("int").cast("string"))
>         
>         return df
> 
> 
> class TestWithFixtures(unittest.TestCase):
>     """Base test class with common fixtures"""
>     
>     @classmethod
>     def setUpClass(cls):
>         """Set up Spark session once for all tests"""
>         cls.spark = SparkSession.builder \
>             .appName("FixturePattern") \
>             .master("local[2]") \
>             .config("spark.sql.shuffle.partitions", "2") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>         
>         # Create temp directory for file-based tests
>         cls.temp_dir = tempfile.mkdtemp()
>     
>     @classmethod
>     def tearDownClass(cls):
>         """Clean up after all tests"""
>         cls.spark.stop()
>         shutil.rmtree(cls.temp_dir)
>     
>     def setUp(self):
>         """Set up fixtures before each test"""
>         self.customers_df = TestFixtures.create_sample_customers(self.spark)
>         self.orders_df = TestFixtures.create_sample_orders(self.spark)
>     
>     def tearDown(self):
>         """Clean up after each test"""
>         # Optional: Clean up any test-specific resources
>         pass
> 
> 
> class TestCustomerAnalysis(TestWithFixtures):
>     """Tests using customer fixtures"""
>     
>     def test_count_active_customers(self):
>         """Test counting active customers"""
>         from pyspark.sql.functions import col
>         
>         active_count = self.customers_df.filter(col("status") == "active").count()
>         
>         self.assertEqual(active_count, 3)
>     
>     def test_filter_by_signup_date(self):
>         """Test filtering customers by signup date"""
>         from pyspark.sql.functions import col
>         
>         recent_customers = self.customers_df.filter(
>             col("signup_date") >= date(2023, 1, 1)
>         )
>         
>         self.assertEqual(recent_customers.count(), 4)
>     
>     def test_customer_order_join(self):
>         """Test joining customers with orders"""
>         from pyspark.sql.functions import col
>         
>         result = self.customers_df.join(
>             self.orders_df,
>             self.customers_df.customer_id == self.orders_df.customer_id,
>             "inner"
>         )
>         
>         self.assertEqual(result.count(), 6)
>         
>         # Verify John Doe has 3 orders
>         john_orders = result.filter(col("name") == "John Doe").count()
>         self.assertEqual(john_orders, 3)
> 
> 
> class TestOrderAnalysis(TestWithFixtures):
>     """Tests using order fixtures"""
>     
>     def test_total_revenue(self):
>         """Test calculating total revenue"""
>         from pyspark.sql.functions import sum as spark_sum
>         
>         total = self.orders_df.agg(spark_sum("amount").alias("total")).first().total
>         
>         self.assertAlmostEqual(total, 1025.0, places=2)
>     
>     def test_average_order_value(self):
>         """Test calculating average order value"""
>         from pyspark.sql.functions import avg
>         
>         average = self.orders_df.agg(avg("amount").alias("average")).first().average
>         
>         self.assertAlmostEqual(average, 170.83, places=2)
>     
>     def test_orders_per_customer(self):
>         """Test counting orders per customer"""
>         from pyspark.sql.functions import count, col
>         
>         orders_per_customer = self.orders_df.groupBy("customer_id") \
>             .agg(count("*").alias("order_count"))
>         
>         customer_1_orders = orders_per_customer.filter(
>             col("customer_id") == 1
>         ).first().order_count
>         
>         self.assertEqual(customer_1_orders, 3)
> 
> 
> class TestWithFileFixtures(TestWithFixtures):
>     """Tests using file-based fixtures"""
>     
>     def setUp(self):
>         """Set up file-based fixtures"""
>         super().setUp()
>         
>         # Write customers to CSV
>         self.customers_path = f"{self.temp_dir}/customers.csv"
>         self.customers_df.write.mode("overwrite").csv(
>             self.customers_path, 
>             header=True
>         )
>         
>         # Write orders to Parquet
>         self.orders_path = f"{self.temp_dir}/orders.parquet"
>         self.orders_df.write.mode("overwrite").parquet(self.orders_path)
>     
>     def test_read_csv_fixture(self):
>         """Test reading CSV fixture"""
>         df = self.spark.read.csv(self.customers_path, header=True, inferSchema=True)
>         
>         self.assertEqual(df.count(), 5)
>         self.assertIn("customer_id", df.columns)
>     
>     def test_read_parquet_fixture(self):
>         """Test reading Parquet fixture"""
>         df = self.spark.read.parquet(self.orders_path)
>         
>         self.assertEqual(df.count(), 6)
>         self.assertIn("order_id", df.columns)
> 
> 
> class TestWithLargeDataFixture(TestWithFixtures):
>     """Tests using large dataset fixtures"""
>     
>     def setUp(self):
>         """Set up large dataset fixture"""
>         super().setUp()
>         self.large_df = TestFixtures.create_large_dataset(self.spark, 10000)
>     
>     def test_large_dataset_count(self):
>         """Test operations on large dataset"""
>         self.assertEqual(self.large_df.count(), 10000)
>     
>     def test_large_dataset_aggregation(self):
>         """Test aggregation on large dataset"""
>         from pyspark.sql.functions import avg, max, min
>         
>         stats = self.large_df.agg(
>             avg("value").alias("avg_value"),
>             max("value").alias("max_value"),
>             min("value").alias("min_value")
>         ).first()
>         
>         # Verify stats are in expected ranges
>         self.assertGreater(stats.avg_value, 0)
>         self.assertLess(stats.avg_value, 1000)
>         self.assertGreaterEqual(stats.max_value, stats.avg_value)
>         self.assertLessEqual(stats.min_value, stats.avg_value)
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 7. Test Data Generation Pattern
> 
> ```python test_patterns/test_data_generation_pattern.py
> """
> Test Data Generation Pattern - Programmatically generate test data
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, BooleanType
> from pyspark.sql.functions import rand, randn, monotonically_increasing_id, expr, lit, concat, col
> from datetime import date, timedelta
> import random
> import string
> 
> 
> class TestDataGenerator:
>     """Generate various types of test data"""
>     
>     @staticmethod
>     def generate_random_strings(spark, count, length=10, prefix=""):
>         """Generate DataFrame with random strings"""
>         df = spark.range(count)
>         df = df.withColumn(
>             "random_string",
>             concat(
>                 lit(prefix),
>                 expr(f"substring(md5(cast(id as string)), 1, {length})")
>             )
>         )
>         return df.select("random_string")
>     
>     @staticmethod
>     def generate_numeric_data(spark, count, min_val=0, max_val=100):
>         """Generate DataFrame with random numeric data"""
>         df = spark.range(count)
>         df = df.withColumn(
>             "random_int",
>             (rand() * (max_val - min_val) + min_val).cast("int")
>         )
>         df = df.withColumn(
>             "random_double",
>             rand() * (max_val - min_val) + min_val
>         )
>         df = df.withColumn(
>             "normal_distribution",
>             randn() * 10 + 50  # Mean=50, StdDev=10
>         )
>         return df
>     
>     @staticmethod
>     def generate_date_range(spark, start_date, end_date, count):
>         """Generate DataFrame with random dates in range"""
>         days_diff = (end_date - start_date).days
>         
>         df = spark.range(count)
>         df = df.withColumn(
>             "random_days",
>             (rand() * days_diff).cast("int")
>         )
>         df = df.withColumn(
>             "random_date",
>             expr(f"date_add(cast('{start_date}' as date), random_days)")
>         )
>         return df.select("random_date")
>     
>     @staticmethod
>     def generate_categorical_data(spark, count, categories):
>         """Generate DataFrame with random categorical data"""
>         num_categories = len(categories)
>         
>         df = spark.range(count)
>         df = df.withColumn(
>             "category_index",
>             (rand() * num_categories).cast("int")
>         )
>         
>         # Create mapping expression
>         when_expr = None
>         from pyspark.sql.functions import when
>         
>         for i, category in enumerate(categories):
>             if when_expr is None:
>                 when_expr = when(col("category_index") == i, lit(category))
>             else:
>                 when_expr = when_expr.when(col("category_index") == i, lit(category))
>         
>         df = df.withColumn("category", when_expr)
>         return df.select("category")
>     
>     @staticmethod
>     def generate_customer_data(spark, count):
>         """Generate realistic customer test data"""
>         df = spark.range(count)
>         
>         # Generate customer IDs
>         df = df.withColumn("customer_id", monotonically_increasing_id())
>         
>         # Generate names
>         df = df.withColumn(
>             "first_name",
>             expr("substring(md5(cast(id as string)), 1, 8)")
>         )
>         df = df.withColumn(
>             "last_name",
>             expr("substring(md5(cast(id * 2 as string)), 1, 10)")
>         )
>         df = df.withColumn(
>             "full_name",
>             concat(col("first_name"), lit(" "), col("last_name"))
>         )
>         
>         # Generate emails
>         df = df.withColumn(
>             "email",
>             concat(col("first_name"), lit("@example.com"))
>         )
>         
>         # Generate age (18-80)
>         df = df.withColumn("age", (rand() * 62 + 18).cast("int"))
>         
>         # Generate signup date (last 2 years)
>         df = df.withColumn(
>             "signup_days_ago",
>             (rand() * 730).cast("int")
>         )
>         df = df.withColumn(
>             "signup_date",
>             expr("date_sub(current_date(), signup_days_ago)")
>         )
>         
>         # Generate random balance
>         df = df.withColumn("balance", rand() * 10000)
>         
>         # Generate status
>         df = df.withColumn("status_rand", rand())
>         df = df.withColumn(
>             "status",
>             when(col("status_rand") < 0.7, lit("active"))
>             .when(col("status_rand") < 0.9, lit("inactive"))
>             .otherwise(lit("suspended"))
>         )
>         
>         return df.select(
>             "customer_id", "full_name", "email", "age", 
>             "signup_date", "balance", "status"
>         )
>     
>     @staticmethod
>     def generate_transaction_data(spark, count, customer_ids):
>         """Generate transaction data for given customer IDs"""
>         num_customers = len(customer_ids)
>         
>         df = spark.range(count)
>         df = df.withColumn("transaction_id", monotonically_increasing_id())
>         
>         # Randomly assign to customers
>         df = df.withColumn(
>             "customer_index",
>             (rand() * num_customers).cast("int")
>         )
>         
>         # Map to actual customer IDs (simplified - using modulo)
>         df = df.withColumn(
>             "customer_id",
>             (col("customer_index") % num_customers)
>         )
>         
>         # Generate transaction amounts (log-normal distribution)
>         df = df.withColumn(
>             "amount",
>             expr("exp(randn() * 0.5 + 4)")  # Mean ~$55, varies widely
>         )
>         
>         # Generate transaction dates (last 90 days)
>         df = df.withColumn(
>             "days_ago",
>             (rand() * 90).cast("int")
>         )
>         df = df.withColumn(
>             "transaction_date",
>             expr("date_sub(current_date(), days_ago)")
>         )
>         
>         # Generate product categories
>         df = df.withColumn("product_rand", rand())
>         df = df.withColumn(
>             "product_category",
>             when(col("product_rand") < 0.3, lit("Electronics"))
>             .when(col("product_rand") < 0.6, lit("Clothing"))
>             .when(col("product_rand") < 0.8, lit("Food"))
>             .otherwise(lit("Other"))
>         )
>         
>         return df.select(
>             "transaction_id", "customer_id", "amount", 
>             "transaction_date", "product_category"
>         )
> 
> 
> class TestDataGenerationPatterns(unittest.TestCase):
>     """Tests for data generation patterns"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("TestDataGenerationPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>     
>     def test_generate_random_strings(self):
>         """Test random string generation"""
>         df = TestDataGenerator.generate_random_strings(
>             self.spark, 
>             count=100, 
>             length=8, 
>             prefix="TEST_"
>         )
>         
>         self.assertEqual(df.count(), 100)
>         
>         # Verify all strings start with prefix
>         for row in df.take(10):
>             self.assertTrue(row.random_string.startswith("TEST_"))
>     
>     def test_generate_numeric_data(self):
>         """Test numeric data generation"""
>         df = TestDataGenerator.generate_numeric_data(
>             self.spark,
>             count=1000,
>             min_val=10,
>             max_val=100
>         )
>         
>         self.assertEqual(df.count(), 1000)
>         
>         # Verify values are in range
>         stats = df.agg(
>             {"random_int": "min", "random_int": "max"}
>         ).first()
>         
>         min_val = df.agg({"random_int": "min"}).first()[0]
>         max_val = df.agg({"random_int": "max"}).first()[0]
>         
>         self.assertGreaterEqual(min_val, 10)
>         self.assertLessEqual(max_val, 100)
>     
>     def test_generate_date_range(self):
>         """Test date range generation"""
>         start = date(2023, 1, 1)
>         end = date(2023, 12, 31)
>         
>         df = TestDataGenerator.generate_date_range(
>             self.spark,
>             start_date=start,
>             end_date=end,
>             count=500
>         )
>         
>         self.assertEqual(df.count(), 500)
>         
>         # Verify dates are in range
>         min_date = df.agg({"random_date": "min"}).first()[0]
>         max_date = df.agg({"random_date": "max"}).first()[0]
>         
>         self.assertGreaterEqual(min_date, start)
>         self.assertLessEqual(max_date, end)
>     
>     def test_generate_categorical_data(self):
>         """Test categorical data generation"""
>         categories = ["A", "B", "C", "D"]
>         
>         df = TestDataGenerator.generate_categorical_data(
>             self.spark,
>             count=1000,
>             categories=categories
>         )
>         
>         self.assertEqual(df.count(), 1000)
>         
>         # Verify all categories are present
>         distinct_categories = [row.category for row in df.select("category").distinct().collect()]
>         for cat in categories:
>             self.assertIn(cat, distinct_categories)
>     
>     def test_generate_customer_data(self):
>         """Test customer data generation"""
>         df = TestDataGenerator.generate_customer_data(self.spark, 1000)
>         
>         self.assertEqual(df.count(), 1000)
>         
>         # Verify schema
>         expected_columns = ["customer_id", "full_name", "email", "age", "signup_date", "balance", "status"]
>         self.assertEqual(df.columns, expected_columns)
>         
>         # Verify age range
>         min_age = df.agg({"age": "min"}).first()[0]
>         max_age = df.agg({"age": "max"}).first()[0]
>         
>         self.assertGreaterEqual(min_age, 18)
>         self.assertLessEqual(max_age, 80)
>         
>         # Verify status values
>         statuses = [row.status for row in df.select("status").distinct().collect()]
>         for status in statuses:
>             self.assertIn(status, ["active", "inactive", "suspended"])
>     
>     def test_generate_transaction_data(self):
>         """Test transaction data generation"""
>         customer_ids = list(range(100))
>         
>         df = TestDataGenerator.generate_transaction_data(
>             self.spark,
>             count=5000,
>             customer_ids=customer_ids
>         )
>         
>         self.assertEqual(df.count(), 5000)
>         
>         # Verify schema
>         expected_columns = ["transaction_id", "customer_id", "amount", "transaction_date", "product_category"]
>         self.assertEqual(df.columns, expected_columns)
>         
>         # Verify amounts are positive
>         min_amount = df.agg({"amount": "min"}).first()[0]
>         self.assertGreater(min_amount, 0)
>     
>     def test_generate_realistic_dataset_integration(self):
>         """Test generating a complete realistic dataset"""
>         # Generate customers
>         customers_df = TestDataGenerator.generate_customer_data(self.spark, 100)
>         customer_ids = [row.customer_id for row in customers_df.select("customer_id").collect()]
>         
>         # Generate transactions for those customers
>         transactions_df = TestDataGenerator.generate_transaction_data(
>             self.spark,
>             count=1000,
>             customer_ids=customer_ids
>         )
>         
>         # Join and verify
>         joined = customers_df.join(
>             transactions_df,
>             "customer_id",
>             "inner"
>         )
>         
>         self.assertGreater(joined.count(), 0)
>         
>         # Verify we have transactions for multiple customers
>         distinct_customers = joined.select("customer_id").distinct().count()
>         self.assertGreater(distinct_customers, 1)
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 8. Property-based Testing Pattern
> 
> ```python test_patterns/property_based_testing_pattern.py
> """
> Property-based Testing Pattern - Test properties that should hold for all inputs
> Uses hypothesis library for property-based testing
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
> from pyspark.sql.functions import col, sum as spark_sum, count, avg, min as spark_min, max as spark_max
> from hypothesis import given, strategies as st, settings
> from hypothesis.strategies import lists, integers, floats, text
> 
> 
> class DataFrameOperations:
>     """Operations to test with property-based testing"""
>     
>     @staticmethod
>     def filter_and_count(df, column, threshold):
>         """Filter by threshold and return count"""
>         return df.filter(col(column) > threshold).count()
>     
>     @staticmethod
>     def sum_column(df, column):
>         """Sum a numeric column"""
>         result = df.agg(spark_sum(column).alias("total")).first()
>         return result.total if result.total is not None else 0
>     
>     @staticmethod
>     def deduplicate(df, column):
>         """Remove duplicates based on column"""
>         return df.dropDuplicates([column])
> 
> 
> class TestPropertyBasedPatterns(unittest.TestCase):
>     """Property-based tests for PySpark operations"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("PropertyBasedTestingPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>     
>     @given(lists(integers(min_value=0, max_value=1000), min_size=1, max_size=100))
>     @settings(max_examples=50, deadline=None)
>     def test_property_filter_reduces_count(self, numbers):
>         """Property: Filtering should never increase count"""
>         schema = StructType([StructField("value", IntegerType(), True)])
>         data = [(n,) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         original_count = df.count()
>         threshold = 500
>         
>         filtered_count = DataFrameOperations.filter_and_count(df, "value", threshold)
>         
>         # Property: Filtered count should be <= original count
>         self.assertLessEqual(filtered_count, original_count)
>     
>     @given(lists(integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
>     @settings(max_examples=50, deadline=None)
>     def test_property_sum_equals_python_sum(self, numbers):
>         """Property: DataFrame sum should equal Python sum"""
>         schema = StructType([StructField("value", IntegerType(), True)])
>         data = [(n,) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         df_sum = DataFrameOperations.sum_column(df, "value")
>         python_sum = sum(numbers)
>         
>         # Property: DataFrame sum equals Python sum
>         self.assertEqual(df_sum, python_sum)
>     
>     @given(lists(integers(min_value=1, max_value=100), min_size=1, max_size=100))
>     @settings(max_examples=50, deadline=None)
>     def test_property_deduplication_idempotent(self, numbers):
>         """Property: Deduplication is idempotent"""
>         schema = StructType([StructField("id", IntegerType(), True)])
>         data = [(n,) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         # Apply deduplication once
>         dedup_once = DataFrameOperations.deduplicate(df, "id")
>         count_once = dedup_once.count()
>         
>         # Apply deduplication again
>         dedup_twice = DataFrameOperations.deduplicate(dedup_once, "id")
>         count_twice = dedup_twice.count()
>         
>         # Property: Deduplication is idempotent (second application doesn't change result)
>         self.assertEqual(count_once, count_twice)
>     
>     @given(lists(floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False), 
>                  min_size=2, max_size=100))
>     @settings(max_examples=50, deadline=None)
>     def test_property_average_within_bounds(self, numbers):
>         """Property: Average should be between min and max"""
>         schema = StructType([StructField("value", DoubleType(), True)])
>         data = [(float(n),) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         stats = df.agg(
>             avg("value").alias("avg"),
>             spark_min("value").alias("min"),
>             spark_max("value").alias("max")
>         ).first()
>         
>         # Property: min <= average <= max
>         self.assertGreaterEqual(stats.avg, stats.min)
>         self.assertLessEqual(stats.avg, stats.max)
>     
>     @given(
>         lists(integers(min_value=1, max_value=100), min_size=1, max_size=50),
>         integers(min_value=1, max_value=10)
>     )
>     @settings(max_examples=30, deadline=None)
>     def test_property_partition_count_invariant(self, numbers, num_partitions):
>         """Property: Repartitioning doesn't change count"""
>         schema = StructType([StructField("value", IntegerType(), True)])
>         data = [(n,) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         original_count = df.count()
>         repartitioned_df = df.repartition(num_partitions)
>         new_count = repartitioned_df.count()
>         
>         # Property: Count remains same after repartitioning
>         self.assertEqual(original_count, new_count)
>     
>     @given(lists(text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))), 
>                  min_size=1, max_size=50))
>     @settings(max_examples=30, deadline=None)
>     def test_property_union_count(self, strings):
>         """Property: Union count equals sum of individual counts"""
>         if len(strings) < 2:
>             return  # Skip if not enough data
>         
>         mid = len(strings) // 2
>         list1 = strings[:mid]
>         list2 = strings[mid:]
>         
>         schema = StructType([StructField("text", StringType(), True)])
>         
>         df1 = self.spark.createDataFrame([(s,) for s in list1], schema)
>         df2 = self.spark.createDataFrame([(s,) for s in list2], schema)
>         
>         count1 = df1.count()
>         count2 = df2.count()
>         union_count = df1.union(df2).count()
>         
>         # Property: Union count equals sum of individual counts
>         self.assertEqual(union_count, count1 + count2)
>     
>     @given(lists(integers(min_value=0, max_value=100), min_size=5, max_size=100))
>     @settings(max_examples=30, deadline=None)
>     def test_property_groupby_count_sum(self, numbers):
>         """Property: Sum of group counts equals total count"""
>         schema = StructType([
>             StructField("id", IntegerType(), True),
>             StructField("group", IntegerType(), True)
>         ])
>         data = [(i, n % 10) for i, n in enumerate(numbers)]  # Group by mod 10
>         df = self.spark.createDataFrame(data, schema)
>         
>         total_count = df.count()
>         
>         group_counts = df.groupBy("group").agg(count("*").alias("cnt"))
>         sum_of_group_counts = group_counts.agg(spark_sum("cnt").alias("total")).first().total
>         
>         # Property: Sum of all group counts equals total count
>         self.assertEqual(sum_of_group_counts, total_count)
>     
>     @given(lists(integers(min_value=-100, max_value=100), min_size=1, max_size=100))
>     @settings(max_examples=30, deadline=None)
>     def test_property_absolute_value_non_negative(self, numbers):
>         """Property: Absolute values are always non-negative"""
>         from pyspark.sql.functions import abs as spark_abs
>         
>         schema = StructType([StructField("value", IntegerType(), True)])
>         data = [(n,) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         df_with_abs = df.withColumn("abs_value", spark_abs(col("value")))
>         
>         min_abs = df_with_abs.agg(spark_min("abs_value").alias("min")).first().min
>         
>         # Property: Minimum absolute value is non-negative
>         self.assertGreaterEqual(min_abs, 0)
>     
>     @given(
>         lists(integers(min_value=1

#### _User_

> continue exactly where you left off

#### _Assistant_

> ```python test_patterns/property_based_testing_pattern.py
>     @given(
>         lists(integers(min_value=1, max_value=100), min_size=1, max_size=100),
>         integers(min_value=1, max_value=50)
>     )
>     @settings(max_examples=30, deadline=None)
>     def test_property_filter_threshold_monotonic(self, numbers, threshold):
>         """Property: Higher threshold = fewer or equal results"""
>         schema = StructType([StructField("value", IntegerType(), True)])
>         data = [(n,) for n in numbers]
>         df = self.spark.createDataFrame(data, schema)
>         
>         count_at_threshold = df.filter(col("value") > threshold).count()
>         count_at_higher_threshold = df.filter(col("value") > threshold + 10).count()
>         
>         # Property: Stricter filter produces fewer or equal results
>         self.assertLessEqual(count_at_higher_threshold, count_at_threshold)
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 9. Performance Testing Pattern
> 
> ```python test_patterns/performance_testing_pattern.py
> """
> Performance Testing Pattern - Test performance characteristics and benchmarks
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
> from pyspark.sql.functions import col, rand, count, avg, sum as spark_sum
> import time
> from contextlib import contextmanager
> 
> 
> class PerformanceTimer:
>     """Context manager for timing operations"""
>     
>     def __init__(self, operation_name):
>         self.operation_name = operation_name
>         self.start_time = None
>         self.end_time = None
>         self.elapsed = None
>     
>     def __enter__(self):
>         self.start_time = time.time()
>         return self
>     
>     def __exit__(self, exc_type, exc_val, exc_tb):
>         self.end_time = time.time()
>         self.elapsed = self.end_time - self.start_time
>         print(f"{self.operation_name}: {self.elapsed:.4f} seconds")
>         return False
> 
> 
> class PerformanceBenchmark:
>     """Performance benchmark utilities"""
>     
>     @staticmethod
>     def measure_execution_time(func, *args, **kwargs):
>         """Measure execution time of a function"""
>         start = time.time()
>         result = func(*args, **kwargs)
>         end = time.time()
>         return result, end - start
>     
>     @staticmethod
>     def compare_operations(operations, iterations=3):
>         """Compare execution times of multiple operations"""
>         results = {}
>         
>         for name, func in operations.items():
>             times = []
>             for _ in range(iterations):
>                 _, elapsed = PerformanceBenchmark.measure_execution_time(func)
>                 times.append(elapsed)
>             
>             results[name] = {
>                 'min': min(times),
>                 'max': max(times),
>                 'avg': sum(times) / len(times),
>                 'times': times
>             }
>         
>         return results
> 
> 
> class TestPerformancePatterns(unittest.TestCase):
>     """Performance tests for PySpark operations"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("PerformanceTestingPattern") \
>             .master("local[4]") \
>             .config("spark.sql.shuffle.partitions", "8") \
>             .config("spark.default.parallelism", "8") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>     
>     def setUp(self):
>         """Create test datasets for performance testing"""
>         # Small dataset
>         self.small_df = self.spark.range(1000).toDF("id")
>         self.small_df = self.small_df.withColumn("value", (rand() * 100).cast("int"))
>         
>         # Medium dataset
>         self.medium_df = self.spark.range(100000).toDF("id")
>         self.medium_df = self.medium_df.withColumn("value", (rand() * 100).cast("int"))
>         
>         # Large dataset
>         self.large_df = self.spark.range(1000000).toDF("id")
>         self.large_df = self.large_df.withColumn("value", (rand() * 100).cast("int"))
>     
>     def test_count_performance_threshold(self):
>         """Test that count operation completes within threshold"""
>         max_time = 2.0  # seconds
>         
>         with PerformanceTimer("Count operation") as timer:
>             count = self.large_df.count()
>         
>         self.assertGreater(count, 0)
>         self.assertLess(timer.elapsed, max_time, 
>                        f"Count took {timer.elapsed}s, expected < {max_time}s")
>     
>     def test_aggregation_performance(self):
>         """Test aggregation performance on large dataset"""
>         max_time = 5.0  # seconds
>         
>         with PerformanceTimer("Aggregation operation") as timer:
>             result = self.large_df.groupBy("value").agg(
>                 count("*").alias("cnt"),
>                 avg("id").alias("avg_id")
>             ).collect()
>         
>         self.assertGreater(len(result), 0)
>         self.assertLess(timer.elapsed, max_time,
>                        f"Aggregation took {timer.elapsed}s, expected < {max_time}s")
>     
>     def test_join_performance(self):
>         """Test join performance"""
>         max_time = 5.0  # seconds
>         
>         # Create two datasets to join
>         df1 = self.spark.range(10000).toDF("id")
>         df2 = self.spark.range(10000).toDF("id").withColumn("data", rand())
>         
>         with PerformanceTimer("Join operation") as timer:
>             result = df1.join(df2, "id", "inner").count()
>         
>         self.assertEqual(result, 10000)
>         self.assertLess(timer.elapsed, max_time,
>                        f"Join took {timer.elapsed}s, expected < {max_time}s")
>     
>     def test_cache_performance_improvement(self):
>         """Test that caching improves performance for repeated operations"""
>         df = self.medium_df
>         
>         # First run without cache
>         _, time_without_cache_1 = PerformanceBenchmark.measure_execution_time(
>             lambda: df.filter(col("value") > 50).count()
>         )
>         _, time_without_cache_2 = PerformanceBenchmark.measure_execution_time(
>             lambda: df.filter(col("value") > 50).count()
>         )
>         
>         # Now with cache
>         df_cached = df.cache()
>         df_cached.count()  # Materialize cache
>         
>         _, time_with_cache_1 = PerformanceBenchmark.measure_execution_time(
>             lambda: df_cached.filter(col("value") > 50).count()
>         )
>         _, time_with_cache_2 = PerformanceBenchmark.measure_execution_time(
>             lambda: df_cached.filter(col("value") > 50).count()
>         )
>         
>         df_cached.unpersist()
>         
>         # Second run with cache should be faster than second run without cache
>         print(f"Without cache (run 2): {time_without_cache_2:.4f}s")
>         print(f"With cache (run 2): {time_with_cache_2:.4f}s")
>         
>         # Cache should provide some speedup (allowing 10% margin for variance)
>         self.assertLess(time_with_cache_2 * 0.9, time_without_cache_2,
>                        "Cached operation should be faster")
>     
>     def test_partition_performance(self):
>         """Test performance with different partition counts"""
>         df = self.spark.range(100000).toDF("id")
>         df = df.withColumn("value", (rand() * 1000).cast("int"))
>         
>         operations = {
>             '2_partitions': lambda: df.repartition(2).groupBy("value").count().count(),
>             '8_partitions': lambda: df.repartition(8).groupBy("value").count().count(),
>             '32_partitions': lambda: df.repartition(32).groupBy("value").count().count(),
>         }
>         
>         results = PerformanceBenchmark.compare_operations(operations, iterations=2)
>         
>         for name, stats in results.items():
>             print(f"{name}: avg={stats['avg']:.4f}s, min={stats['min']:.4f}s, max={stats['max']:.4f}s")
>         
>         # Verify all operations complete
>         for stats in results.values():
>             self.assertGreater(stats['avg'], 0)
>     
>     def test_broadcast_join_performance(self):
>         """Test broadcast join vs regular join performance"""
>         from pyspark.sql.functions import broadcast
>         
>         # Large table
>         large_table = self.spark.range(100000).toDF("id")
>         large_table = large_table.withColumn("value", rand())
>         
>         # Small table (good candidate for broadcast)
>         small_table = self.spark.range(100).toDF("id")
>         small_table = small_table.withColumn("name", (rand() * 100).cast("string"))
>         
>         # Regular join
>         _, regular_join_time = PerformanceBenchmark.measure_execution_time(
>             lambda: large_table.join(small_table, "id", "inner").count()
>         )
>         
>         # Broadcast join
>         _, broadcast_join_time = PerformanceBenchmark.measure_execution_time(
>             lambda: large_table.join(broadcast(small_table), "id", "inner").count()
>         )
>         
>         print(f"Regular join: {regular_join_time:.4f}s")
>         print(f"Broadcast join: {broadcast_join_time:.4f}s")
>         
>         # Broadcast join should be faster or comparable
>         self.assertLess(broadcast_join_time, regular_join_time * 1.5,
>                        "Broadcast join should be competitive")
>     
>     def test_udf_vs_builtin_performance(self):
>         """Test UDF performance vs built-in functions"""
>         from pyspark.sql.functions import udf, upper
>         from pyspark.sql.types import StringType
>         
>         df = self.spark.range(50000).toDF("id")
>         df = df.withColumn("text", (rand() * 100).cast("string"))
>         
>         # Built-in function
>         _, builtin_time = PerformanceBenchmark.measure_execution_time(
>             lambda: df.withColumn("upper_text", upper(col("text"))).count()
>         )
>         
>         # UDF
>         upper_udf = udf(lambda s: s.upper() if s else None, StringType())
>         _, udf_time = PerformanceBenchmark.measure_execution_time(
>             lambda: df.withColumn("upper_text", upper_udf(col("text"))).count()
>         )
>         
>         print(f"Built-in function: {builtin_time:.4f}s")
>         print(f"UDF: {udf_time:.4f}s")
>         
>         # Built-in should be faster (typically 2-10x)
>         self.assertLess(builtin_time, udf_time,
>                        "Built-in function should be faster than UDF")
>     
>     def test_write_performance_formats(self):
>         """Test write performance for different formats"""
>         import tempfile
>         import shutil
>         
>         temp_dir = tempfile.mkdtemp()
>         
>         try:
>             df = self.spark.range(50000).toDF("id")
>             df = df.withColumn("value", rand())
>             df = df.withColumn("category", (rand() * 10).cast("int").cast("string"))
>             
>             # Test different formats
>             formats = {
>                 'parquet': f"{temp_dir}/parquet",
>                 'csv': f"{temp_dir}/csv",
>                 'json': f"{temp_dir}/json"
>             }
>             
>             results = {}
>             for format_name, path in formats.items():
>                 start = time.time()
>                 if format_name == 'parquet':
>                     df.write.mode("overwrite").parquet(path)
>                 elif format_name == 'csv':
>                     df.write.mode("overwrite").csv(path, header=True)
>                 elif format_name == 'json':
>                     df.write.mode("overwrite").json(path)
>                 end = time.time()
>                 
>                 results[format_name] = end - start
>                 print(f"Write {format_name}: {results[format_name]:.4f}s")
>             
>             # Parquet should typically be fastest
>             self.assertLess(results['parquet'], results['json'],
>                            "Parquet should be faster than JSON")
>         
>         finally:
>             shutil.rmtree(temp_dir)
>     
>     def test_memory_usage_estimation(self):
>         """Test memory usage stays within bounds"""
>         from pyspark import SparkContext
>         
>         # Create a reasonably large dataset
>         df = self.spark.range(500000).toDF("id")
>         df = df.withColumn("value1", rand())
>         df = df.withColumn("value2", rand())
>         df = df.cache()
>         
>         # Materialize cache
>         df.count()
>         
>         # Get storage level info (this is a basic check)
>         storage_level = df.storageLevel
>         self.assertTrue(storage_level.useMemory)
>         
>         df.unpersist()
>     
>     def test_skew_handling_performance(self):
>         """Test performance with skewed data"""
>         from pyspark.sql.functions import when
>         
>         # Create skewed dataset (90% of records have value=1)
>         df = self.spark.range(100000).toDF("id")
>         df = df.withColumn(
>             "skewed_key",
>             when(rand() < 0.9, 1).otherwise((rand() * 100).cast("int"))
>         )
>         df = df.withColumn("value", rand())
>         
>         # Test aggregation on skewed data
>         with PerformanceTimer("Skewed aggregation") as timer:
>             result = df.groupBy("skewed_key").agg(
>                 count("*").alias("cnt"),
>                 avg("value").alias("avg_val")
>             ).collect()
>         
>         # Should complete despite skew
>         self.assertGreater(len(result), 0)
>         
>         # Find the skewed partition
>         skewed_count = [r.cnt for r in result if r.skewed_key == 1]
>         if skewed_count:
>             self.assertGreater(skewed_count[0], 80000)  # ~90% of 100k
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ## 10. Regression Testing Pattern
> 
> ```python test_patterns/regression_testing_pattern.py
> """
> Regression Testing Pattern - Ensure new changes don't break existing functionality
> """
> import unittest
> from pyspark.sql import SparkSession
> from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType
> from pyspark.sql.functions import col, sum as spark_sum, avg, count
> import json
> import tempfile
> import shutil
> from datetime import date
> 
> 
> class DataPipelineV1:
>     """Original version of data pipeline"""
>     
>     @staticmethod
>     def process_sales_data(df):
>         """Process sales data - Version 1"""
>         # Filter out invalid records
>         cleaned = df.filter(col("amount") > 0)
>         
>         # Aggregate by category
>         result = cleaned.groupBy("category").agg(
>             spark_sum("amount").alias("total_sales"),
>             count("*").alias("transaction_count")
>         )
>         
>         return result
> 
> 
> class DataPipelineV2:
>     """Updated version of data pipeline with enhancements"""
>     
>     @staticmethod
>     def process_sales_data(df):
>         """Process sales data - Version 2 (with additional features)"""
>         # Filter out invalid records (same as V1)
>         cleaned = df.filter(col("amount") > 0)
>         
>         # Aggregate by category with additional metrics
>         result = cleaned.groupBy("category").agg(
>             spark_sum("amount").alias("total_sales"),
>             count("*").alias("transaction_count"),
>             avg("amount").alias("avg_transaction")  # New metric
>         )
>         
>         return result
> 
> 
> class RegressionTestHelper:
>     """Helper for regression testing"""
>     
>     @staticmethod
>     def save_baseline(df, path):
>         """Save baseline results for comparison"""
>         df.write.mode("overwrite").json(path)
>     
>     @staticmethod
>     def load_baseline(spark, path):
>         """Load baseline results"""
>         return spark.read.json(path)
>     
>     @staticmethod
>     def compare_dataframes(df1, df2, key_columns, value_columns, tolerance=0.01):
>         """Compare two DataFrames with tolerance for numeric values"""
>         from pyspark.sql.functions import abs as spark_abs
>         
>         # Join on key columns
>         joined = df1.alias("baseline").join(
>             df2.alias("current"),
>             key_columns,
>             "outer"
>         )
>         
>         differences = []
>         
>         for col_name in value_columns:
>             baseline_col = f"baseline.{col_name}"
>             current_col = f"current.{col_name}"
>             
>             # Check for missing values
>             missing_in_current = joined.filter(
>                 col(baseline_col).isNotNull() & col(current_col).isNull()
>             ).count()
>             
>             missing_in_baseline = joined.filter(
>                 col(baseline_col).isNull() & col(current_col).isNotNull()
>             ).count()
>             
>             if missing_in_current > 0 or missing_in_baseline > 0:
>                 differences.append({
>                     'column': col_name,
>                     'missing_in_current': missing_in_current,
>                     'missing_in_baseline': missing_in_baseline
>                 })
>             
>             # Check for value differences (for numeric columns)
>             try:
>                 diff_df = joined.filter(
>                     col(baseline_col).isNotNull() & col(current_col).isNotNull()
>                 ).withColumn(
>                     "diff",
>                     spark_abs(col(baseline_col) - col(current_col))
>                 ).filter(col("diff") > tolerance)
>                 
>                 diff_count = diff_df.count()
>                 if diff_count > 0:
>                     differences.append({
>                         'column': col_name,
>                         'significant_differences': diff_count
>                     })
>             except:
>                 # Non-numeric column, skip numeric comparison
>                 pass
>         
>         return differences
> 
> 
> class TestRegressionPatterns(unittest.TestCase):
>     """Regression tests to ensure backwards compatibility"""
>     
>     @classmethod
>     def setUpClass(cls):
>         cls.spark = SparkSession.builder \
>             .appName("RegressionTestingPattern") \
>             .master("local[2]") \
>             .getOrCreate()
>         cls.spark.sparkContext.setLogLevel("ERROR")
>         
>         cls.temp_dir = tempfile.mkdtemp()
>         cls.baseline_path = f"{cls.temp_dir}/baseline"
>     
>     @classmethod
>     def tearDownClass(cls):
>         cls.spark.stop()
>         shutil.rmtree(cls.temp_dir)
>     
>     def setUp(self):
>         """Create test data"""
>         schema = StructType([
>             StructField("transaction_id", IntegerType(), False),
>             StructField("category", StringType(), True),
>             StructField("amount", DoubleType(), True),
>             StructField("date", DateType(), True)
>         ])
>         
>         self.test_data = [
>             (1, "Electronics", 299.99, date(2024, 1, 15)),
>             (2, "Electronics", 599.99, date(2024, 1, 16)),
>             (3, "Clothing", 49.99, date(2024, 1, 17)),
>             (4, "Clothing", 89.99, date(2024, 1, 18)),
>             (5, "Food", 25.50, date(2024, 1, 19)),
>             (6, "Food", -10.00, date(2024, 1, 20)),  # Invalid
>             (7, "Electronics", 0.00, date(2024, 1, 21)),  # Invalid
>         ]
>         
>         self.df = self.spark.createDataFrame(self.test_data, schema)
>     
>     def test_v2_produces_same_core_results_as_v1(self):
>         """Test that V2 produces same core results as V1"""
>         # Run both versions
>         result_v1 = DataPipelineV1.process_sales_data(self.df)
>         result_v2 = DataPipelineV2.process_sales_data(self.df)
>         
>         # Compare common columns
>         common_columns = ["category", "total_sales", "transaction_count"]
>         
>         result_v1_subset = result_v1.select(common_columns)
>         result_v2_subset = result_v2.select(common_columns)
>         
>         # Convert to comparable format
>         v1_data = {row.category: (row.total_sales, row.transaction_count) 
>                    for row in result_v1_subset.collect()}
>         v2_data = {row.category: (row.total_sales, row.transaction_count) 
>                    for row in result_v2_subset.collect()}
>         
>         # Assert same results for core functionality
>         self.assertEqual(v1_data, v2_data,
>                         "V2 should produce same core results as V1")
>     
>     def test_baseline_comparison(self):
>         """Test against saved baseline results"""
>         # Generate baseline (in real scenario, this would be from a previous run)
>         baseline_result = DataPipelineV1.process_sales_data(self.df)
>         baseline_path = f"{self.baseline_path}/sales_baseline"
>         RegressionTestHelper.save_baseline(baseline_result, baseline_path)
>         
>         # Run current version
>         current_result = DataPipelineV2.process_sales_data(self.df)
>         
>         # Load baseline
>         baseline_loaded = RegressionTestHelper.load_baseline(self.spark, baseline_path)
>         
>         # Compare (only common columns for regression test)
>         differences = RegressionTestHelper.compare_dataframes(
>             baseline_loaded.select("category", "total_sales", "transaction_count"),
>             current_result.select("category", "total_sales", "transaction_count"),
>             key_columns=["category"],
>             value_columns=["total_sales", "transaction_count"],
>             tolerance=0.01
>         )
>         
>         # Assert no significant differences in core metrics
>         self.assertEqual(len(differences), 0,
>                         f"Found differences from baseline: {differences}")
>     
>     def test_output_schema_backwards_compatible(self):
>         """Test that output schema is backwards compatible"""
>         result_v1 = DataPipelineV1.process_sales_data(self.df)
>         result_v2 = DataPipelineV2.process_sales_data(self.df)
>         
>         # V2 should have all columns from V1
>         v1_columns = set(result_v1.columns)
>         v2_columns = set(result_v2.columns)
>         
>         self.assertTrue(v1_columns.issubset(v2_columns),
>                        "V2 should include all V1 columns for backwards compatibility")
>         
>         # Check data types are compatible for common columns
>         v1_schema = {f.name: f.dataType for f in result_v1.schema.fields}
>         v2_schema = {f.name: f.dataType for f in result_v2.schema.fields}
>         
>         for col_name in v1_columns:
>             self.assertEqual(v1_schema[col_name], v2_schema[col_name],
>                            f"Data type for {col_name} changed")
>     
>     def test_edge_cases_still_handled(self):
>         """Test that edge cases from V1 are still handled in V2"""
>         # Empty DataFrame
>         empty_df = self.spark.createDataFrame([], self.df.schema)
>         
>         result_v1_empty = DataPipelineV1.process_sales_data(empty_df)
>         result_v2_empty = DataPipelineV2.process_sales_data(empty_df)
>         
>         self.assertEqual(result_v1_empty.count(), 0)
>         self.assertEqual(result_v2_empty.count(), 0)
>         
>         # All invalid data
>         invalid_schema = StructType([
>             StructField("transaction_id", IntegerType(), False),
>             StructField("category", StringType(), True),
>             StructField("amount", DoubleType(), True),
>             StructField("date", DateType(), True)
>         ])
>         
>         invalid_data = [
>             (1, "Test", -100.0, date(2024, 1, 1)),
>             (2, "Test", 0.0, date(2024, 1, 2)),
>             (3, "Test", -50.0, date(2024, 1, 3))
>         ]
>         
>         invalid_df = self.spark.createDataFrame(invalid_data, invalid_schema)
>         
>         result_v1_invalid = DataPipelineV1.process_sales_data(invalid_df)
>         result_v2_invalid = DataPipelineV2.process_sales_data(invalid_df)
>         
>         self.assertEqual(result_v1_invalid.count(), 0)
>         self.assertEqual(result_v2_invalid.count(), 0)
>     
>     def test_performance_regression(self):
>         """Test that performance hasn't regressed significantly"""
>         import time
>         
>         # Create larger dataset for performance testing
>         large_data = []
>         for i in range(10000):
>             large_data.append((
>                 i,
>                 ["Electronics", "Clothing", "Food"][i % 3],
>                 float(i % 1000 + 1),
>                 date(2024, 1, (i % 28) + 1)
>             ))
>         
>         large_df = self.spark.createDataFrame(large_data, self.df.schema)
>         
>         # Measure V1
>         start_v1 = time.time()
>         result_v1 = DataPipelineV1.process_sales_data(large_df)
>         result_v1.count()  # Force evaluation
>         time_v1 = time.time() - start_v1
>         
>         # Measure V2
>         start_v2 = time.time()
>         result_v2 = DataPipelineV2.process_sales_data(large_df)
>         result_v2.count()  # Force evaluation
>         time_v2 = time.time() - start_v2
>         
>         print(f"V1 time: {time_v1:.4f}s")
>         print(f"V2 time: {time_v2:.4f}s")
>         
>         # V2 should not be significantly slower (allow 50% overhead)
>         self.assertLess(time_v2, time_v1 * 1.5,
>                        f"V2 is too slow: {time_v2}s vs V1: {time_v1}s")
>     
>     def test_null_handling_consistency(self):
>         """Test that null handling is consistent between versions"""
>         null_schema = StructType([
>             StructField("transaction_id", IntegerType(), False),
>             StructField("category", StringType(), True),
>             StructField("amount", DoubleType(), True),
>             StructField("date", DateType(), True)
>         ])
>         
>         null_data = [
>             (1, None, 100.0, date(2024, 1, 1)),
>             (2, "Electronics", None, date(2024, 1, 2)),
>             (3, "Food", 50.0, None),
>             (4, "Clothing", 75.0, date(2024, 1, 3))
>         ]
>         
>         null_df = self.spark.createDataFrame(null_data, null_schema)
>         
>         result_v1 = DataPipelineV1.process_sales_data(null_df)
>         result_v2 = DataPipelineV2.process_sales_data(null_df)
>         
>         # Both should handle nulls the same way
>         v1_count = result_v1.count()
>         v2_count = result_v2.count()
>         
>         self.assertEqual(v1_count, v2_count,
>                         "Null handling should be consistent")
>     
>     def test_deterministic_results(self):
>         """Test that results are deterministic across multiple runs"""
>         # Run V2 multiple times
>         results = []
>         for _ in range(3):
>             result = DataPipelineV2.process_sales_data(self.df)
>             result_data = sorted(
>                 [(r.category, r.total_sales, r.transaction_count) 
>                  for r in result.collect()],
>                 key=lambda x: x[0]
>             )
>             results.append(result_data)
>         
>         # All runs should produce identical results
>         for i in range(1, len(results)):
>             self.assertEqual(results[0], results[i],
>                            "Results should be deterministic")
> 
> 
> if __name__ == "__main__":
>     unittest.main()
> ```
> 
> ---
> 
> ## Summary
> 
> I've created comprehensive PySpark testing patterns covering all 10 requested patterns:
> 
> 1. **Unit Testing Pattern** - Tests individual functions in isolation
> 2. **Integration Testing Pattern** - Tests complete ETL pipelines
> 3. **Data Validation Testing Pattern** - Validates data quality and constraints
> 4. **Schema Testing Pattern** - Validates DataFrame schemas
> 5. **Mock Data Pattern** - Creates and uses mock data for testing
> 6. **Fixture Pattern** - Reusable test data setup and teardown
> 7. **Test Data Generation Pattern** - Programmatically generates test data
> 8. **Property-based Testing Pattern** - Tests properties using hypothesis
> 9. **Performance Testing Pattern** - Tests performance characteristics
> 10. **Regression Testing Pattern** - Ensures backwards compatibility
> 
> Each pattern includes practical examples, best practices, and can be run independently using `python <filename>`.