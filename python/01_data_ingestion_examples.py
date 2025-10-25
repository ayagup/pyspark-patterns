"""
PySpark Data Ingestion Patterns - Example Implementations
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast
from abc import ABC, abstractmethod

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DataIngestionPatterns") \
    .master("local[*]") \
    .getOrCreate()


# ============================================
# 1.1 Schema Evolution Pattern
# ============================================
def schema_evolution_example():
    """Demonstrate schema evolution with mergeSchema"""
    print("\n=== Schema Evolution Pattern ===")
    
    # Create initial data
    df1 = spark.createDataFrame([
        (1, "Alice", 30),
        (2, "Bob", 25)
    ], ["id", "name", "age"])
    df1.write.mode("overwrite").parquet("/tmp/schema_evolution")
    
    # Add new data with additional column
    df2 = spark.createDataFrame([
        (3, "Charlie", 35, "Engineering"),
        (4, "Diana", 28, "Marketing")
    ], ["id", "name", "age", "department"])
    df2.write.mode("append").parquet("/tmp/schema_evolution")
    
    # Read with schema merge
    df_merged = spark.read.option("mergeSchema", "true").parquet("/tmp/schema_evolution")
    print("Merged Schema:")
    df_merged.printSchema()
    df_merged.show()


# ============================================
# 1.2 Incremental Load Pattern
# ============================================
def incremental_load_example():
    """Demonstrate incremental data loading"""
    print("\n=== Incremental Load Pattern ===")
    
    # Create source data with timestamps
    from pyspark.sql.functions import current_timestamp, date_add, lit
    
    df_initial = spark.createDataFrame([
        (1, "Product A", 100, "2024-01-01"),
        (2, "Product B", 200, "2024-01-02"),
        (3, "Product C", 150, "2024-01-03")
    ], ["id", "name", "price", "date"])
    
    df_initial.write.mode("overwrite").parquet("/tmp/source_data")
    
    # Simulate target table
    df_target = df_initial.limit(2)
    df_target.write.mode("overwrite").parquet("/tmp/target_data")
    
    # Get last loaded date
    last_date = spark.read.parquet("/tmp/target_data") \
        .agg({"date": "max"}).collect()[0][0]
    
    print(f"Last loaded date: {last_date}")
    
    # Incremental load - only new records
    df_incremental = spark.read.parquet("/tmp/source_data") \
        .filter(col("date") > last_date)
    
    print("Incremental records:")
    df_incremental.show()


# ============================================
# 1.3 Multiformat Reader Pattern
# ============================================
class DataReader:
    """Unified interface for reading multiple formats"""
    
    def __init__(self, spark):
        self.spark = spark
    
    def read(self, format_type, path, **options):
        readers = {
            'parquet': lambda: self.spark.read.parquet(path),
            'csv': lambda: self.spark.read.option("header", "true").csv(path),
            'json': lambda: self.spark.read.json(path),
        }
        return readers.get(format_type, lambda: None)()


def multiformat_reader_example():
    """Demonstrate multiformat reader"""
    print("\n=== Multiformat Reader Pattern ===")
    
    # Create sample data in different formats
    df = spark.createDataFrame([
        (1, "Alice", 30),
        (2, "Bob", 25),
        (3, "Charlie", 35)
    ], ["id", "name", "age"])
    
    # Write in different formats
    df.write.mode("overwrite").parquet("/tmp/data.parquet")
    df.write.mode("overwrite").option("header", "true").csv("/tmp/data.csv")
    df.write.mode("overwrite").json("/tmp/data.json")
    
    # Read using unified reader
    reader = DataReader(spark)
    
    print("Reading Parquet:")
    reader.read("parquet", "/tmp/data.parquet").show()
    
    print("Reading CSV:")
    reader.read("csv", "/tmp/data.csv").show()
    
    print("Reading JSON:")
    reader.read("json", "/tmp/data.json").show()


# ============================================
# 1.4 Data Source Abstraction Pattern
# ============================================
class DataSource(ABC):
    """Abstract data source interface"""
    @abstractmethod
    def read(self, spark):
        pass


class ParquetDataSource(DataSource):
    """Parquet data source implementation"""
    def __init__(self, path):
        self.path = path
    
    def read(self, spark):
        return spark.read.parquet(self.path)


class CSVDataSource(DataSource):
    """CSV data source implementation"""
    def __init__(self, path):
        self.path = path
    
    def read(self, spark):
        return spark.read.option("header", "true").csv(self.path)


def data_source_abstraction_example():
    """Demonstrate data source abstraction"""
    print("\n=== Data Source Abstraction Pattern ===")
    
    # Create sample data
    df = spark.createDataFrame([
        (1, "Product A", 100),
        (2, "Product B", 200)
    ], ["id", "name", "price"])
    
    df.write.mode("overwrite").parquet("/tmp/products.parquet")
    df.write.mode("overwrite").option("header", "true").csv("/tmp/products.csv")
    
    # Use abstraction
    parquet_source = ParquetDataSource("/tmp/products.parquet")
    csv_source = CSVDataSource("/tmp/products.csv")
    
    print("From Parquet Source:")
    parquet_source.read(spark).show()
    
    print("From CSV Source:")
    csv_source.read(spark).show()


# ============================================
# 1.5 Partition Pruning Pattern
# ============================================
def partition_pruning_example():
    """Demonstrate partition pruning for efficient reads"""
    print("\n=== Partition Pruning Pattern ===")
    
    # Create partitioned data
    df = spark.createDataFrame([
        (1, "Alice", "2024", "01"),
        (2, "Bob", "2024", "01"),
        (3, "Charlie", "2024", "02"),
        (4, "Diana", "2024", "02"),
        (5, "Eve", "2024", "03")
    ], ["id", "name", "year", "month"])
    
    df.write.mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("/tmp/partitioned_data")
    
    # Read with partition pruning
    print("Reading only 2024-01 partition:")
    df_filtered = spark.read.parquet("/tmp/partitioned_data") \
        .filter((col("year") == "2024") & (col("month") == "01"))
    
    df_filtered.show()
    print(f"Records read: {df_filtered.count()}")


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("PySpark Data Ingestion Patterns Examples")
    print("=" * 60)
    
    schema_evolution_example()
    incremental_load_example()
    multiformat_reader_example()
    data_source_abstraction_example()
    partition_pruning_example()
    
    spark.stop()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
