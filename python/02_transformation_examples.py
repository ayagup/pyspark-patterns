"""
PySpark Data Transformation Patterns - Example Implementations
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, upper, trim, when, sum, avg, row_number
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("TransformationPatterns") \
    .master("local[*]") \
    .getOrCreate()


# ============================================
# 2.1 Pipeline Pattern
# ============================================
class TransformationPipeline:
    """Chain transformations in a reusable way"""
    
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, func):
        self.transformations.append(func)
        return self
    
    def execute(self, df: DataFrame) -> DataFrame:
        for transform in self.transformations:
            df = transform(df)
        return df


def pipeline_pattern_example():
    """Demonstrate pipeline pattern"""
    print("\n=== Pipeline Pattern ===")
    
    # Create sample data
    df = spark.createDataFrame([
        (1, "alice", 17, 1000),
        (2, "  bob  ", 25, 2000),
        (3, "charlie", 30, 1500),
        (4, "diana", 15, 500)
    ], ["id", "name", "age", "salary"])
    
    print("Original Data:")
    df.show()
    
    # Build pipeline
    pipeline = TransformationPipeline()
    pipeline.add_transformation(lambda df: df.filter(col("age") >= 18))
    pipeline.add_transformation(lambda df: df.withColumn("name", upper(trim(col("name")))))
    pipeline.add_transformation(lambda df: df.withColumn("salary_bonus", col("salary") * 0.1))
    
    # Execute pipeline
    result = pipeline.execute(df)
    print("After Pipeline:")
    result.show()


# ============================================
# 2.2 Transformation Registry Pattern
# ============================================
class TransformationRegistry:
    """Register and manage reusable transformations"""
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name):
        return cls._registry.get(name)


@TransformationRegistry.register("clean_names")
def clean_names(df):
    return df.withColumn("name", trim(upper(col("name"))))


@TransformationRegistry.register("add_age_category")
def add_age_category(df):
    return df.withColumn("age_category",
        when(col("age") < 18, "minor")
        .when((col("age") >= 18) & (col("age") < 65), "adult")
        .otherwise("senior")
    )


def registry_pattern_example():
    """Demonstrate transformation registry"""
    print("\n=== Transformation Registry Pattern ===")
    
    df = spark.createDataFrame([
        (1, "  alice  ", 17),
        (2, "bob", 25),
        (3, "charlie", 70)
    ], ["id", "name", "age"])
    
    print("Original Data:")
    df.show()
    
    # Apply registered transformations
    df_cleaned = TransformationRegistry.get("clean_names")(df)
    df_categorized = TransformationRegistry.get("add_age_category")(df_cleaned)
    
    print("After Transformations:")
    df_categorized.show()


# ============================================
# 2.3 Window Function Pattern
# ============================================
def window_function_example():
    """Demonstrate window functions"""
    print("\n=== Window Function Pattern ===")
    
    # Create sales data
    df = spark.createDataFrame([
        ("Sales", "Alice", 5000),
        ("Sales", "Bob", 6000),
        ("Sales", "Charlie", 5500),
        ("Engineering", "Diana", 7000),
        ("Engineering", "Eve", 7500),
        ("Engineering", "Frank", 6500)
    ], ["department", "name", "salary"])
    
    print("Original Data:")
    df.show()
    
    # Ranking within department
    window_spec = Window.partitionBy("department").orderBy(col("salary").desc())
    df_ranked = df.withColumn("rank", row_number().over(window_spec))
    
    print("With Ranking:")
    df_ranked.show()
    
    # Running total
    window_spec_running = Window.partitionBy("department") \
        .orderBy("salary") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    df_with_running = df_ranked.withColumn(
        "running_total", 
        sum("salary").over(window_spec_running)
    )
    
    print("With Running Total:")
    df_with_running.show()


# ============================================
# 2.4 Pivot Pattern
# ============================================
def pivot_pattern_example():
    """Demonstrate pivot operations"""
    print("\n=== Pivot Pattern ===")
    
    # Create sales data
    df = spark.createDataFrame([
        ("2024-01", "Electronics", 1000),
        ("2024-01", "Clothing", 500),
        ("2024-02", "Electronics", 1200),
        ("2024-02", "Clothing", 600),
        ("2024-03", "Electronics", 1100),
        ("2024-03", "Clothing", 550)
    ], ["month", "category", "sales"])
    
    print("Original Data:")
    df.show()
    
    # Pivot to wide format
    df_pivot = df.groupBy("month").pivot("category").sum("sales")
    
    print("Pivoted Data:")
    df_pivot.show()


# ============================================
# 2.5 Deduplication Pattern
# ============================================
def deduplication_example():
    """Demonstrate deduplication strategies"""
    print("\n=== Deduplication Pattern ===")
    
    from pyspark.sql.functions import col
    from datetime import datetime
    
    # Create data with duplicates
    df = spark.createDataFrame([
        (1, "Alice", "2024-01-01", 100),
        (1, "Alice", "2024-01-02", 150),  # duplicate, newer
        (2, "Bob", "2024-01-01", 200),
        (3, "Charlie", "2024-01-01", 300),
        (3, "Charlie", "2024-01-01", 300)  # exact duplicate
    ], ["id", "name", "date", "amount"])
    
    print("Original Data with Duplicates:")
    df.show()
    
    # Simple deduplication
    df_deduped_simple = df.dropDuplicates(["id"])
    print("Simple Deduplication (by id):")
    df_deduped_simple.show()
    
    # Keep latest record
    window_spec = Window.partitionBy("id").orderBy(col("date").desc())
    df_deduped_latest = df.withColumn("rn", row_number().over(window_spec)) \
        .filter(col("rn") == 1) \
        .drop("rn")
    
    print("Deduplication (keep latest):")
    df_deduped_latest.show()


# ============================================
# 2.6 Data Enrichment Pattern
# ============================================
def data_enrichment_example():
    """Demonstrate data enrichment with joins"""
    print("\n=== Data Enrichment Pattern ===")
    
    from pyspark.sql.functions import broadcast, coalesce, lit
    
    # Main data
    df_orders = spark.createDataFrame([
        (1, 101, 1000),
        (2, 102, 2000),
        (3, 103, 1500),
        (4, 999, 500)  # No matching customer
    ], ["order_id", "customer_id", "amount"])
    
    # Lookup data
    df_customers = spark.createDataFrame([
        (101, "Alice", "US"),
        (102, "Bob", "UK"),
        (103, "Charlie", "CA")
    ], ["customer_id", "name", "country"])
    
    print("Orders:")
    df_orders.show()
    
    print("Customers:")
    df_customers.show()
    
    # Enrich with broadcast join
    df_enriched = df_orders.join(
        broadcast(df_customers), 
        "customer_id", 
        "left"
    ).withColumn("name", coalesce(col("name"), lit("UNKNOWN")))
    
    print("Enriched Orders:")
    df_enriched.show()


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("PySpark Transformation Patterns Examples")
    print("=" * 60)
    
    pipeline_pattern_example()
    registry_pattern_example()
    window_function_example()
    pivot_pattern_example()
    deduplication_example()
    data_enrichment_example()
    
    spark.stop()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
