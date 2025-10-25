"""
PySpark Data Quality Patterns - Example Implementations
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev, abs, lit, count, countDistinct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DataQualityPatterns") \
    .master("local[*]") \
    .getOrCreate()


# ============================================
# 3.1 Validation Pattern
# ============================================
class DataValidator:
    """Validate data quality rules"""
    
    def __init__(self, df):
        self.df = df
        self.validations = []
    
    def add_rule(self, name, condition):
        self.validations.append((name, condition))
        return self
    
    def validate(self):
        results = {}
        total_count = self.df.count()
        
        for name, condition in self.validations:
            invalid_count = self.df.filter(~condition).count()
            results[name] = {
                'invalid_count': invalid_count,
                'valid_percentage': ((total_count - invalid_count) / total_count) * 100
            }
        return results


def validation_pattern_example():
    """Demonstrate data validation"""
    print("\n=== Validation Pattern ===")
    
    # Create sample data with quality issues
    df = spark.createDataFrame([
        (1, "alice@email.com", 25, 50000),
        (2, None, 30, 60000),  # Missing email
        (3, "charlie@email.com", -5, 55000),  # Invalid age
        (4, "diana@email.com", 35, -1000),  # Invalid salary
        (5, "eve@email.com", 28, 58000)
    ], ["id", "email", "age", "salary"])
    
    print("Original Data:")
    df.show()
    
    # Create validator
    validator = DataValidator(df)
    validator.add_rule("email_not_null", col("email").isNotNull())
    validator.add_rule("age_positive", col("age") > 0)
    validator.add_rule("salary_positive", col("salary") > 0)
    validator.add_rule("age_reasonable", (col("age") >= 18) & (col("age") <= 100))
    
    # Run validation
    results = validator.validate()
    
    print("\nValidation Results:")
    for rule, metrics in results.items():
        print(f"{rule}:")
        print(f"  Invalid Count: {metrics['invalid_count']}")
        print(f"  Valid Percentage: {metrics['valid_percentage']:.2f}%")


# ============================================
# 3.2 Null Handling Pattern
# ============================================
from pyspark.sql.functions import coalesce, last
from pyspark.sql.window import Window


class NullHandlingStrategy:
    """Base class for null handling strategies"""
    def handle(self, df, column):
        raise NotImplementedError


class FillDefaultStrategy(NullHandlingStrategy):
    """Fill nulls with default value"""
    def __init__(self, default_value):
        self.default_value = default_value
    
    def handle(self, df, column):
        return df.withColumn(column, coalesce(col(column), lit(self.default_value)))


class FillMeanStrategy(NullHandlingStrategy):
    """Fill nulls with mean value"""
    def handle(self, df, column):
        mean_value = df.select(mean(col(column))).collect()[0][0]
        return df.fillna({column: mean_value})


def null_handling_example():
    """Demonstrate null handling strategies"""
    print("\n=== Null Handling Pattern ===")
    
    # Create data with nulls
    df = spark.createDataFrame([
        (1, "Alice", 30, 50000),
        (2, "Bob", None, 60000),
        (3, "Charlie", 35, None),
        (4, None, 28, 55000),
        (5, "Eve", 32, 58000)
    ], ["id", "name", "age", "salary"])
    
    print("Original Data with Nulls:")
    df.show()
    
    # Fill name with default
    name_strategy = FillDefaultStrategy("UNKNOWN")
    df_filled_name = name_strategy.handle(df, "name")
    
    # Fill age with mean
    age_strategy = FillMeanStrategy()
    df_filled_age = age_strategy.handle(df_filled_name, "age")
    
    # Fill salary with mean
    salary_strategy = FillMeanStrategy()
    df_filled_salary = salary_strategy.handle(df_filled_age, "salary")
    
    print("After Null Handling:")
    df_filled_salary.show()


# ============================================
# 3.3 Outlier Detection Pattern
# ============================================
def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
    Q1, Q3 = quantiles[0], quantiles[1]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df.withColumn(
        f"{column}_is_outlier",
        when((col(column) < lower_bound) | (col(column) > upper_bound), True)
        .otherwise(False)
    )


def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    stats = df.select(mean(col(column)), stddev(col(column))).collect()[0]
    mean_val, stddev_val = stats[0], stats[1]
    
    return df.withColumn(
        f"{column}_zscore",
        abs((col(column) - lit(mean_val)) / lit(stddev_val))
    ).withColumn(
        f"{column}_is_outlier",
        col(f"{column}_zscore") > threshold
    )


def outlier_detection_example():
    """Demonstrate outlier detection"""
    print("\n=== Outlier Detection Pattern ===")
    
    # Create data with outliers
    df = spark.createDataFrame([
        (1, 100),
        (2, 105),
        (3, 98),
        (4, 102),
        (5, 500),  # Outlier
        (6, 101),
        (7, 99),
        (8, 1000),  # Outlier
        (9, 103)
    ], ["id", "value"])
    
    print("Original Data:")
    df.show()
    
    # Detect outliers using IQR
    df_iqr = detect_outliers_iqr(df, "value")
    print("Outlier Detection (IQR):")
    df_iqr.show()
    
    # Detect outliers using Z-score
    df_zscore = detect_outliers_zscore(df, "value", threshold=2)
    print("Outlier Detection (Z-score):")
    df_zscore.select("id", "value", "value_zscore", "value_is_outlier").show()


# ============================================
# 3.4 Data Profiling Pattern
# ============================================
def profile_dataframe(df):
    """Generate comprehensive data profile"""
    profile = {
        'row_count': df.count(),
        'column_count': len(df.columns),
        'columns': {}
    }
    
    for col_name in df.columns:
        col_stats = df.select(
            count(col(col_name)).alias('count'),
            count(when(col(col_name).isNull(), True)).alias('null_count'),
            countDistinct(col(col_name)).alias('distinct_count')
        ).collect()[0]
        
        profile['columns'][col_name] = {
            'count': col_stats['count'],
            'null_count': col_stats['null_count'],
            'null_percentage': (col_stats['null_count'] / profile['row_count']) * 100,
            'distinct_count': col_stats['distinct_count'],
            'cardinality': col_stats['distinct_count'] / profile['row_count']
        }
    
    return profile


def data_profiling_example():
    """Demonstrate data profiling"""
    print("\n=== Data Profiling Pattern ===")
    
    # Create sample data
    df = spark.createDataFrame([
        (1, "Alice", "Engineering", 50000),
        (2, "Bob", "Sales", 60000),
        (3, "Charlie", "Engineering", 55000),
        (4, None, "Marketing", 52000),
        (5, "Eve", None, 58000)
    ], ["id", "name", "department", "salary"])
    
    print("Data to Profile:")
    df.show()
    
    # Generate profile
    profile = profile_dataframe(df)
    
    print("\nData Profile:")
    print(f"Total Rows: {profile['row_count']}")
    print(f"Total Columns: {profile['column_count']}")
    print("\nColumn Statistics:")
    for col_name, stats in profile['columns'].items():
        print(f"\n{col_name}:")
        print(f"  Non-null Count: {stats['count']}")
        print(f"  Null Count: {stats['null_count']}")
        print(f"  Null %: {stats['null_percentage']:.2f}%")
        print(f"  Distinct Count: {stats['distinct_count']}")
        print(f"  Cardinality: {stats['cardinality']:.2f}")


# ============================================
# 3.5 Schema Validation Pattern
# ============================================
class SchemaValidator:
    """Validate DataFrame schema"""
    
    def __init__(self, expected_schema):
        self.expected_schema = expected_schema
    
    def validate(self, df):
        """Validate DataFrame against expected schema"""
        errors = []
        
        # Check column names
        expected_cols = set(self.expected_schema.fieldNames())
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            errors.append(f"Extra columns: {extra_cols}")
        
        # Check data types
        for field in self.expected_schema.fields:
            if field.name in df.columns:
                actual_type = df.schema[field.name].dataType
                if actual_type != field.dataType:
                    errors.append(
                        f"Column {field.name}: expected {field.dataType}, got {actual_type}"
                    )
        
        return {'valid': len(errors) == 0, 'errors': errors}


def schema_validation_example():
    """Demonstrate schema validation"""
    print("\n=== Schema Validation Pattern ===")
    
    # Define expected schema
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("age", IntegerType(), True)
    ])
    
    # Create DataFrame with correct schema
    df_correct = spark.createDataFrame([
        (1, "Alice", 30),
        (2, "Bob", 25)
    ], schema=expected_schema)
    
    # Create DataFrame with incorrect schema
    df_incorrect = spark.createDataFrame([
        (1, "Alice", "30", "extra"),  # age as string, extra column
        (2, "Bob", "25", "data")
    ], ["id", "name", "age", "extra_col"])
    
    # Validate
    validator = SchemaValidator(expected_schema)
    
    print("Validating Correct Schema:")
    result_correct = validator.validate(df_correct)
    print(f"Valid: {result_correct['valid']}")
    print(f"Errors: {result_correct['errors']}")
    
    print("\nValidating Incorrect Schema:")
    result_incorrect = validator.validate(df_incorrect)
    print(f"Valid: {result_incorrect['valid']}")
    print(f"Errors: {result_incorrect['errors']}")


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("PySpark Data Quality Patterns Examples")
    print("=" * 60)
    
    validation_pattern_example()
    null_handling_example()
    outlier_detection_example()
    data_profiling_example()
    schema_validation_example()
    
    spark.stop()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
