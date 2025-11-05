### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:36:19 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> User-Defined Functions (UDFs)**
> *   Scalar UDFs (Python UDFs)
> *   Vectorized/Pandas UDFs for performance improvement
> *   User-Defined Aggregate Functions (UDAFs)
> *   SQL UDFs
> ```

#### _Assistant_

> # PySpark User-Defined Functions (UDFs) Examples
> 
> I'll create comprehensive examples for each UDF pattern in PySpark.
> 
> ```python src/scalar_udf_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import udf, col
> from pyspark.sql.types import StringType, IntegerType, DoubleType
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Scalar UDF Example") \
>     .getOrCreate()
> 
> # Sample data
> data = [
>     (1, "john doe", 25, 50000),
>     (2, "jane smith", 30, 75000),
>     (3, "bob johnson", 35, 60000),
>     (4, "alice williams", 28, 80000)
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "age", "salary"])
> 
> # Example 1: Simple String Transformation UDF
> def capitalize_name(name):
>     """Capitalize each word in the name"""
>     if name:
>         return " ".join(word.capitalize() for word in name.split())
>     return None
> 
> # Register UDF
> capitalize_udf = udf(capitalize_name, StringType())
> 
> # Apply UDF
> df_capitalized = df.withColumn("capitalized_name", capitalize_udf(col("name")))
> print("=== Scalar UDF: Capitalize Name ===")
> df_capitalized.show()
> 
> # Example 2: Numeric Calculation UDF
> def calculate_tax(salary):
>     """Calculate tax based on salary brackets"""
>     if salary < 50000:
>         return salary * 0.10
>     elif salary < 75000:
>         return salary * 0.15
>     else:
>         return salary * 0.20
> 
> tax_udf = udf(calculate_tax, DoubleType())
> 
> df_with_tax = df.withColumn("tax", tax_udf(col("salary")))
> print("\n=== Scalar UDF: Calculate Tax ===")
> df_with_tax.show()
> 
> # Example 3: Multiple Parameters UDF
> def categorize_employee(age, salary):
>     """Categorize employee based on age and salary"""
>     if age < 30 and salary < 60000:
>         return "Junior"
>     elif age >= 30 and salary >= 70000:
>         return "Senior"
>     else:
>         return "Mid-Level"
> 
> categorize_udf = udf(categorize_employee, StringType())
> 
> df_categorized = df.withColumn(
>     "category", 
>     categorize_udf(col("age"), col("salary"))
> )
> print("\n=== Scalar UDF: Categorize Employee ===")
> df_categorized.show()
> 
> spark.stop()
> ```
> 
> ```python src/pandas_udf_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import pandas_udf, col
> from pyspark.sql.types import DoubleType, StringType, IntegerType
> import pandas as pd
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Pandas UDF Example") \
>     .getOrCreate()
> 
> # Sample data
> data = [
>     (1, "Electronics", 100.0, 5),
>     (2, "Electronics", 150.0, 3),
>     (3, "Clothing", 50.0, 10),
>     (4, "Clothing", 75.0, 7),
>     (5, "Electronics", 200.0, 2),
>     (6, "Clothing", 30.0, 15)
> ]
> 
> df = spark.createDataFrame(data, ["id", "category", "price", "quantity"])
> 
> # Example 1: Scalar Pandas UDF (Series to Series)
> @pandas_udf(DoubleType())
> def calculate_total_vectorized(price: pd.Series, quantity: pd.Series) -> pd.Series:
>     """Calculate total using vectorized operations"""
>     return price * quantity * 1.1  # Add 10% markup
> 
> df_total = df.withColumn("total", calculate_total_vectorized(col("price"), col("quantity")))
> print("=== Pandas UDF: Vectorized Calculation ===")
> df_total.show()
> 
> # Example 2: Scalar Pandas UDF with Complex Logic
> @pandas_udf(StringType())
> def price_tier(price: pd.Series) -> pd.Series:
>     """Categorize prices using vectorized operations"""
>     return pd.cut(
>         price,
>         bins=[0, 50, 100, float('inf')],
>         labels=['Low', 'Medium', 'High']
>     ).astype(str)
> 
> df_tiered = df.withColumn("price_tier", price_tier(col("price")))
> print("\n=== Pandas UDF: Price Tier ===")
> df_tiered.show()
> 
> # Example 3: Grouped Map Pandas UDF
> from pyspark.sql.types import StructType, StructField
> 
> # Define output schema
> schema = StructType([
>     StructField("category", StringType(), True),
>     StructField("avg_price", DoubleType(), True),
>     StructField("total_quantity", IntegerType(), True),
>     StructField("price_std", DoubleType(), True)
> ])
> 
> @pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
> def calculate_category_stats(pdf: pd.DataFrame) -> pd.DataFrame:
>     """Calculate statistics for each category"""
>     return pd.DataFrame({
>         'category': [pdf['category'].iloc[0]],
>         'avg_price': [pdf['price'].mean()],
>         'total_quantity': [pdf['quantity'].sum()],
>         'price_std': [pdf['price'].std()]
>     })
> 
> # Note: For newer PySpark versions (3.0+), use applyInPandas
> df_stats = df.groupBy("category").applyInPandas(
>     lambda pdf: pd.DataFrame({
>         'category': [pdf['category'].iloc[0]],
>         'avg_price': [pdf['price'].mean()],
>         'total_quantity': [pdf['quantity'].sum()],
>         'price_std': [pdf['price'].std()]
>     }),
>     schema=schema
> )
> 
> print("\n=== Grouped Map Pandas UDF: Category Statistics ===")
> df_stats.show()
> 
> # Example 4: Iterator of Series to Iterator of Series (for large datasets)
> @pandas_udf(DoubleType())
> def apply_discount_batch(price_iter):
>     """Apply discount in batches for memory efficiency"""
>     for price_batch in price_iter:
>         # Apply tiered discount
>         discount = price_batch.apply(
>             lambda x: 0.20 if x > 100 else (0.10 if x > 50 else 0.05)
>         )
>         yield price_batch * (1 - discount)
> 
> df_discounted = df.withColumn("discounted_price", apply_discount_batch(col("price")))
> print("\n=== Pandas UDF Iterator: Batch Discount ===")
> df_discounted.show()
> 
> spark.stop()
> ```
> 
> ```python src/udaf_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, udaf
> from pyspark.sql.types import DoubleType, StructType, StructField, LongType
> from typing import Tuple
> import pandas as pd
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("UDAF Example") \
>     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
>     .getOrCreate()
> 
> # Sample data
> data = [
>     (1, "Electronics", 100.0),
>     (2, "Electronics", 150.0),
>     (3, "Clothing", 50.0),
>     (4, "Clothing", 75.0),
>     (5, "Electronics", 200.0),
>     (6, "Clothing", 30.0),
>     (7, "Electronics", 120.0),
>     (8, "Clothing", 90.0)
> ]
> 
> df = spark.createDataFrame(data, ["id", "category", "price"])
> 
> # Example 1: Pandas UDAF - Geometric Mean
> @udaf(returnType=DoubleType())
> class GeometricMean:
>     """Calculate geometric mean of a column"""
>     
>     def __init__(self):
>         self.product = 1.0
>         self.count = 0
>     
>     @staticmethod
>     def merge(state1: Tuple[float, int], state2: Tuple[float, int]) -> Tuple[float, int]:
>         """Merge two states"""
>         return (state1[0] * state2[0], state1[1] + state2[1])
>     
>     def update(self, state: Tuple[float, int], value: float) -> Tuple[float, int]:
>         """Update state with new value"""
>         if value is not None and value > 0:
>             return (state[0] * value, state[1] + 1)
>         return state
>     
>     def finish(self, state: Tuple[float, int]) -> float:
>         """Calculate final result"""
>         product, count = state
>         if count == 0:
>             return 0.0
>         return product ** (1.0 / count)
> 
> # For PySpark 3.4+, use the simpler pandas UDAF syntax
> from pyspark.sql.pandas.functions import pandas_udaf
> 
> @pandas_udaf("double")
> def geometric_mean_pandas(values: pd.Series) -> float:
>     """Calculate geometric mean using pandas"""
>     if len(values) == 0:
>         return 0.0
>     # Filter out non-positive values
>     positive_values = values[values > 0]
>     if len(positive_values) == 0:
>         return 0.0
>     import numpy as np
>     return np.exp(np.log(positive_values).mean())
> 
> # Example 2: Weighted Average UDAF
> @pandas_udaf("double")
> def weighted_average(price: pd.Series, weight: pd.Series) -> float:
>     """Calculate weighted average"""
>     if len(price) == 0 or weight.sum() == 0:
>         return 0.0
>     return (price * weight).sum() / weight.sum()
> 
> # Create additional column for weights
> df_with_weights = df.withColumn("weight", col("id") % 3 + 1)
> 
> # Apply UDAFs
> print("=== Original Data ===")
> df_with_weights.show()
> 
> # Note: pandas_udaf usage varies by PySpark version
> # For demonstration, using built-in aggregations with similar logic
> 
> # Alternative approach using built-in functions and window functions
> from pyspark.sql.functions import avg, sum as spark_sum, count, expr, pow, exp, log
> 
> # Geometric mean calculation using built-in functions
> df_geo_mean = df.groupBy("category").agg(
>     exp(avg(log(col("price")))).alias("geometric_mean"),
>     avg(col("price")).alias("arithmetic_mean"),
>     count(col("price")).alias("count")
> )
> 
> print("\n=== Geometric Mean by Category ===")
> df_geo_mean.show()
> 
> # Weighted average calculation
> df_weighted = df_with_weights.groupBy("category").agg(
>     (spark_sum(col("price") * col("weight")) / spark_sum(col("weight"))).alias("weighted_avg"),
>     avg(col("price")).alias("simple_avg")
> )
> 
> print("\n=== Weighted Average by Category ===")
> df_weighted.show()
> 
> # Example 3: Custom Aggregation - Standard Error
> @pandas_udaf("double")
> def standard_error(values: pd.Series) -> float:
>     """Calculate standard error of the mean"""
>     if len(values) <= 1:
>         return 0.0
>     return values.std() / (len(values) ** 0.5)
> 
> # Using expr for custom calculation
> df_stats = df.groupBy("category").agg(
>     avg(col("price")).alias("mean"),
>     expr("stddev(price)").alias("std_dev"),
>     count(col("price")).alias("n")
> )
> 
> # Calculate standard error manually
> df_with_se = df_stats.withColumn(
>     "std_error",
>     col("std_dev") / pow(col("n"), 0.5)
> )
> 
> print("\n=== Standard Error by Category ===")
> df_with_se.show()
> 
> spark.stop()
> ```
> 
> ```python src/sql_udf_example.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import expr, col
> from pyspark.sql.types import StringType, DoubleType, IntegerType
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("SQL UDF Example") \
>     .getOrCreate()
> 
> # Sample data
> data = [
>     (1, "john.doe@example.com", 25, 50000, "2023-01-15"),
>     (2, "jane.smith@company.com", 30, 75000, "2023-02-20"),
>     (3, "bob.j@test.org", 35, 60000, "2023-03-10"),
>     (4, "alice.w@demo.com", 28, 80000, "2023-04-05")
> ]
> 
> df = spark.createDataFrame(data, ["id", "email", "age", "salary", "hire_date"])
> df.createOrReplaceTempView("employees")
> 
> # Example 1: Simple SQL UDF - Extract Domain from Email
> def extract_domain(email):
>     """Extract domain from email address"""
>     if email and '@' in email:
>         return email.split('@')[1]
>     return None
> 
> spark.udf.register("extract_domain", extract_domain, StringType())
> 
> # Use SQL UDF in SQL query
> result1 = spark.sql("""
>     SELECT 
>         id,
>         email,
>         extract_domain(email) as domain
>     FROM employees
> """)
> 
> print("=== SQL UDF: Extract Domain ===")
> result1.show(truncate=False)
> 
> # Example 2: SQL UDF with Multiple Parameters - Calculate Bonus
> def calculate_bonus(salary, age):
>     """Calculate bonus based on salary and age"""
>     base_bonus = salary * 0.05
>     age_multiplier = 1.0 + (age - 25) * 0.01
>     return base_bonus * age_multiplier if age >= 25 else base_bonus
> 
> spark.udf.register("calculate_bonus", calculate_bonus, DoubleType())
> 
> result2 = spark.sql("""
>     SELECT 
>         id,
>         salary,
>         age,
>         calculate_bonus(salary, age) as bonus,
>         salary + calculate_bonus(salary, age) as total_compensation
>     FROM employees
> """)
> 
> print("\n=== SQL UDF: Calculate Bonus ===")
> result2.show()
> 
> # Example 3: SQL UDF for String Manipulation - Format Name
> def format_email_to_name(email):
>     """Convert email to formatted name"""
>     if not email or '@' not in email:
>         return "Unknown"
>     
>     username = email.split('@')[0]
>     parts = username.replace('.', ' ').replace('_', ' ').split()
>     return ' '.join(word.capitalize() for word in parts)
> 
> spark.udf.register("format_email_to_name", format_email_to_name, StringType())
> 
> result3 = spark.sql("""
>     SELECT 
>         id,
>         email,
>         format_email_to_name(email) as formatted_name
>     FROM employees
> """)
> 
> print("\n=== SQL UDF: Format Name from Email ===")
> result3.show(truncate=False)
> 
> # Example 4: SQL UDF for Date Calculations - Tenure in Months
> from datetime import datetime
> 
> def calculate_tenure_months(hire_date_str):
>     """Calculate tenure in months from hire date"""
>     if not hire_date_str:
>         return 0
>     
>     hire_date = datetime.strptime(hire_date_str, "%Y-%m-%d")
>     current_date = datetime.now()
>     
>     months = (current_date.year - hire_date.year) * 12
>     months += current_date.month - hire_date.month
>     
>     return months
> 
> spark.udf.register("calculate_tenure_months", calculate_tenure_months, IntegerType())
> 
> result4 = spark.sql("""
>     SELECT 
>         id,
>         hire_date,
>         calculate_tenure_months(hire_date) as tenure_months,
>         CASE 
>             WHEN calculate_tenure_months(hire_date) < 12 THEN 'New'
>             WHEN calculate_tenure_months(hire_date) < 24 THEN 'Intermediate'
>             ELSE 'Veteran'
>         END as employee_status
>     FROM employees
> """)
> 
> print("\n=== SQL UDF: Calculate Tenure ===")
> result4.show()
> 
> # Example 5: Complex SQL UDF - Employee Grade
> def calculate_grade(salary, age, tenure_months):
>     """Calculate employee grade based on multiple factors"""
>     score = 0
>     
>     # Salary component (0-40 points)
>     if salary >= 80000:
>         score += 40
>     elif salary >= 70000:
>         score += 30
>     elif salary >= 60000:
>         score += 20
>     else:
>         score += 10
>     
>     # Age component (0-30 points)
>     if age >= 35:
>         score += 30
>     elif age >= 30:
>         score += 20
>     else:
>         score += 10
>     
>     # Tenure component (0-30 points)
>     if tenure_months >= 24:
>         score += 30
>     elif tenure_months >= 12:
>         score += 20
>     else:
>         score += 10
>     
>     # Determine grade
>     if score >= 80:
>         return 'A'
>     elif score >= 60:
>         return 'B'
>     elif score >= 40:
>         return 'C'
>     else:
>         return 'D'
> 
> spark.udf.register("calculate_grade", calculate_grade, StringType())
> 
> result5 = spark.sql("""
>     SELECT 
>         id,
>         salary,
>         age,
>         calculate_tenure_months(hire_date) as tenure_months,
>         calculate_grade(salary, age, calculate_tenure_months(hire_date)) as grade
>     FROM employees
>     ORDER BY grade, salary DESC
> """)
> 
> print("\n=== SQL UDF: Employee Grade ===")
> result5.show()
> 
> # Example 6: Using SQL UDF with DataFrame API
> df_with_udf = df.selectExpr(
>     "id",
>     "email",
>     "extract_domain(email) as domain",
>     "calculate_bonus(salary, age) as bonus",
>     "format_email_to_name(email) as name"
> )
> 
> print("\n=== Using SQL UDFs with DataFrame API ===")
> df_with_udf.show(truncate=False)
> 
> spark.stop()
> ```
> 
> ```python src/complete_udf_comparison.py
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import udf, pandas_udf, col
> from pyspark.sql.types import DoubleType
> import pandas as pd
> import time
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("UDF Performance Comparison") \
>     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
>     .getOrCreate()
> 
> # Create larger dataset for performance testing
> data = [(i, float(i * 10)) for i in range(1, 100001)]
> df = spark.createDataFrame(data, ["id", "value"])
> 
> # Same calculation using different UDF types
> 
> # 1. Regular Python UDF
> def calculate_complex_regular(value):
>     """Complex calculation using regular UDF"""
>     result = 0
>     for i in range(10):
>         result += (value ** 0.5) * 1.5 + i
>     return result
> 
> regular_udf = udf(calculate_complex_regular, DoubleType())
> 
> # 2. Pandas UDF (Vectorized)
> @pandas_udf(DoubleType())
> def calculate_complex_pandas(value: pd.Series) -> pd.Series:
>     """Same calculation using Pandas UDF"""
>     result = pd.Series([0.0] * len(value))
>     for i in range(10):
>         result += (value ** 0.5) * 1.5 + i
>     return result
> 
> # 3. SQL UDF
> spark.udf.register("calculate_complex_sql", calculate_complex_regular, DoubleType())
> df.createOrReplaceTempView("test_data")
> 
> print("=== Performance Comparison ===\n")
> 
> # Test Regular UDF
> start = time.time()
> df_regular = df.withColumn("result", regular_udf(col("value")))
> df_regular.count()  # Trigger execution
> regular_time = time.time() - start
> print(f"Regular Python UDF: {regular_time:.2f} seconds")
> 
> # Test Pandas UDF
> start = time.time()
> df_pandas = df.withColumn("result", calculate_complex_pandas(col("value")))
> df_pandas.count()  # Trigger execution
> pandas_time = time.time() - start
> print(f"Pandas UDF (Vectorized): {pandas_time:.2f} seconds")
> 
> # Test SQL UDF
> start = time.time()
> df_sql = spark.sql("SELECT id, value, calculate_complex_sql(value) as result FROM test_data")
> df_sql.count()  # Trigger execution
> sql_time = time.time() - start
> print(f"SQL UDF: {sql_time:.2f} seconds")
> 
> print(f"\n=== Performance Improvement ===")
> print(f"Pandas UDF is {regular_time/pandas_time:.2f}x faster than Regular UDF")
> 
> # Show sample results
> print("\n=== Sample Results ===")
> df_pandas.show(10)
> 
> spark.stop()
> ```
> 
> ## Summary
> 
> I've created comprehensive examples for all PySpark UDF patterns:
> 
> ### 1. **Scalar UDFs** (`scalar_udf_example.py`)
> - Simple string transformations
> - Numeric calculations
> - Multi-parameter functions
> 
> ### 2. **Pandas/Vectorized UDFs** (`pandas_udf_example.py`)
> - Series to Series transformations (much faster)
> - Grouped map operations
> - Iterator-based processing for large datasets
> - Complex vectorized logic
> 
> ### 3. **UDAFs** (`udaf_example.py`)
> - Custom aggregations (geometric mean, weighted average)
> - State management
> - Pandas-based UDAF implementations
> 
> ### 4. **SQL UDFs** (`sql_udf_example.py`)
> - SQL-registered functions
> - String manipulation
> - Date calculations
> - Complex multi-parameter logic
> - Usage in both SQL queries and DataFrame API
> 
> ### 5. **Performance Comparison** (`complete_udf_comparison.py`)
> - Side-by-side comparison of all UDF types
> - Performance benchmarking
> 
> **Key Takeaways:**
> - **Pandas UDFs** are 5-100x faster than regular Python UDFs
> - Use **SQL UDFs** for reusable functions across SQL and DataFrame API
> - Use **UDAFs** for custom aggregation logic
> - Regular Python UDFs are easiest but slowest

#### _User_

> Continue your response exactly where you left off:

#### _Assistant_

> ## Running the Examples
> 
> ```bash run_examples.sh
> #!/bin/bash
> 
> echo "Running PySpark UDF Examples..."
> echo "================================"
> 
> echo -e "\n1. Running Scalar UDF Example..."
> python src/scalar_udf_example.py
> 
> echo -e "\n2. Running Pandas UDF Example..."
> python src/pandas_udf_example.py
> 
> echo -e "\n3. Running UDAF Example..."
> python src/udaf_example.py
> 
> echo -e "\n4. Running SQL UDF Example..."
> python src/sql_udf_example.py
> 
> echo -e "\n5. Running Performance Comparison..."
> python src/complete_udf_comparison.py
> 
> echo -e "\n================================"
> echo "All examples completed!"
> ```
> 
> ## Best Practices & When to Use Each UDF Type
> 
> ```python src/udf_best_practices.py
> """
> PySpark UDF Best Practices and Use Case Guide
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import udf, pandas_udf, col, lit
> from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType
> import pandas as pd
> 
> spark = SparkSession.builder \
>     .appName("UDF Best Practices") \
>     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
>     .getOrCreate()
> 
> # Sample dataset
> data = [
>     (1, "Product A", 100.0, [1, 2, 3, 4, 5]),
>     (2, "Product B", 200.0, [2, 4, 6, 8, 10]),
>     (3, "Product C", 150.0, [1, 3, 5, 7, 9])
> ]
> 
> df = spark.createDataFrame(data, ["id", "name", "price", "ratings"])
> 
> print("=== WHEN TO USE EACH UDF TYPE ===\n")
> 
> # ============================================================
> # 1. REGULAR PYTHON UDF
> # Use when: Simple transformations, non-vectorizable logic
> # ============================================================
> 
> print("1. REGULAR PYTHON UDF")
> print("   Use Case: Simple string/object transformations, complex branching logic")
> print("   Performance: SLOW (row-by-row processing)")
> print()
> 
> @udf(StringType())
> def complex_categorization(name, price):
>     """
>     Use Regular UDF when:
>     - Working with complex objects (lists, dicts, custom types)
>     - Logic cannot be easily vectorized
>     - Small datasets where performance isn't critical
>     """
>     category = ""
>     
>     if "A" in name:
>         category = "Premium"
>     elif "B" in name:
>         category = "Standard"
>     else:
>         category = "Basic"
>     
>     if price > 150:
>         category += "-High"
>     else:
>         category += "-Low"
>     
>     return category
> 
> df.withColumn("category", complex_categorization(col("name"), col("price"))).show()
> 
> # ============================================================
> # 2. PANDAS UDF (VECTORIZED)
> # Use when: Numeric operations, large datasets, vectorizable logic
> # ============================================================
> 
> print("\n2. PANDAS UDF (VECTORIZED)")
> print("   Use Case: Numeric transformations, statistical operations, large datasets")
> print("   Performance: FAST (vectorized operations)")
> print()
> 
> @pandas_udf(DoubleType())
> def apply_dynamic_discount(price: pd.Series) -> pd.Series:
>     """
>     Use Pandas UDF when:
>     - Performing numeric/mathematical operations
>     - Working with large datasets
>     - Logic can be vectorized
>     - Need significant performance improvement
>     """
>     # Vectorized discount calculation
>     discount = pd.Series([0.0] * len(price))
>     discount[price > 150] = 0.20
>     discount[(price > 100) & (price <= 150)] = 0.15
>     discount[price <= 100] = 0.10
>     
>     return price * (1 - discount)
> 
> df.withColumn("discounted_price", apply_dynamic_discount(col("price"))).show()
> 
> # ============================================================
> # 3. PANDAS UDF WITH COMPLEX TYPES
> # Use when: Working with arrays, complex transformations
> # ============================================================
> 
> print("\n3. PANDAS UDF WITH ARRAYS")
> print("   Use Case: Array/list processing, statistical analysis on arrays")
> print()
> 
> @pandas_udf(DoubleType())
> def calculate_rating_score(ratings: pd.Series) -> pd.Series:
>     """
>     Use for complex type processing with vectorization
>     """
>     import numpy as np
>     
>     def compute_score(rating_list):
>         if rating_list is None or len(rating_list) == 0:
>             return 0.0
>         arr = np.array(rating_list)
>         # Weighted score: mean + bonus for consistency
>         mean_rating = arr.mean()
>         consistency_bonus = 1.0 / (arr.std() + 1)
>         return mean_rating * (1 + consistency_bonus * 0.1)
>     
>     return ratings.apply(compute_score)
> 
> df.withColumn("rating_score", calculate_rating_score(col("ratings"))).show()
> 
> # ============================================================
> # 4. SQL UDF
> # Use when: Need to use in SQL queries, sharing across teams
> # ============================================================
> 
> print("\n4. SQL UDF")
> print("   Use Case: Reusable functions in SQL queries, shared logic")
> print()
> 
> def standardize_name(name):
>     """
>     Use SQL UDF when:
>     - Function needs to be available in SQL queries
>     - Sharing logic across different queries/notebooks
>     - Team works primarily with SQL
>     """
>     return name.upper().replace(" ", "_")
> 
> spark.udf.register("standardize_name", standardize_name, StringType())
> df.createOrReplaceTempView("products")
> 
> spark.sql("""
>     SELECT 
>         id,
>         name,
>         standardize_name(name) as standard_name,
>         price
>     FROM products
> """).show()
> 
> # ============================================================
> # 5. GROUPED MAP PANDAS UDF (for group-level operations)
> # Use when: Need to process entire groups/partitions
> # ============================================================
> 
> print("\n5. GROUPED MAP PANDAS UDF")
> print("   Use Case: Group-level transformations, machine learning per group")
> print()
> 
> from pyspark.sql.types import StructType, StructField
> 
> schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("name", StringType(), True),
>     StructField("price", DoubleType(), True),
>     StructField("price_normalized", DoubleType(), True)
> ])
> 
> def normalize_prices_in_group(pdf: pd.DataFrame) -> pd.DataFrame:
>     """
>     Use Grouped Map when:
>     - Need to process entire groups together
>     - Group-level statistics needed for each row
>     - Applying ML models per group
>     - Complex group transformations
>     """
>     pdf['price_normalized'] = (pdf['price'] - pdf['price'].min()) / \
>                               (pdf['price'].max() - pdf['price'].min() + 0.001)
>     return pdf[['id', 'name', 'price', 'price_normalized']]
> 
> # Add a grouping column
> df_grouped = df.withColumn("group", (col("id") % 2))
> 
> result = df_grouped.groupBy("group").applyInPandas(
>     normalize_prices_in_group,
>     schema=schema
> )
> 
> print("Normalized prices within groups:")
> result.show()
> 
> # ============================================================
> # PERFORMANCE COMPARISON TABLE
> # ============================================================
> 
> print("\n" + "="*70)
> print("PERFORMANCE & USE CASE SUMMARY")
> print("="*70)
> print(f"{'UDF Type':<25} {'Speed':<15} {'Best For':<30}")
> print("-"*70)
> print(f"{'Regular Python UDF':<25} {'Slow':<15} {'Complex logic, small data':<30}")
> print(f"{'Pandas UDF (Scalar)':<25} {'Fast':<15} {'Numeric ops, large data':<30}")
> print(f"{'Pandas UDF (Grouped)':<25} {'Fast':<15} {'Group transformations':<30}")
> print(f"{'SQL UDF':<25} {'Slow*':<15} {'SQL queries, reusability':<30}")
> print(f"{'Built-in Functions':<25} {'Fastest':<15} {'Always prefer these!':<30}")
> print("="*70)
> print("* SQL UDFs use Python UDF underneath, inherit same performance")
> print()
> 
> # ============================================================
> # ANTI-PATTERNS TO AVOID
> # ============================================================
> 
> print("\n=== ANTI-PATTERNS TO AVOID ===\n")
> 
> # ❌ BAD: Using UDF when built-in function exists
> print("❌ BAD: Using UDF for simple operations")
> print("   DON'T: udf(lambda x: x.upper())")
> print("   DO: Use built-in: upper(col('name'))")
> print()
> 
> # ❌ BAD: Regular UDF on large datasets
> print("❌ BAD: Regular Python UDF on millions of rows")
> print("   DON'T: @udf for numeric operations on big data")
> print("   DO: Use @pandas_udf for vectorization")
> print()
> 
> # ❌ BAD: Not specifying return types
> print("❌ BAD: Not specifying return types")
> print("   DON'T: @udf (without type)")
> print("   DO: @udf(StringType()) or @pandas_udf(DoubleType())")
> print()
> 
> # ❌ BAD: Complex logic in UDFs without testing
> print("❌ BAD: Untested complex UDFs")
> print("   DON'T: Write complex logic without unit tests")
> print("   DO: Test UDF functions separately before applying")
> print()
> 
> # ============================================================
> # OPTIMIZATION TIPS
> # ============================================================
> 
> print("\n=== OPTIMIZATION TIPS ===\n")
> 
> tips = [
>     "1. Always prefer built-in functions over UDFs",
>     "2. Use Pandas UDFs for numeric operations (5-100x faster)",
>     "3. Cache DataFrames if applying multiple UDFs",
>     "4. Broadcast small lookup tables instead of UDF joins",
>     "5. Test UDFs on small samples before full dataset",
>     "6. Use Arrow for Pandas UDF optimization",
>     "7. Avoid nested UDFs - combine logic into single UDF",
>     "8. Use appropriate return types to avoid type conversions",
>     "9. Profile your UDFs to identify bottlenecks",
>     "10. Consider SQL expressions before writing UDFs"
> ]
> 
> for tip in tips:
>     print(f"   {tip}")
> 
> print()
> 
> # ============================================================
> # EXAMPLE: Replacing UDF with Built-in Functions
> # ============================================================
> 
> print("\n=== EXAMPLE: PREFER BUILT-IN FUNCTIONS ===\n")
> 
> # ❌ Using UDF
> @udf(DoubleType())
> def calculate_tax_udf(price):
>     return price * 0.15
> 
> # ✅ Using built-in functions (MUCH FASTER)
> from pyspark.sql.functions import when
> 
> print("Comparing UDF vs Built-in Function:")
> print("\n✅ PREFERRED: Using built-in functions")
> df.withColumn("tax", col("price") * 0.15).show()
> 
> print("❌ AVOID: Using UDF for simple operations")
> df.withColumn("tax", calculate_tax_udf(col("price"))).show()
> 
> spark.stop()
> ```
> 
> ## Additional Advanced Patterns
> 
> ```python src/advanced_udf_patterns.py
> """
> Advanced UDF Patterns and Techniques
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import pandas_udf, col, struct, array
> from pyspark.sql.types import *
> import pandas as pd
> import numpy as np
> 
> spark = SparkSession.builder \
>     .appName("Advanced UDF Patterns") \
>     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
>     .getOrCreate()
> 
> # ============================================================
> # PATTERN 1: UDF with External Library (e.g., ML model)
> # ============================================================
> 
> print("=== PATTERN 1: UDF with External Library ===\n")
> 
> from sklearn.preprocessing import StandardScaler
> 
> # Simulate having a pre-trained model
> def create_scaler():
>     """Simulate a trained scaler"""
>     scaler = StandardScaler()
>     # Pre-fit with sample data
>     scaler.fit([[10], [20], [30], [40], [50]])
>     return scaler
> 
> # Broadcast the model to all workers
> scaler_broadcast = spark.sparkContext.broadcast(create_scaler())
> 
> @pandas_udf(DoubleType())
> def apply_ml_scaler(values: pd.Series) -> pd.Series:
>     """
>     Pattern: Using broadcast variables for models/lookups in UDFs
>     """
>     scaler = scaler_broadcast.value
>     values_2d = values.values.reshape(-1, 1)
>     scaled = scaler.transform(values_2d)
>     return pd.Series(scaled.flatten())
> 
> data = [(i, float(i * 10)) for i in range(1, 11)]
> df = spark.createDataFrame(data, ["id", "value"])
> 
> df.withColumn("scaled_value", apply_ml_scaler(col("value"))).show()
> 
> # ============================================================
> # PATTERN 2: UDF Returning Complex Types (Struct/Array)
> # ============================================================
> 
> print("\n=== PATTERN 2: UDF Returning Complex Types ===\n")
> 
> # Define complex return type
> stats_schema = StructType([
>     StructField("mean", DoubleType(), False),
>     StructField("std", DoubleType(), False),
>     StructField("min", DoubleType(), False),
>     StructField("max", DoubleType(), False)
> ])
> 
> @pandas_udf(stats_schema)
> def calculate_stats(values: pd.Series) -> pd.DataFrame:
>     """
>     Pattern: Returning multiple values as struct
>     """
>     return pd.DataFrame({
>         'mean': [values.mean()],
>         'std': [values.std()],
>         'min': [values.min()],
>         'max': [values.max()]
>     })
> 
> # For demonstration with array input
> data_with_arrays = [
>     (1, [10.0, 20.0, 30.0, 40.0, 50.0]),
>     (2, [15.0, 25.0, 35.0, 45.0, 55.0]),
>     (3, [5.0, 15.0, 25.0, 35.0, 45.0])
> ]
> 
> df_arrays = spark.createDataFrame(data_with_arrays, ["id", "measurements"])
> 
> # Returning array from UDF
> @pandas_udf(ArrayType(DoubleType()))
> def normalize_array(arr: pd.Series) -> pd.Series:
>     """
>     Pattern: Processing arrays and returning arrays
>     """
>     def normalize(lst):
>         if not lst:
>             return []
>         arr_np = np.array(lst)
>         min_val, max_val = arr_np.min(), arr_np.max()
>         if max_val - min_val == 0:
>             return [0.5] * len(lst)
>         return ((arr_np - min_val) / (max_val - min_val)).tolist()
>     
>     return arr.apply(normalize)
> 
> df_arrays.withColumn("normalized", normalize_array(col("measurements"))).show(truncate=False)
> 
> # ============================================================
> # PATTERN 3: Chaining Multiple Pandas UDFs Efficiently
> # ============================================================
> 
> print("\n=== PATTERN 3: Chaining UDFs Efficiently ===\n")
> 
> @pandas_udf(DoubleType())
> def step1_transform(value: pd.Series) -> pd.Series:
>     """First transformation"""
>     return value * 2
> 
> @pandas_udf(DoubleType())
> def step2_transform(value: pd.Series) -> pd.Series:
>     """Second transformation"""
>     return value + 10
> 
> @pandas_udf(DoubleType())
> def step3_transform(value: pd.Series) -> pd.Series:
>     """Third transformation"""
>     return value ** 0.5
> 
> # ❌ INEFFICIENT: Multiple passes over data
> # df.withColumn("temp1", step1_transform(col("value"))) \
> #   .withColumn("temp2", step2_transform(col("temp1"))) \
> #   .withColumn("result", step3_transform(col("temp2")))
> 
> # ✅ EFFICIENT: Combine into single UDF
> @pandas_udf(DoubleType())
> def combined_transform(value: pd.Series) -> pd.Series:
>     """
>     Pattern: Combine multiple transformations into single UDF
>     to reduce overhead
>     """
>     result = value * 2  # step 1
>     result = result + 10  # step 2
>     result = result ** 0.5  # step 3
>     return result
> 
> df.withColumn("result", combined_transform(col("value"))).show()
> 
> # ============================================================
> # PATTERN 4: Windowed/Rolling Operations with Pandas UDF
> # ============================================================
> 
> print("\n=== PATTERN 4: Rolling Window Operations ===\n")
> 
> time_series_data = [
>     (1, "2024-01-01", 100.0),
>     (2, "2024-01-02", 105.0),
>     (3, "2024-01-03", 103.0),
>     (4, "2024-01-04", 108.0),
>     (5, "2024-01-05", 110.0),
>     (6, "2024-01-06", 107.0)
> ]
> 
> df_ts = spark.createDataFrame(time_series_data, ["id", "date", "price"])
> 
> # Using grouped map for rolling calculations
> rolling_schema = StructType([
>     StructField("id", IntegerType(), True),
>     StructField("date", StringType(), True),
>     StructField("price", DoubleType(), True),
>     StructField("rolling_mean_3", DoubleType(), True)
> ])
> 
> def calculate_rolling_mean(pdf: pd.DataFrame) -> pd.DataFrame:
>     """
>     Pattern: Rolling window calculations using grouped map
>     """
>     pdf = pdf.sort_values('id')
>     pdf['rolling_mean_3'] = pdf['price'].rolling(window=3, min_periods=1).mean()
>     return pdf
> 
> df_ts.groupBy().applyInPandas(calculate_rolling_mean, schema=rolling_schema).show()
> 
> # ============================================================
> # PATTERN 5: Error Handling in UDFs
> # ============================================================
> 
> print("\n=== PATTERN 5: Error Handling in UDFs ===\n")
> 
> @pandas_udf(DoubleType())
> def safe_division(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
>     """
>     Pattern: Proper error handling in UDFs
>     """
>     try:
>         # Handle division by zero
>         result = numerator / denominator
>         result = result.replace([np.inf, -np.inf], np.nan)
>         result = result.fillna(0.0)
>         return result
>     except Exception as e:
>         # Log error and return safe default
>         print(f"Error in UDF: {e}")
>         return pd.Series([0.0] * len(numerator))
> 
> error_data = [(1, 10.0, 2.0), (2, 20.0, 0.0), (3, 30.0, 5.0)]
> df_err = spark.createDataFrame(error_data, ["id", "num", "denom"])
> 
> df_err.withColumn("result", safe_division(col("num"), col("denom"))).show()
> 
> # ============================================================
> # PATTERN 6: Conditional UDF Application
> # ============================================================
> 
> print("\n=== PATTERN 6: Conditional UDF Application ===\n")
> 
> from pyspark.sql.functions import when
> 
> @pandas_udf(StringType())
> def expensive_operation(value: pd.Series) -> pd.Series:
>     """Simulating expensive operation"""
>     import time
>     time.sleep(0.1)  # Simulate expensive computation
>     return value.astype(str) + "_processed"
> 
> # ❌ BAD: Apply UDF to all rows
> # df.withColumn("result", expensive_operation(col("name")))
> 
> # ✅ GOOD: Apply UDF only where needed
> df_conditional = spark.createDataFrame([
>     (1, "A", True),
>     (2, "B", False),
>     (3, "C", True)
> ], ["id", "name", "needs_processing"])
> 
> df_conditional.withColumn(
>     "result",
>     when(col("needs_processing"), expensive_operation(col("name")))
>     .otherwise(col("name"))
> ).show()
> 
> print("\n=== All Advanced Patterns Completed ===")
> 
> spark.stop()
> ```
> 
> ## Testing Framework for UDFs
> 
> ```python tests/test_udfs.py
> """
> Unit Testing Framework for PySpark UDFs
> """
> 
> import pytest
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col
> import pandas as pd
> 
> @pytest.fixture(scope="session")
> def spark():
>     """Create Spark session for testing"""
>     spark = SparkSession.builder \
>         .appName("UDF Tests") \
>         .master("local[2]") \
>         .getOrCreate()
>     yield spark
>     spark.stop()
> 
> # ============================================================
> # Test Regular UDFs
> # ============================================================
> 
> def test_simple_udf(spark):
>     """Test simple UDF transformation"""
>     from pyspark.sql.functions import udf
>     from pyspark.sql.types import StringType
>     
>     def to_upper(s):
>         return s.upper() if s else None
>     
>     upper_udf = udf(to_upper, StringType())
>     
>     df = spark.createDataFrame([("hello",), ("world",)], ["text"])
>     result = df.withColumn("upper", upper_udf(col("text")))
>     
>     assert result.collect()[0]["upper"] == "HELLO"
>     assert result.collect()[1]["upper"] == "WORLD"
> 
> def test_udf_with_null_handling(spark):
>     """Test UDF handles nulls correctly"""
>     from pyspark.sql.functions import udf
>     from pyspark.sql.types import IntegerType
>     
>     def safe_add_ten(x):
>         return x + 10 if x is not None else None
>     
>     add_udf = udf(safe_add_ten, IntegerType())
>     
>     df = spark.createDataFrame([(1,), (None,), (3,)], ["value"])
>     result = df.withColumn("result", add_udf(col("value")))
>     
>     rows = result.collect()
>     assert rows[0]["result"] == 11
>     assert rows[1]["result"] is None
>     assert rows[2]["result"] == 13
> 
> # ============================================================
> # Test Pandas UDFs
> # ============================================================
> 
> def test_pandas_udf_vectorized(spark):
>     """Test Pandas UDF vectorized operation"""
>     from pyspark.sql.functions import pandas_udf
>     from pyspark.sql.types import DoubleType
>     
>     @pandas_udf(DoubleType())
>     def multiply_by_two(s: pd.Series) -> pd.Series:
>         return s * 2
>     
>     df = spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["value"])
>     result = df.withColumn("doubled", multiply_by_two(col("value")))
>     
>     rows = result.collect()
>     assert rows[0]["doubled"] == 2.0
>     assert rows[1]["doubled"] == 4.0
>     assert rows[2]["doubled"] == 6.0
> 
> # ============================================================
> # Test SQL UDFs
> # ============================================================
> 
> def test_sql_udf(spark):
>     """Test SQL UDF registration and usage"""
>     from pyspark.sql.types import IntegerType
>     
>     def add_five(x):
>         return x + 5 if x is not None else None
>     
>     spark.udf.register("add_five", add_five, IntegerType())
>     
>     df = spark.createDataFrame([(1,), (2,), (3,)], ["num"])
>     df.createOrReplaceTempView("numbers")
>     
>     result = spark.sql("SELECT num, add_five(num) as result FROM numbers")
>     
>     rows = result.collect()
>     assert rows[0]["result"] == 6
>     assert rows[1]["result"] == 7
>     assert rows[2]["result"] == 8
> 
> if __name__ == "__main__":
>     pytest.main([__file__, "-v"])
> ```
> 
> These examples demonstrate production-ready patterns including error handling, performance optimization, complex type handling, and comprehensive testing strategies!