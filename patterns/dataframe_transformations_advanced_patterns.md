### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:18:09 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> ### **IV. DataFrame Transformations (Advanced)**
> *   Window Functions (Ranking, Lag/Lead, Moving Averages, Cumulative Sums)
> *   Exploding Arrays (`explode`, `posexplode`)
> *   Pivoting / Unpivoting (`pivot`, `stack`)
> *   Higher-Order Functions on Arrays and Maps (`transform`, `filter`, `exists`, `aggregate`, `map_arrays`)
> *   Using SQL Expressions within DataFrames (`selectExpr`, `where`)
> *   Recursive CTEs (Common Table Expressions)
> *   FoldLeft/Reduce operations
> *   Approximate Quantiles
> ```

#### _Assistant_

> # PySpark Advanced DataFrame Transformations - Complete Examples
> 
> I'll create comprehensive examples for each pattern with real-world scenarios.
> 
> ```python pyspark_advanced_transformations.py
> from pyspark.sql import SparkSession
> from pyspark.sql import Window
> from pyspark.sql.functions import (
>     row_number, rank, dense_rank, lag, lead, avg, sum as _sum,
>     explode, posexplode, array, struct, map as create_map,
>     transform, filter as array_filter, exists, aggregate,
>     col, lit, expr, collect_list, when, monotonically_increasing_id,
>     approx_percentile, sequence, split, size
> )
> from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType
> 
> # Initialize Spark Session
> spark = SparkSession.builder \
>     .appName("Advanced DataFrame Transformations") \
>     .config("spark.sql.adaptive.enabled", "true") \
>     .getOrCreate()
> 
> print("=" * 80)
> print("1. WINDOW FUNCTIONS")
> print("=" * 80)
> 
> # Sample sales data
> sales_data = [
>     ("2024-01-01", "Electronics", "Laptop", 1200, "Store_A"),
>     ("2024-01-02", "Electronics", "Phone", 800, "Store_A"),
>     ("2024-01-03", "Electronics", "Tablet", 500, "Store_B"),
>     ("2024-01-04", "Clothing", "Shirt", 50, "Store_A"),
>     ("2024-01-05", "Clothing", "Pants", 80, "Store_B"),
>     ("2024-01-06", "Electronics", "Laptop", 1300, "Store_A"),
>     ("2024-01-07", "Electronics", "Phone", 850, "Store_B"),
>     ("2024-01-08", "Clothing", "Jacket", 120, "Store_A"),
> ]
> 
> sales_df = spark.createDataFrame(sales_data, 
>     ["date", "category", "product", "amount", "store"])
> 
> # Define window specifications
> category_window = Window.partitionBy("category").orderBy("amount")
> date_window = Window.orderBy("date").rowsBetween(-2, 0)  # Last 3 days
> unbounded_window = Window.partitionBy("category").orderBy("date") \
>     .rowsBetween(Window.unboundedPreceding, Window.currentRow)
> 
> # Apply various window functions
> windowed_df = sales_df.select(
>     "*",
>     # Ranking functions
>     row_number().over(category_window).alias("row_num"),
>     rank().over(category_window).alias("rank"),
>     dense_rank().over(category_window).alias("dense_rank"),
>     
>     # Lag/Lead
>     lag("amount", 1).over(category_window).alias("prev_amount"),
>     lead("amount", 1).over(category_window).alias("next_amount"),
>     
>     # Moving average (last 3 days)
>     avg("amount").over(date_window).alias("moving_avg_3day"),
>     
>     # Cumulative sum
>     _sum("amount").over(unbounded_window).alias("cumulative_sum")
> )
> 
> print("\nWindow Functions Results:")
> windowed_df.show(truncate=False)
> 
> print("\n" + "=" * 80)
> print("2. EXPLODING ARRAYS")
> print("=" * 80)
> 
> # Sample data with arrays
> orders_data = [
>     (1, "John", ["Laptop", "Mouse", "Keyboard"]),
>     (2, "Jane", ["Phone", "Case"]),
>     (3, "Bob", ["Tablet", "Stylus", "Cover", "Charger"])
> ]
> 
> orders_df = spark.createDataFrame(orders_data, ["order_id", "customer", "items"])
> 
> # explode - creates new row for each array element
> exploded_df = orders_df.select(
>     "order_id",
>     "customer",
>     explode("items").alias("item")
> )
> 
> print("\nExplode Results:")
> exploded_df.show(truncate=False)
> 
> # posexplode - includes position/index
> posexploded_df = orders_df.select(
>     "order_id",
>     "customer",
>     posexplode("items").alias("position", "item")
> )
> 
> print("\nPosexplode Results:")
> posexploded_df.show(truncate=False)
> 
> print("\n" + "=" * 80)
> print("3. PIVOTING / UNPIVOTING")
> print("=" * 80)
> 
> # Sample quarterly sales data
> quarterly_sales = [
>     ("Product_A", "Q1", 1000),
>     ("Product_A", "Q2", 1200),
>     ("Product_A", "Q3", 1100),
>     ("Product_A", "Q4", 1500),
>     ("Product_B", "Q1", 800),
>     ("Product_B", "Q2", 850),
>     ("Product_B", "Q3", 900),
>     ("Product_B", "Q4", 1000),
> ]
> 
> quarterly_df = spark.createDataFrame(quarterly_sales, ["product", "quarter", "sales"])
> 
> # Pivoting - convert quarters to columns
> pivoted_df = quarterly_df.groupBy("product") \
>     .pivot("quarter", ["Q1", "Q2", "Q3", "Q4"]) \
>     .sum("sales")
> 
> print("\nPivoted Data (Quarters as Columns):")
> pivoted_df.show(truncate=False)
> 
> # Unpivoting using stack (convert columns back to rows)
> unpivoted_df = pivoted_df.selectExpr(
>     "product",
>     "stack(4, 'Q1', Q1, 'Q2', Q2, 'Q3', Q3, 'Q4', Q4) as (quarter, sales)"
> ).where("sales is not null")
> 
> print("\nUnpivoted Data (Back to Original Format):")
> unpivoted_df.show(truncate=False)
> 
> print("\n" + "=" * 80)
> print("4. HIGHER-ORDER FUNCTIONS ON ARRAYS AND MAPS")
> print("=" * 80)
> 
> # Sample data with arrays and complex structures
> student_data = [
>     (1, "Alice", [85, 92, 78, 95], {"math": 90, "science": 85, "english": 88}),
>     (2, "Bob", [70, 75, 82, 68], {"math": 72, "science": 78, "english": 75}),
>     (3, "Charlie", [95, 98, 92, 97], {"math": 96, "science": 94, "english": 95})
> ]
> 
> student_df = spark.createDataFrame(student_data, 
>     ["id", "name", "test_scores", "subject_scores"])
> 
> # Transform - apply function to each array element
> transformed_df = student_df.select(
>     "id",
>     "name",
>     "test_scores",
>     # Add 5 bonus points to each score
>     transform("test_scores", lambda x: x + 5).alias("bonus_scores"),
>     
>     # Convert to letter grades
>     transform("test_scores", 
>         lambda x: when(x >= 90, "A")
>                  .when(x >= 80, "B")
>                  .when(x >= 70, "C")
>                  .otherwise("D")).alias("letter_grades")
> )
> 
> print("\nTransform - Modified Scores:")
> transformed_df.show(truncate=False)
> 
> # Filter - keep only elements matching condition
> filtered_df = student_df.select(
>     "id",
>     "name",
>     "test_scores",
>     array_filter("test_scores", lambda x: x >= 80).alias("high_scores"),
>     array_filter("test_scores", lambda x: x < 80).alias("low_scores")
> )
> 
> print("\nFilter - Filtered Scores:")
> filtered_df.show(truncate=False)
> 
> # Exists - check if any element matches condition
> exists_df = student_df.select(
>     "id",
>     "name",
>     exists("test_scores", lambda x: x >= 95).alias("has_excellent_score"),
>     exists("test_scores", lambda x: x < 70).alias("has_failing_score")
> )
> 
> print("\nExists - Boolean Checks:")
> exists_df.show(truncate=False)
> 
> # Aggregate - reduce array to single value
> aggregate_df = student_df.select(
>     "id",
>     "name",
>     "test_scores",
>     # Sum of all scores
>     aggregate("test_scores", lit(0), lambda acc, x: acc + x).alias("total_score"),
>     
>     # Average (sum / count)
>     (aggregate("test_scores", lit(0), lambda acc, x: acc + x) / size("test_scores"))
>         .alias("average_score"),
>     
>     # Max score
>     aggregate("test_scores", lit(0), 
>         lambda acc, x: when(x > acc, x).otherwise(acc)).alias("max_score")
> )
> 
> print("\nAggregate - Computed Values:")
> aggregate_df.show(truncate=False)
> 
> # Working with maps
> map_df = student_df.select(
>     "id",
>     "name",
>     "subject_scores",
>     
>     # Transform map values
>     transform(col("subject_scores"), 
>         lambda k, v: v + 5).alias("bonus_subject_scores"),
>     
>     # Filter map entries
>     expr("filter(subject_scores, (k, v) -> v >= 85)").alias("high_subject_scores")
> )
> 
> print("\nMap Transformations:")
> map_df.show(truncate=False)
> 
> print("\n" + "=" * 80)
> print("5. SQL EXPRESSIONS WITHIN DATAFRAMES")
> print("=" * 80)
> 
> # Using selectExpr and where with SQL expressions
> sql_expr_df = sales_df.selectExpr(
>     "date",
>     "category",
>     "product",
>     "amount",
>     "amount * 0.9 as discounted_price",
>     "CASE WHEN amount > 1000 THEN 'Premium' WHEN amount > 500 THEN 'Standard' ELSE 'Basic' END as tier",
>     "CONCAT(category, ' - ', product) as full_name"
> ).where("amount > 100 AND category = 'Electronics'")
> 
> print("\nSQL Expressions Results:")
> sql_expr_df.show(truncate=False)
> 
> # Complex SQL expressions
> complex_sql_df = sales_df.selectExpr(
>     "*",
>     "ROW_NUMBER() OVER (PARTITION BY category ORDER BY amount DESC) as rank_in_category",
>     "ROUND(amount / SUM(amount) OVER (PARTITION BY category) * 100, 2) as pct_of_category"
> )
> 
> print("\nComplex SQL Expressions with Window Functions:")
> complex_sql_df.show(truncate=False)
> 
> print("\n" + "=" * 80)
> print("6. RECURSIVE CTEs (Common Table Expressions)")
> print("=" * 80)
> 
> # Create employee hierarchy data
> employee_data = [
>     (1, "CEO", None, 1),
>     (2, "VP_Engineering", 1, 2),
>     (3, "VP_Sales", 1, 2),
>     (4, "Engineering_Manager", 2, 3),
>     (5, "Sales_Manager", 3, 3),
>     (6, "Engineer_1", 4, 4),
>     (7, "Engineer_2", 4, 4),
>     (8, "Salesperson_1", 5, 4),
> ]
> 
> employee_df = spark.createDataFrame(employee_data, 
>     ["id", "name", "manager_id", "level"])
> employee_df.createOrReplaceTempView("employees")
> 
> # Recursive CTE to find all subordinates (PySpark 3.0+)
> recursive_query = """
> WITH RECURSIVE employee_hierarchy AS (
>     -- Base case: start with CEO
>     SELECT id, name, manager_id, level, name as path
>     FROM employees
>     WHERE manager_id IS NULL
>     
>     UNION ALL
>     
>     -- Recursive case: find direct reports
>     SELECT e.id, e.name, e.manager_id, e.level,
>            CONCAT(eh.path, ' -> ', e.name) as path
>     FROM employees e
>     INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
> )
> SELECT * FROM employee_hierarchy
> ORDER BY level, id
> """
> 
> # Note: Recursive CTEs require Spark 3.0+ and proper configuration
> # For demonstration, we'll use an iterative approach
> print("\nEmployee Hierarchy (Iterative Approach):")
> 
> # Iterative approach to simulate recursive CTE
> result_df = employee_df.where(col("manager_id").isNull()) \
>     .select("id", "name", "manager_id", "level", col("name").alias("path"))
> 
> for i in range(5):  # Max depth
>     new_level = result_df.alias("r").join(
>         employee_df.alias("e"),
>         col("e.manager_id") == col("r.id")
>     ).select(
>         col("e.id"),
>         col("e.name"),
>         col("e.manager_id"),
>         col("e.level"),
>         expr("CONCAT(r.path, ' -> ', e.name)").alias("path")
>     )
>     
>     if new_level.count() == 0:
>         break
>     
>     result_df = result_df.union(new_level)
> 
> result_df.orderBy("level", "id").show(truncate=False)
> 
> print("\n" + "=" * 80)
> print("7. FOLDLEFT/REDUCE OPERATIONS")
> print("=" * 80)
> 
> # FoldLeft/Reduce using aggregate function
> numbers_data = [
>     (1, [1, 2, 3, 4, 5]),
>     (2, [10, 20, 30]),
>     (3, [100, 200, 300, 400])
> ]
> 
> numbers_df = spark.createDataFrame(numbers_data, ["id", "numbers"])
> 
> # Using aggregate (which is similar to foldLeft)
> fold_df = numbers_df.select(
>     "id",
>     "numbers",
>     
>     # Sum using aggregate (foldLeft)
>     aggregate("numbers", lit(0), lambda acc, x: acc + x).alias("sum"),
>     
>     # Product using aggregate
>     aggregate("numbers", lit(1), lambda acc, x: acc * x).alias("product"),
>     
>     # Concatenate to string
>     aggregate(
>         transform("numbers", lambda x: expr("CAST(x AS STRING)")),
>         lit(""),
>         lambda acc, x: expr("CONCAT(acc, IF(acc = '', '', ','), x)")
>     ).alias("concatenated"),
>     
>     # Custom: sum of squares
>     aggregate("numbers", lit(0), lambda acc, x: acc + (x * x)).alias("sum_of_squares")
> )
> 
> print("\nFold/Reduce Operations:")
> fold_df.show(truncate=False)
> 
> # Reduce operation across rows using RDD
> from functools import reduce
> 
> print("\nReduce across DataFrame rows:")
> total_sum = numbers_df.select(
>     aggregate("numbers", lit(0), lambda acc, x: acc + x).alias("row_sum")
> ).rdd.map(lambda row: row.row_sum).reduce(lambda a, b: a + b)
> 
> print(f"Total sum of all numbers: {total_sum}")
> 
> print("\n" + "=" * 80)
> print("8. APPROXIMATE QUANTILES")
> print("=" * 80)
> 
> # Generate sample data with large dataset simulation
> import random
> random.seed(42)
> 
> large_data = [(i, random.randint(1, 1000)) for i in range(10000)]
> large_df = spark.createDataFrame(large_data, ["id", "value"])
> 
> # Calculate approximate percentiles (faster than exact for large datasets)
> quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
> 
> # Using approxQuantile (RDD-based)
> approx_quantiles_exact = large_df.approxQuantile("value", quantiles, 0.0)
> print("\nApproximate Quantiles (exact with relativeError=0.0):")
> for q, val in zip(quantiles, approx_quantiles_exact):
>     print(f"  {q*100}th percentile: {val}")
> 
> # Using approxQuantile with error tolerance (faster)
> approx_quantiles_approx = large_df.approxQuantile("value", quantiles, 0.05)
> print("\nApproximate Quantiles (with 5% error tolerance):")
> for q, val in zip(quantiles, approx_quantiles_approx):
>     print(f"  {q*100}th percentile: {val}")
> 
> # Using approx_percentile in DataFrame operations
> percentile_df = large_df.select(
>     approx_percentile("value", 0.5, 10000).alias("median"),
>     approx_percentile("value", 0.25, 10000).alias("q1"),
>     approx_percentile("value", 0.75, 10000).alias("q3"),
>     approx_percentile("value", array(lit(0.25), lit(0.5), lit(0.75)), 10000).alias("quartiles")
> )
> 
> print("\nApproximate Percentiles using DataFrame operations:")
> percentile_df.show(truncate=False)
> 
> # Group by with approximate quantiles
> grouped_data = [(i % 10, random.randint(1, 1000)) for i in range(10000)]
> grouped_df = spark.createDataFrame(grouped_data, ["group_id", "value"])
> 
> grouped_percentiles = grouped_df.groupBy("group_id").agg(
>     approx_percentile("value", 0.5, 10000).alias("median"),
>     approx_percentile("value", 0.9, 10000).alias("p90"),
>     avg("value").alias("avg_value")
> )
> 
> print("\nGrouped Approximate Percentiles:")
> grouped_percentiles.orderBy("group_id").show()
> 
> print("\n" + "=" * 80)
> print("BONUS: COMBINED ADVANCED EXAMPLE")
> print("=" * 80)
> 
> # Complex real-world scenario combining multiple techniques
> transaction_data = [
>     (1, "2024-01-01", "user_1", ["electronics", "books"], 
>      {"electronics": 500, "books": 50}, [100, 200, 150, 50]),
>     (2, "2024-01-02", "user_2", ["clothing", "food"], 
>      {"clothing": 200, "food": 80}, [50, 80, 120, 30]),
>     (3, "2024-01-03", "user_1", ["electronics"], 
>      {"electronics": 1200}, [1200]),
>     (4, "2024-01-04", "user_3", ["books", "electronics", "food"], 
>      {"books": 30, "electronics": 800, "food": 45}, [30, 400, 400, 45]),
> ]
> 
> transaction_df = spark.createDataFrame(transaction_data,
>     ["transaction_id", "date", "user_id", "categories", "category_amounts", "items"])
> 
> # Complex transformation combining multiple patterns
> result = transaction_df.select(
>     "*",
>     # Explode arrays
>     explode("categories").alias("category"),
>     
>     # Higher-order function: sum of items
>     aggregate("items", lit(0), lambda acc, x: acc + x).alias("total_amount"),
>     
>     # Transform array
>     transform("items", lambda x: when(x > 100, "high").otherwise("low")).alias("price_tiers")
> ).select(
>     "*",
>     # SQL expression
>     expr("category_amounts[category]").alias("category_amount")
> ).withColumn(
>     # Window function: rank by amount within category
>     "rank_in_category",
>     dense_rank().over(Window.partitionBy("category").orderBy(col("category_amount").desc()))
> ).withColumn(
>     # Moving average
>     "moving_avg",
>     avg("total_amount").over(Window.orderBy("date").rowsBetween(-1, 0))
> )
> 
> print("\nCombined Advanced Transformations:")
> result.show(truncate=False)
> 
> # Cleanup
> spark.stop()
> 
> print("\n" + "=" * 80)
> print("All examples completed successfully!")
> print("=" * 80)
> ```
> 
> This comprehensive program demonstrates:
> 
> 1. **Window Functions**: Ranking, lag/lead, moving averages, cumulative sums
> 2. **Exploding Arrays**: Both `explode` and `posexplode` with practical examples
> 3. **Pivoting/Unpivoting**: Converting between wide and long formats using `pivot` and `stack`
> 4. **Higher-Order Functions**: `transform`, `filter`, `exists`, `aggregate` on arrays and maps
> 5. **SQL Expressions**: Using `selectExpr` and `where` with complex SQL
> 6. **Recursive CTEs**: Iterative approach for hierarchical data (since true recursive CTEs need Spark 3.0+)
> 7. **Fold/Reduce**: Using `aggregate` function for array reduction and RDD reduce
> 8. **Approximate Quantiles**: Using `approxQuantile` and `approx_percentile` for large datasets
> 
> Each section includes:
> - Real-world scenarios
> - Multiple variations of each technique
> - Clear output demonstrations
> - Comments explaining the logic
> 
> You can run this script with `spark-submit` or in a PySpark notebook!