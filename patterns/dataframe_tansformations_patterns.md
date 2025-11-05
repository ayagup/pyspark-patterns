### [Continue](https://continue.dev) session transcript
 Exported: 11/5/2025, 4:16:19 PM

#### _User_

> generate a python program in pyspark for each of the below patterns:
> ```
> ### **III. DataFrame Transformations (Basic)**
> *   Column Selection & Renaming (`select`, `selectExpr`, `withColumnRenamed`)
> *   Filtering Data (`filter`, `where`)
> *   Adding/Modifying Columns (`withColumn`)
> *   Dropping Columns (`drop`, `dropDuplicates`)
> *   Handling Missing Values (`na.drop`, `na.fill`, `na.replace`)
> *   Sorting Data (`orderBy`, `sort`)
> *   Aggregations (`groupBy`, `agg`)
> *   Joining DataFrames (Inner, Outer, Left, Right, Semi, Anti)
> *   Unioning DataFrames (`union`, `unionByName`)
> *   Type Casting (`cast`)
> *   Sampling Data (`sample`)
> *   Repartioning / Coalescing Data (`repartition`, `coalesce`)
> ```

#### _Assistant_

> # PySpark DataFrame Transformations - Comprehensive Examples
> 
> I'll create a complete Python program demonstrating all these patterns with practical examples.
> 
> ```python src/dataframe_transformations.py
> """
> PySpark DataFrame Transformations - Complete Guide
> Demonstrates all basic transformation patterns with practical examples
> """
> 
> from pyspark.sql import SparkSession
> from pyspark.sql.functions import col, lit, when, avg, sum, count, max, min
> from pyspark.sql.types import IntegerType, DoubleType, StringType
> 
> def create_spark_session():
>     """Initialize Spark Session"""
>     return SparkSession.builder \
>         .appName("DataFrame Transformations Guide") \
>         .master("local[*]") \
>         .getOrCreate()
> 
> def create_sample_data(spark):
>     """Create sample datasets for demonstrations"""
>     
>     # Employees DataFrame
>     employees_data = [
>         (1, "John Doe", "Engineering", 75000.0, 28, None),
>         (2, "Jane Smith", "Marketing", 65000.0, 32, "New York"),
>         (3, "Bob Johnson", "Engineering", 80000.0, 35, "San Francisco"),
>         (4, "Alice Williams", "HR", 60000.0, 29, "Chicago"),
>         (5, "Charlie Brown", "Engineering", 85000.0, 40, "Seattle"),
>         (6, "Diana Prince", "Marketing", 70000.0, 31, None),
>         (7, "Eve Davis", "Engineering", 75000.0, 28, "Boston"),
>         (8, "Frank Miller", "HR", 62000.0, 33, "New York"),
>         (9, "Grace Lee", "Engineering", 90000.0, 38, "San Francisco"),
>         (10, "Henry Wilson", "Marketing", 68000.0, 30, "Chicago")
>     ]
>     
>     employees_df = spark.createDataFrame(
>         employees_data,
>         ["id", "name", "department", "salary", "age", "city"]
>     )
>     
>     # Departments DataFrame (for joins)
>     departments_data = [
>         ("Engineering", "Tech Building", 50),
>         ("Marketing", "Main Building", 20),
>         ("HR", "Admin Building", 10),
>         ("Finance", "Tower A", 15)
>     ]
>     
>     departments_df = spark.createDataFrame(
>         departments_data,
>         ["dept_name", "location", "employee_count"]
>     )
>     
>     # Projects DataFrame (for additional joins)
>     projects_data = [
>         (1, 101, "Project Alpha"),
>         (2, 101, "Project Beta"),
>         (3, 102, "Project Gamma"),
>         (5, 103, "Project Delta"),
>         (7, 101, "Project Epsilon")
>     ]
>     
>     projects_df = spark.createDataFrame(
>         projects_data,
>         ["emp_id", "project_id", "project_name"]
>     )
>     
>     return employees_df, departments_df, projects_df
> 
> 
> def demo_column_selection_renaming(df):
>     """1. Column Selection & Renaming"""
>     print("\n" + "="*80)
>     print("1. COLUMN SELECTION & RENAMING")
>     print("="*80)
>     
>     # Basic selection
>     print("\n--- Select specific columns ---")
>     df.select("name", "department", "salary").show(5)
>     
>     # Select with column objects
>     print("\n--- Select using col() ---")
>     df.select(col("name"), col("salary")).show(5)
>     
>     # Select with expressions
>     print("\n--- SelectExpr with calculations ---")
>     df.selectExpr(
>         "name",
>         "salary",
>         "salary * 12 as annual_salary",
>         "UPPER(department) as dept_upper"
>     ).show(5)
>     
>     # Rename columns
>     print("\n--- Rename columns ---")
>     df.withColumnRenamed("name", "employee_name") \
>       .withColumnRenamed("department", "dept") \
>       .show(5)
>     
>     # Select all columns and add new ones
>     print("\n--- Select all + new calculated column ---")
>     df.select("*", (col("salary") * 1.1).alias("salary_with_bonus")).show(5)
> 
> 
> def demo_filtering(df):
>     """2. Filtering Data"""
>     print("\n" + "="*80)
>     print("2. FILTERING DATA")
>     print("="*80)
>     
>     # Filter using filter()
>     print("\n--- Filter: salary > 70000 ---")
>     df.filter(col("salary") > 70000).show()
>     
>     # Filter using where() (alias for filter)
>     print("\n--- Where: department = 'Engineering' ---")
>     df.where(col("department") == "Engineering").show()
>     
>     # Multiple conditions with &
>     print("\n--- Multiple conditions (AND) ---")
>     df.filter((col("salary") > 70000) & (col("age") < 35)).show()
>     
>     # Multiple conditions with |
>     print("\n--- Multiple conditions (OR) ---")
>     df.filter((col("department") == "Engineering") | (col("department") == "Marketing")).show()
>     
>     # Filter with SQL-like expressions
>     print("\n--- Filter with SQL expressions ---")
>     df.filter("salary > 75000 AND age < 35").show()
>     
>     # Filter NULL values
>     print("\n--- Filter NULL cities ---")
>     df.filter(col("city").isNull()).show()
>     
>     # Filter NOT NULL values
>     print("\n--- Filter NOT NULL cities ---")
>     df.filter(col("city").isNotNull()).show()
> 
> 
> def demo_adding_modifying_columns(df):
>     """3. Adding/Modifying Columns"""
>     print("\n" + "="*80)
>     print("3. ADDING/MODIFYING COLUMNS")
>     print("="*80)
>     
>     # Add a new column with literal value
>     print("\n--- Add literal column ---")
>     df.withColumn("country", lit("USA")).show(5)
>     
>     # Add calculated column
>     print("\n--- Add calculated column ---")
>     df.withColumn("monthly_salary", col("salary") / 12).show(5)
>     
>     # Add conditional column
>     print("\n--- Add conditional column ---")
>     df.withColumn(
>         "salary_grade",
>         when(col("salary") >= 80000, "High")
>         .when(col("salary") >= 70000, "Medium")
>         .otherwise("Low")
>     ).show()
>     
>     # Modify existing column
>     print("\n--- Modify existing column ---")
>     df.withColumn("salary", col("salary") * 1.05).show(5)
>     
>     # Add multiple columns at once
>     print("\n--- Add multiple columns ---")
>     df.withColumn("annual_salary", col("salary") * 12) \
>       .withColumn("tax", col("salary") * 0.2) \
>       .withColumn("net_salary", col("salary") - col("tax")) \
>       .show(5)
> 
> 
> def demo_dropping_columns(df):
>     """4. Dropping Columns & Duplicates"""
>     print("\n" + "="*80)
>     print("4. DROPPING COLUMNS & DUPLICATES")
>     print("="*80)
>     
>     # Drop single column
>     print("\n--- Drop single column ---")
>     df.drop("city").show(5)
>     
>     # Drop multiple columns
>     print("\n--- Drop multiple columns ---")
>     df.drop("city", "age").show(5)
>     
>     # Create DataFrame with duplicates
>     duplicate_data = [
>         (1, "John", "Engineering"),
>         (2, "Jane", "Marketing"),
>         (3, "John", "Engineering"),  # Duplicate
>         (4, "Alice", "HR"),
>         (5, "Jane", "Marketing")  # Duplicate
>     ]
>     duplicate_df = df.sparkSession.createDataFrame(
>         duplicate_data,
>         ["id", "name", "department"]
>     )
>     
>     print("\n--- Original data with duplicates ---")
>     duplicate_df.show()
>     
>     # Drop all duplicate rows
>     print("\n--- Drop all duplicates ---")
>     duplicate_df.dropDuplicates().show()
>     
>     # Drop duplicates based on specific columns
>     print("\n--- Drop duplicates based on name, department ---")
>     duplicate_df.dropDuplicates(["name", "department"]).show()
> 
> 
> def demo_handling_missing_values(df):
>     """5. Handling Missing Values"""
>     print("\n" + "="*80)
>     print("5. HANDLING MISSING VALUES")
>     print("="*80)
>     
>     print("\n--- Original data (notice NULL cities) ---")
>     df.show()
>     
>     # Drop rows with any NULL values
>     print("\n--- Drop rows with ANY null ---")
>     df.na.drop().show()
>     
>     # Drop rows with NULL in specific columns
>     print("\n--- Drop rows with null in 'city' ---")
>     df.na.drop(subset=["city"]).show()
>     
>     # Drop only if all values are NULL
>     print("\n--- Drop only if ALL values are null ---")
>     df.na.drop(how="all").show()
>     
>     # Fill NULL with specific value
>     print("\n--- Fill null cities with 'Unknown' ---")
>     df.na.fill({"city": "Unknown"}).show()
>     
>     # Fill NULL with different values for different columns
>     print("\n--- Fill with multiple values ---")
>     df.na.fill({"city": "Not Specified", "age": 0}).show()
>     
>     # Replace specific values
>     print("\n--- Replace 'Engineering' with 'Tech' ---")
>     df.na.replace("Engineering", "Tech", subset=["department"]).show()
> 
> 
> def demo_sorting(df):
>     """6. Sorting Data"""
>     print("\n" + "="*80)
>     print("6. SORTING DATA")
>     print("="*80)
>     
>     # Sort by single column (ascending)
>     print("\n--- Sort by salary (ascending) ---")
>     df.orderBy("salary").show()
>     
>     # Sort by single column (descending)
>     print("\n--- Sort by salary (descending) ---")
>     df.orderBy(col("salary").desc()).show()
>     
>     # Sort by multiple columns
>     print("\n--- Sort by department (asc), then salary (desc) ---")
>     df.orderBy(col("department").asc(), col("salary").desc()).show()
>     
>     # Using sort() - alias for orderBy()
>     print("\n--- Using sort() instead of orderBy() ---")
>     df.sort(col("age").desc()).show(5)
> 
> 
> def demo_aggregations(df):
>     """7. Aggregations"""
>     print("\n" + "="*80)
>     print("7. AGGREGATIONS")
>     print("="*80)
>     
>     # Simple groupBy with count
>     print("\n--- Count by department ---")
>     df.groupBy("department").count().show()
>     
>     # Multiple aggregations
>     print("\n--- Multiple aggregations by department ---")
>     df.groupBy("department").agg(
>         count("*").alias("employee_count"),
>         avg("salary").alias("avg_salary"),
>         max("salary").alias("max_salary"),
>         min("salary").alias("min_salary"),
>         sum("salary").alias("total_salary")
>     ).show()
>     
>     # Aggregation without groupBy
>     print("\n--- Overall statistics ---")
>     df.agg(
>         count("*").alias("total_employees"),
>         avg("salary").alias("average_salary"),
>         max("age").alias("oldest_employee"),
>         min("age").alias("youngest_employee")
>     ).show()
>     
>     # Group by multiple columns
>     print("\n--- Group by department and city ---")
>     df.groupBy("department", "city").agg(
>         count("*").alias("count"),
>         avg("salary").alias("avg_salary")
>     ).orderBy("department", "city").show()
> 
> 
> def demo_joins(employees_df, departments_df, projects_df):
>     """8. Joining DataFrames"""
>     print("\n" + "="*80)
>     print("8. JOINING DATAFRAMES")
>     print("="*80)
>     
>     print("\n--- Employees DataFrame ---")
>     employees_df.show()
>     
>     print("\n--- Departments DataFrame ---")
>     departments_df.show()
>     
>     print("\n--- Projects DataFrame ---")
>     projects_df.show()
>     
>     # Inner Join
>     print("\n--- INNER JOIN ---")
>     employees_df.join(
>         departments_df,
>         employees_df.department == departments_df.dept_name,
>         "inner"
>     ).select(
>         employees_df.name,
>         employees_df.department,
>         departments_df.location
>     ).show()
>     
>     # Left Outer Join
>     print("\n--- LEFT OUTER JOIN ---")
>     employees_df.join(
>         projects_df,
>         employees_df.id == projects_df.emp_id,
>         "left"
>     ).select(
>         employees_df.name,
>         projects_df.project_name
>     ).show()
>     
>     # Right Outer Join
>     print("\n--- RIGHT OUTER JOIN ---")
>     employees_df.join(
>         departments_df,
>         employees_df.department == departments_df.dept_name,
>         "right"
>     ).select(
>         employees_df.name,
>         departments_df.dept_name,
>         departments_df.location
>     ).show()
>     
>     # Full Outer Join
>     print("\n--- FULL OUTER JOIN ---")
>     employees_df.join(
>         departments_df,
>         employees_df.department == departments_df.dept_name,
>         "outer"
>     ).select(
>         employees_df.name,
>         departments_df.dept_name,
>         departments_df.location
>     ).show()
>     
>     # Left Semi Join (returns only left side records that have matches)
>     print("\n--- LEFT SEMI JOIN ---")
>     employees_df.join(
>         projects_df,
>         employees_df.id == projects_df.emp_id,
>         "semi"
>     ).show()
>     
>     # Left Anti Join (returns only left side records that DON'T have matches)
>     print("\n--- LEFT ANTI JOIN (employees without projects) ---")
>     employees_df.join(
>         projects_df,
>         employees_df.id == projects_df.emp_id,
>         "anti"
>     ).show()
> 
> 
> def demo_unions(df):
>     """9. Unioning DataFrames"""
>     print("\n" + "="*80)
>     print("9. UNIONING DATAFRAMES")
>     print("="*80)
>     
>     # Create two DataFrames for union
>     new_employees_data = [
>         (11, "Ian Moore", "Finance", 72000.0, 34, "Austin"),
>         (12, "Julia Roberts", "Finance", 78000.0, 36, "Dallas")
>     ]
>     
>     new_employees_df = df.sparkSession.createDataFrame(
>         new_employees_data,
>         ["id", "name", "department", "salary", "age", "city"]
>     )
>     
>     print("\n--- Original employees (first 5) ---")
>     df.show(5)
>     
>     print("\n--- New employees ---")
>     new_employees_df.show()
>     
>     # Union (requires same schema and order)
>     print("\n--- UNION (combined data) ---")
>     df.union(new_employees_df).orderBy("id").show()
>     
>     # Create DataFrame with columns in different order
>     reordered_data = [
>         (13, "Engineering", "Kevin Nash", 81000.0, "Miami", 37),
>     ]
>     
>     reordered_df = df.sparkSession.createDataFrame(
>         reordered_data,
>         ["id", "department", "name", "salary", "city", "age"]
>     )
>     
>     print("\n--- DataFrame with different column order ---")
>     reordered_df.show()
>     
>     # UnionByName (matches by column name, not position)
>     print("\n--- UNION BY NAME (handles different column order) ---")
>     df.unionByName(reordered_df).orderBy("id").show()
> 
> 
> def demo_type_casting(df):
>     """10. Type Casting"""
>     print("\n" + "="*80)
>     print("10. TYPE CASTING")
>     print("="*80)
>     
>     print("\n--- Original schema ---")
>     df.printSchema()
>     
>     # Cast salary to integer
>     print("\n--- Cast salary to Integer ---")
>     df_casted = df.withColumn("salary", col("salary").cast(IntegerType()))
>     df_casted.printSchema()
>     df_casted.show(5)
>     
>     # Cast to string
>     print("\n--- Cast age to String ---")
>     df.withColumn("age", col("age").cast(StringType())).printSchema()
>     
>     # Multiple casts using selectExpr
>     print("\n--- Multiple casts using selectExpr ---")
>     df.selectExpr(
>         "id",
>         "name",
>         "department",
>         "CAST(salary AS INTEGER) as salary_int",
>         "CAST(age AS STRING) as age_str",
>         "city"
>     ).printSchema()
> 
> 
> def demo_sampling(df):
>     """11. Sampling Data"""
>     print("\n" + "="*80)
>     print("11. SAMPLING DATA")
>     print("="*80)
>     
>     print(f"\n--- Original data count: {df.count()} ---")
>     
>     # Random sample (30% of data, without replacement)
>     print("\n--- Random sample (30%, without replacement) ---")
>     sample1 = df.sample(withReplacement=False, fraction=0.3, seed=42)
>     print(f"Sample count: {sample1.count()}")
>     sample1.show()
>     
>     # Random sample (50% of data, with replacement)
>     print("\n--- Random sample (50%, with replacement) ---")
>     sample2 = df.sample(withReplacement=True, fraction=0.5, seed=42)
>     print(f"Sample count: {sample2.count()}")
>     sample2.show()
>     
>     # Sample by fraction per stratum
>     print("\n--- Stratified sample by department ---")
>     fractions = {"Engineering": 0.5, "Marketing": 0.7, "HR": 1.0}
>     stratified_sample = df.sampleBy("department", fractions, seed=42)
>     print(f"Stratified sample count: {stratified_sample.count()}")
>     stratified_sample.groupBy("department").count().show()
> 
> 
> def demo_repartitioning_coalescing(df):
>     """12. Repartitioning & Coalescing"""
>     print("\n" + "="*80)
>     print("12. REPARTITIONING & COALESCING")
>     print("="*80)
>     
>     print(f"\n--- Original partition count: {df.rdd.getNumPartitions()} ---")
>     
>     # Repartition to increase partitions
>     print("\n--- Repartition to 5 partitions ---")
>     df_repartitioned = df.repartition(5)
>     print(f"New partition count: {df_repartitioned.rdd.getNumPartitions()}")
>     
>     # Repartition by column (hash partitioning)
>     print("\n--- Repartition by department column ---")
>     df_repartitioned_by_col = df.repartition(3, "department")
>     print(f"Partition count: {df_repartitioned_by_col.rdd.getNumPartitions()}")
>     
>     # Show distribution across partitions
>     print("\n--- Records per partition (by department) ---")
>     df_repartitioned_by_col.groupBy("department").count().show()
>     
>     # Coalesce to reduce partitions (more efficient than repartition)
>     print("\n--- Coalesce to 2 partitions ---")
>     df_coalesced = df_repartitioned.coalesce(2)
>     print(f"New partition count: {df_coalesced.rdd.getNumPartitions()}")
>     
>     print("\n--- Difference between Repartition and Coalesce ---")
>     print("Repartition: Full shuffle, can increase or decrease partitions")
>     print("Coalesce: No shuffle (or minimal), can only decrease partitions")
>     print("Coalesce is more efficient when reducing partitions")
> 
> 
> def main():
>     """Main execution function"""
>     print("\n" + "="*80)
>     print("PYSPARK DATAFRAME TRANSFORMATIONS - COMPLETE GUIDE")
>     print("="*80)
>     
>     # Initialize Spark
>     spark = create_spark_session()
>     spark.sparkContext.setLogLevel("ERROR")
>     
>     # Create sample data
>     employees_df, departments_df, projects_df = create_sample_data(spark)
>     
>     # Run all demonstrations
>     demo_column_selection_renaming(employees_df)
>     demo_filtering(employees_df)
>     demo_adding_modifying_columns(employees_df)
>     demo_dropping_columns(employees_df)
>     demo_handling_missing_values(employees_df)
>     demo_sorting(employees_df)
>     demo_aggregations(employees_df)
>     demo_joins(employees_df, departments_df, projects_df)
>     demo_unions(employees_df)
>     demo_type_casting(employees_df)
>     demo_sampling(employees_df)
>     demo_repartitioning_coalescing(employees_df)
>     
>     print("\n" + "="*80)
>     print("ALL DEMONSTRATIONS COMPLETED!")
>     print("="*80)
>     
>     # Stop Spark session
>     spark.stop()
> 
> 
> if __name__ == "__main__":
>     main()
> ```
> 
> This comprehensive program demonstrates all the DataFrame transformation patterns you requested. Here's what's included:
> 
> ## Key Features:
> 
> 1. **Column Selection & Renaming** - `select()`, `selectExpr()`, `withColumnRenamed()`
> 2. **Filtering** - `filter()`, `where()` with various conditions
> 3. **Adding/Modifying Columns** - `withColumn()`, conditional columns with `when()`
> 4. **Dropping** - `drop()`, `dropDuplicates()` 
> 5. **Missing Values** - `na.drop()`, `na.fill()`, `na.replace()`
> 6. **Sorting** - `orderBy()`, `sort()` with multiple columns
> 7. **Aggregations** - `groupBy()`, `agg()` with multiple functions
> 8. **Joins** - All types: inner, left, right, outer, semi, anti
> 9. **Unions** - `union()`, `unionByName()`
> 10. **Type Casting** - `cast()` to different data types
> 11. **Sampling** - Random and stratified sampling
> 12. **Repartitioning** - `repartition()`, `coalesce()`
> 
> ## To Run:
> ```bash
> python src/dataframe_transformations.py
> ```
> 
> Each section is self-contained with clear examples and outputs!