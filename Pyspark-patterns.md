Here's a comprehensive list of PySpark patterns:

### **I. Core SparkSession & Context Management**
*   Singleton SparkSession
*   Configuring SparkSession (Runtime Properties)
*   Accessing SparkContext

### **II. Data Loading & Saving (I/O)**
*   Reading Various Data Formats (CSV, JSON, Parquet, ORC, JDBC, Delta, Iceberg)
*   Writing to Various Data Formats (modes: overwrite, append, ignore, errorIfExists)
*   Schema Provisioning/Enforcement on Read
*   Partitioned Writes
*   Bucketing on Write
*   Reading/Writing from Cloud Storage (S3, ADLS, GCS)
*   Reading from Streaming Sources (Kafka, FileStream, Rate)
*   Writing to Streaming Sinks (Console, FileStream, Kafka, ForeachBatch)

### **III. DataFrame Transformations (Basic)**
*   Column Selection & Renaming (`select`, `selectExpr`, `withColumnRenamed`)
*   Filtering Data (`filter`, `where`)
*   Adding/Modifying Columns (`withColumn`)
*   Dropping Columns (`drop`, `dropDuplicates`)
*   Handling Missing Values (`na.drop`, `na.fill`, `na.replace`)
*   Sorting Data (`orderBy`, `sort`)
*   Aggregations (`groupBy`, `agg`)
*   Joining DataFrames (Inner, Outer, Left, Right, Semi, Anti)
*   Unioning DataFrames (`union`, `unionByName`)
*   Type Casting (`cast`)
*   Sampling Data (`sample`)
*   Repartioning / Coalescing Data (`repartition`, `coalesce`)

### **IV. DataFrame Transformations (Advanced)**
*   Window Functions (Ranking, Lag/Lead, Moving Averages, Cumulative Sums)
*   Exploding Arrays (`explode`, `posexplode`)
*   Pivoting / Unpivoting (`pivot`, `stack`)
*   Higher-Order Functions on Arrays and Maps (`transform`, `filter`, `exists`, `aggregate`, `map_arrays`)
*   Using SQL Expressions within DataFrames (`selectExpr`, `where`)
*   Recursive CTEs (Common Table Expressions)
*   FoldLeft/Reduce operations
*   Approximate Quantiles

### **V. Schema & Type Handling**
*   Explicit Schema Definition (StructType)
*   Schema Evolution (mergeSchema)
*   Handling Complex Types (ArrayType, MapType, StructType)
*   Schema Inference

### **VI. Performance Optimization**
*   Caching/Persisting DataFrames (`cache`, `persist`)
*   Controlling Shuffle Partitions
*   Broadcasting Small DataFrames/Variables
*   Optimizing Joins (Broadcast Join Hint, Join Reordering)
*   Predicate Pushdown
*   Columnar Pruning
*   Data Partitioning Strategies (Manual Partitioning, Bucketing)
*   Using `explain()` for Query Plan Analysis
*   Spark UI Monitoring for Performance Bottlenecks
*   Memory Management (Storage Levels)
*   Cost-Based Optimizer Configuration

### **VII. User-Defined Functions (UDFs)**
*   Scalar UDFs (Python UDFs)
*   Vectorized/Pandas UDFs for performance improvement
*   User-Defined Aggregate Functions (UDAFs)
*   SQL UDFs

### **VIII. Structured Streaming**
*   Windowed Aggregations on Streams
*   Stateful Stream Processing
*   Watermarking for Late Data
*   Joining Streams with Static DataFrames
*   Joining Streams with Streams
*   Different Output Modes (Append, Complete, Update)
*   Idempotent Sinks (`foreachBatch`)
*   Checkpointing

### **IX. Machine Learning (MLlib)**
*   Feature Engineering (VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, Tokenizer)
*   Model Training (Estimators, Transformers)
*   Model Evaluation (Evaluators)
*   ML Pipelines for Workflow Orchestration
*   Hyperparameter Tuning (CrossValidator, TrainValidationSplit)
*   Model Persistence (Saving/Loading Models)

### **X. Error Handling & Debugging**
*   Using `try-except` blocks for I/O operations
*   Logging Spark operations
*   Inspecting DataFrame content (`show`, `head`, `take`, `collect`, `printSchema`, `describe`)
*   Analyzing Stack Traces for Job Failures
*   Spark UI for Debugging Failed Stages/Tasks

### **XI. Deployment & Orchestration**
*   Spark-Submit Job Submission
*   Passing Configuration and Arguments to Spark Jobs
*   Managing Dependencies (JARs, Python files)
*   Environment Variable Configuration

### **XII. Testing**
*   Unit Testing Spark Code (e.g., using `spark-testing-base` or creating local `SparkSession` for tests)
*   Mocking External Dependencies for Tests
