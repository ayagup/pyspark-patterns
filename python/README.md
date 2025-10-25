# PySpark Design Patterns - Example Programs

This directory contains practical Python examples demonstrating all the PySpark design patterns from the main documentation.

## Files Overview

### 01_data_ingestion_examples.py
Demonstrates data ingestion patterns:
- Schema Evolution Pattern
- Incremental Load Pattern
- Multiformat Reader Pattern
- Data Source Abstraction Pattern
- Partition Pruning Pattern

### 02_transformation_examples.py
Demonstrates transformation patterns:
- Pipeline Pattern
- Transformation Registry Pattern
- Window Function Pattern
- Pivot Pattern
- Deduplication Pattern
- Data Enrichment Pattern

### 03_data_quality_examples.py
Demonstrates data quality patterns:
- Validation Pattern
- Null Handling Pattern
- Outlier Detection Pattern (IQR & Z-score)
- Data Profiling Pattern
- Schema Validation Pattern

### 04_performance_optimization_examples.py
Demonstrates performance optimization patterns:
- Broadcast Join Pattern
- Partitioning Pattern
- Caching Pattern
- Salting Pattern for Data Skew
- Predicate Pushdown Pattern
- Adaptive Query Execution (AQE)
- Bucketing Pattern

### 05_error_handling_examples.py
Demonstrates error handling patterns:
- Try-Catch Pattern
- UDF Error Handling
- Dead Letter Queue (DLQ) Pattern
- Retry Pattern with Exponential Backoff
- Circuit Breaker Pattern
- Validation with Early Exit Pattern

## Running the Examples

### Prerequisites
```bash
pip install pyspark
```

### Run Individual Examples
```bash
# Data Ingestion Patterns
python 01_data_ingestion_examples.py

# Transformation Patterns
python 02_transformation_examples.py

# Data Quality Patterns
python 03_data_quality_examples.py

# Performance Optimization Patterns
python 04_performance_optimization_examples.py

# Error Handling Patterns
python 05_error_handling_examples.py
```

### Run All Examples
```bash
# Windows
for %f in (*.py) do python %f

# Linux/Mac
for file in *.py; do python "$file"; done
```

## Notes

- All examples use local Spark mode (`local[*]`) for easy execution
- Temporary data is written to `/tmp/` directory
- Each example is self-contained and can run independently
- Output includes detailed explanations and results
- Examples demonstrate both the pattern and its practical use case

## Example Output Structure

Each example follows this structure:
1. **Pattern Name** - Clear identification of the pattern
2. **Data Setup** - Sample data creation
3. **Pattern Implementation** - Code demonstrating the pattern
4. **Results** - Output showing the pattern in action
5. **Explanation** - Comments explaining key concepts

## Cleaning Up

Temporary files are created in `/tmp/` during execution. To clean up:

```bash
# Remove temporary Spark data
rm -rf /tmp/*.parquet /tmp/*.csv /tmp/*.json
```

## Learning Path

Recommended order for learning:
1. Start with Data Ingestion patterns
2. Move to Transformation patterns
3. Study Data Quality patterns
4. Explore Performance Optimization patterns
5. Master Error Handling patterns

## Additional Resources

- Main documentation: `../pyspark_design_patterns.md`
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- PySpark API Reference: https://spark.apache.org/docs/latest/api/python/

## Contributing

Feel free to add more examples or improve existing ones!
