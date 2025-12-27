# In WSL 2 with docker installed


# Dump from public-data project to GCS

EXPORT DATA OPTIONS(
  uri='gs://translateqna-spark/bigquery_public_data/austin_311/311_service_requests_*.csv',
  format='CSV',
  overwrite=true,
  header=true,
  field_delimiter=','
) AS
SELECT * FROM `bigquery-public-data.austin_311.311_service_requests`
WHERE date_column > '2025-01-01'




# Run docker spark on local
  docker run -it \
 -v ~/spark_ivy_cache:/tmp/.ivy2 \
 -v /mnt/c/Users/Lenovo/Downloads/translateqna-b3791e36203a.json:/opt/spark/key.json \
spark:python3-java17 /opt/spark/bin/pyspark  \
  --packages com.google.cloud.spark:spark-bigquery-with-dependencies_2.13:0.43.1,com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.22 \
 --conf "spark.driver.extraJavaOptions=-Divy.cache.dir=/tmp -Divy.home=/tmp"   \
 --conf "spark.executor.extraJavaOptions=-Divy.cache.dir=/tmp -Divy.home=/tmp" \
 --conf "spark.hadoop.google.cloud.auth.service.account.enable=true" \
  --conf "spark.hadoop.google.cloud.auth.service.account.json.key.file=/opt/spark/key.json" \
  --conf "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem"  \
  --conf "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS" \
   --conf "spark.hadoop.fs.gs.auth.service.account.enable=true" \
  --conf "spark.hadoop.fs.gs.auth.service.account.json.keyfile=/opt/spark/key.json"
 
 
 
 
   

# Write below commands on pyspark console
# 3. Read data from GCS
# Replace with your actual bucket and file path
df = spark.read.csv("gs://translateqna-spark/bigquery_public_data/austin_311/*.csv", header=True, inferSchema=True)

# 4. Write data to BigQuery
# You must specify a temporary GCS bucket for BigQuery to use during the export/import process
df.write \
  .format("bigquery") \
  .option("temporaryGcsBucket", "translateqna-spark") \
  .option("table", "translateqna.bls_qcew.311_service_requests") \
  .option("project", "translateqna") \
  .option("parentProject", "translateqna") \
  .option("credentialsFile", "/opt/spark/key.json") \
  .mode("overwrite") \
  .save()

print("Data transfer complete!")
