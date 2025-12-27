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
# 3. Read data from GCS, Write to Bigquery
df = spark.read.csv("gs://translateqna-spark/bigquery_public_data/austin_311/*.csv", header=True, inferSchema=True)

df.write \
  .format("bigquery") \
  .option("temporaryGcsBucket", "translateqna-spark") \
  .option("table", "translateqna.bls_qcew.311_service_requests") \
  .option("project", "translateqna") \
  .option("parentProject", "translateqna") \
  .option("credentialsFile", "/opt/spark/key.json") \
  .mode("overwrite") \
  .save()




# Read from Bigquery, write to GCS
df1=spark.read \
	.format('bigquery') \
	  .option("temporaryGcsBucket", "translateqna-spark") \
	  .option("project", "translateqna") \
	  .option("parentProject", "translateqna") \
	  .option("credentialsFile", "/opt/spark/key.json") \
	.load("bigquery-public-data.america_health_rankings.ahr")
	
	

df1.write \
	.mode('overwrite') \
	    .option('header', 'true') \
	.csv('gs://translateqna-spark/bigquery_public_data/america_health_rankings/ahr')




# Read from Bigquery, write to Pubsub
df2=spark.read \
	.format('bigquery') \
	  .option("temporaryGcsBucket", "translateqna-spark") \
	  .option("project", "translateqna") \
	  .option("parentProject", "translateqna") \
	  .option("credentialsFile", "/opt/spark/key.json") \
	.load("bigquery-public-data.austin_bikeshare.bikeshare_stations")
	
df2_cached = df2.cache()	
df2_cached.count()


def write_to_pubsub_with_attributes(partition):
    from google.cloud import pubsub_v1
    from google.oauth2 import service_account
    import json
    from datetime import datetime, date
    
    # Custom JSON encoder for datetime objects
    def json_converter(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    # Load credentials from the service account key file
    credentials = service_account.Credentials.from_service_account_file(
        '/opt/spark/key.json'
    )
    
    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path("translateqna", "feedback")
    
    for row in partition:
        # Use custom converter for datetime objects
        message_data = json.dumps(row.asDict(), default=json_converter).encode('utf-8')
        
        # Add custom attributes
        attributes = {
            "source": "spark",
            "timestamp": str(row.timestamp) if hasattr(row, 'timestamp') else ""
        }
        
        future = publisher.publish(topic_path, message_data, **attributes)
        future.result()

df2.foreachPartition(write_to_pubsub_with_attributes)



