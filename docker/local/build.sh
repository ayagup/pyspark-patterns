docker build -t spark-gcp:latest .


docker run -it \
 -v ~/spark_ivy_cache:/tmp/.ivy2 \
 -v /mnt/c/Users/Lenovo/Downloads/translateqna-b3791e36203a.json:/opt/spark/key.json \
spark-gcp:latest /opt/spark/bin/pyspark  \
  --packages com.google.cloud.spark:spark-bigquery-with-dependencies_2.13:0.43.1,com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.22 \
 --conf "spark.driver.extraJavaOptions=-Divy.cache.dir=/tmp -Divy.home=/tmp"   \
 --conf "spark.executor.extraJavaOptions=-Divy.cache.dir=/tmp -Divy.home=/tmp" \
 --conf "spark.hadoop.google.cloud.auth.service.account.enable=true" \
  --conf "spark.hadoop.google.cloud.auth.service.account.json.key.file=/opt/spark/key.json" \
  --conf "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem"  \
  --conf "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS" \
   --conf "spark.hadoop.fs.gs.auth.service.account.enable=true" \
  --conf "spark.hadoop.fs.gs.auth.service.account.json.keyfile=/opt/spark/key.json"
