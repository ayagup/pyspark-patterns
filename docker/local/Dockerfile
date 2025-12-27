FROM spark:python3-java17


# Install Python packages as root
USER root
RUN pip install --no-cache-dir \
    google-cloud-pubsub \
    google-cloud-datastore \
    google-cloud-bigquery


WORKDIR /opt/spark
