from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    LongType,
)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "user-interactions"

OUTPUT_PATH = "data/processed/kafka_interactions"
CHECKPOINT_PATH = "data/processed/.kafka_checkpoint"


schema = StructType([
    StructField("user_idx", IntegerType(), nullable=False),
    StructField("item_idx", IntegerType(), nullable=False),
    StructField("rating", DoubleType(), nullable=False),
    StructField("timestamp", LongType(), nullable=False),
])


spark = (
    SparkSession.builder
    .appName("MLOpsKafkaConsumer")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# Read events from Kafka
# ---------------------------------------------------------

stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "earliest")
    .load()
)


# ---------------------------------------------------------
# Parse JSON payload
# ---------------------------------------------------------

events = (
    stream
    .select(
        from_json(
            col("value").cast("string"),
            schema,
        ).alias("event")
    )
    .select("event.*")
)


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

valid_events = (
    events
    .filter(
        col("user_idx").isNotNull()
        & col("item_idx").isNotNull()
        & col("rating").isNotNull()
        & col("timestamp").isNotNull()
    )
    .withColumn("ingested_at", current_timestamp())
)


# ---------------------------------------------------------
# Persist processed streaming events
# ---------------------------------------------------------

query = (
    valid_events.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", OUTPUT_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(processingTime="5 seconds")
    .start()
)


print(f"Kafka topic      : {TOPIC}")
print(f"Output path      : {OUTPUT_PATH}")
print(f"Checkpoint path  : {CHECKPOINT_PATH}")

query.awaitTermination()