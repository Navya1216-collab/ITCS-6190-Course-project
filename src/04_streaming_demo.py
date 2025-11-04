# src/04_streaming_demo.py
# Structured Streaming: Average arrival delay by airline → Parquet snapshots (foreachBatch)

import os
from pyspark.sql import SparkSession, functions as F

# -------- Paths (override via env if you want) --------
RAW_STREAM_DIR = os.getenv("STREAM_DIR", "./data/stream")
STREAM_OUT_DIR = os.getenv("STREAM_OUT_DIR", "./outputs/stream_out/avg_by_airline")
CHECKPOINT_DIR = os.getenv("STREAM_CHECKPOINT", "./outputs/stream_ckpt/avg_by_airline")

os.makedirs(STREAM_OUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------- Spark --------
spark = (
    SparkSession.builder
        .appName("FlightDelayStreaming")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# -------- Read stream (match your CSV header with 5 columns) --------
schema = """
  flight_date STRING,
  airline     STRING,
  origin      STRING,
  dest        STRING,
  arr_delay   DOUBLE
"""

stream_df = (
    spark.readStream
         .option("header", True)          # your batch_xxxx.csv have headers
         .option("maxFilesPerTrigger", 1) # 1 file per micro-batch
         .schema(schema)
         .csv(RAW_STREAM_DIR)
)

# Normalize & keep only the needed fields
arr_delay_col = F.coalesce(F.col("arr_delay"), F.col("ARR_DELAY")).cast("double")
stream_df = (
    stream_df
      .withColumn("airline", F.trim(F.col("airline")))
      .withColumn("arr_delay", arr_delay_col)
      .select("airline", "arr_delay")
      .dropna(subset=["airline", "arr_delay"])
)

# -------- Aggregation --------
agg_df = stream_df.groupBy("airline").agg(F.avg("arr_delay").alias("avg_arr_delay"))

# -------- Sink using foreachBatch (so COMPLETE mode works with files) --------
def write_batch(batch_df, batch_id: int):
    out_dir = os.path.join(STREAM_OUT_DIR, f"batch={batch_id}")
    (batch_df.coalesce(1)
            .write.mode("overwrite")
            .parquet(out_dir))

query = (
    agg_df.writeStream
         .outputMode("complete")                 # allowed with foreachBatch
         .option("checkpointLocation", CHECKPOINT_DIR)
         .trigger(processingTime="10 seconds")   # slightly longer; fewer "falling behind" warns
         .foreachBatch(write_batch)
         .start()
)

print("\n🚀 Streaming started")
print(f"   Watching:            {os.path.abspath(RAW_STREAM_DIR)}")
print(f"   Checkpoint location: {os.path.abspath(CHECKPOINT_DIR)}")
print(f"   Parquet snapshots:   {os.path.abspath(STREAM_OUT_DIR)}\n")

query.awaitTermination()
