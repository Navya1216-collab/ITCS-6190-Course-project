# src/04_stream_predict.py
#
# Structured Streaming + ML:
#   Load saved PipelineModel and score each incoming micro-batch.

import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel

RAW_STREAM_DIR   = os.getenv("STREAM_DIR", "./data/stream")
STREAM_OUT_DIR   = os.getenv("STREAM_OUT_DIR", "./outputs/stream_out/predictions")
CHECKPOINT_DIR   = os.getenv("STREAM_CHECKPOINT", "./outputs/stream_ckpt/predictions")
MODELS_DIR       = os.getenv("MODELS_DIR", "./outputs/models")
MODEL_SUBDIR     = os.getenv("MODEL_SUBDIR", "flight_delay_classify_lr")
MODEL_PATH       = os.path.join(MODELS_DIR, MODEL_SUBDIR)

os.makedirs(STREAM_OUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
        .appName("FlightDelayStreamingWithML")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"[INFO] Loading pipeline model from {MODEL_PATH}")
pipeline_model = PipelineModel.load(MODEL_PATH)

# Schema must match the columns we wrote in 04a_make_stream_batches.py
schema = """
  flight_date STRING,
  airline     STRING,
  origin      STRING,
  dest        STRING,
  arr_delay   DOUBLE,
  dep_delay   DOUBLE,
  distance    DOUBLE,
  month       INT,
  year        INT,
  day_of_week STRING
"""

stream_df = (
    spark.readStream
         .option("header", True)
         .option("maxFilesPerTrigger", 1)
         .schema(schema)
         .csv(RAW_STREAM_DIR)
)

# Derive any missing columns to match training logic
stream_df = (
    stream_df
      .withColumn("flight_date", F.to_date("flight_date"))
      .withColumn("year", F.when(F.col("year").isNull(), F.year("flight_date")).otherwise(F.col("year")))
      .withColumn("month", F.when(F.col("month").isNull(), F.month("flight_date")).otherwise(F.col("month")))
      .withColumn(
          "day_of_week",
          F.when(F.col("day_of_week").isNull(), F.date_format("flight_date", "E"))
           .otherwise(F.col("day_of_week"))
      )
)


def score_batch(batch_df, batch_id: int):
    """foreachBatch function:
      - run the PipelineModel to get prediction & probability
      - write predictions to Parquet
    """
    if batch_df.rdd.isEmpty():
        return

    pred_df = pipeline_model.transform(batch_df)

    out_df = (
        pred_df
          .select(
              "flight_date", "airline", "origin", "dest",
              "dep_delay", "distance", "arr_delay",
              F.col("prediction").alias("pred_delayed"),
              F.col("probability").alias("pred_prob_vec")
          )
    )

    out_path = os.path.join(STREAM_OUT_DIR, f"batch={batch_id}")
    (out_df
         .coalesce(1)
         .write
         .mode("overwrite")
         .parquet(out_path))

    print(f"\n[STREAM] Batch {batch_id} predictions written → {out_path}")


query = (
    stream_df.writeStream
             .outputMode("append")  
             .option("checkpointLocation", CHECKPOINT_DIR)
             .foreachBatch(score_batch)
             .start()
)

print("\n🚀 Streaming with ML predictions started")
print(f"   Watching:  {os.path.abspath(RAW_STREAM_DIR)}")
print(f"   Writing:   {os.path.abspath(STREAM_OUT_DIR)}\n")

query.awaitTermination()