# src/01_ingest_eda.py

import os
import glob
from dotenv import load_dotenv
from pyspark.sql import SparkSession, functions as F

# ---------------------------
# 1) Load environment paths
# ---------------------------
load_dotenv()
RAW_DATA_GLOB = os.getenv("RAW_DATA_GLOB", "./samples/*.csv")
CURATED_DIR   = os.getenv("CURATED_DIR", "./curated")
SAMPLE_DATA_GLOB = os.getenv("SAMPLE_DATA_GLOB", "./samples/*.csv")

print(f"[INFO] RAW_DATA_GLOB={RAW_DATA_GLOB}")
print(f"[INFO] CURATED_DIR={CURATED_DIR}")
print(f"[INFO] SAMPLE_DATA_GLOB={SAMPLE_DATA_GLOB}")

# ---------------------------
# 2) Spark session
# ---------------------------
spark = (
    SparkSession.builder
    .appName("Flight-Delay-EDA")
    .getOrCreate()
)

# ---------------------------
# 3) Resolve input files
# ---------------------------
paths = glob.glob(RAW_DATA_GLOB)
if not paths:
    print(f"[WARN] No files found at RAW_DATA_GLOB. Falling back to SAMPLE_DATA_GLOB: {SAMPLE_DATA_GLOB}")
    paths = glob.glob(SAMPLE_DATA_GLOB)

if not paths:
    raise FileNotFoundError(
        f"No CSV files found at RAW_DATA_GLOB={RAW_DATA_GLOB} or SAMPLE_DATA_GLOB={SAMPLE_DATA_GLOB}"
    )

print(f"[INFO] Reading {len(paths)} file(s)")

# ---------------------------
# 4) Read CSV(s)
# ---------------------------
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(paths)
)

print("Rows, Cols =", df.count(), len(df.columns))
df.printSchema()
df.show(5, truncate=False)

# ------------------------------------------------
# 5) Normalize column names (UPPER -> lower used)
#    Adjust/match your dataset's actual headers.
# ------------------------------------------------
rename_map = {
    "FL_DATE": "flight_date",
    "AIRLINE": "airline",
    "ORIGIN": "origin",
    "DEST": "dest",
    "ARR_DELAY": "arr_delay",
    "DEP_DELAY": "dep_delay",
    "DISTANCE": "distance",
    "CANCELLATION_CODE": "cancellation_code",
}
for old, new in rename_map.items():
    if old in df.columns:
        df = df.withColumnRenamed(old, new)

# Robust date casting if present
if "flight_date" in df.columns:
    df = df.withColumn("flight_date", F.to_date("flight_date"))

# Safe numeric casts only if columns exist
for c, t in [("arr_delay", "double"), ("dep_delay", "double"), ("distance", "double")]:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast(t))

# ---------------------------
# 6) Basic cleaning
# ---------------------------
required = [c for c in ["flight_date", "airline", "origin", "dest", "arr_delay"] if c in df.columns]
if required:
    df = df.dropna(subset=required)
df = df.dropDuplicates()

# ---------------------------
# 7) Quick data-quality check
# ---------------------------
nulls = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
print("[INFO] Null counts per column:")
nulls.show(truncate=False)

# ---------------------------
# 8) EDA
# ---------------------------

# 8.1 Avg arrival delay by airline
if "airline" in df.columns and "arr_delay" in df.columns:
    print("[EDA] Average arrival delay by airline:")
    (df.groupBy("airline")
       .agg(F.avg("arr_delay").alias("avg_arr_delay"),
            F.count("*").alias("n"))
       .orderBy(F.desc("avg_arr_delay"))
       .show(20, truncate=False))
else:
    print("[EDA] Skipped airline avg delay (missing columns).")

# 8.2 Seasonal pattern by month
if "flight_date" in df.columns and "arr_delay" in df.columns:
    print("[EDA] Monthly average arrival delay:")
    eda_month = (df
        .withColumn("month", F.month("flight_date"))
        .groupBy("month")
        .agg(F.avg("arr_delay").alias("avg_arr_delay"),
             F.count("*").alias("n"))
        .orderBy("month"))
    eda_month.show(12, truncate=False)
else:
    eda_month = None
    print("[EDA] Skipped monthly trend (missing columns).")

# 8.3 Most delayed routes (support > 500)
if all(c in df.columns for c in ["origin", "dest", "arr_delay"]):
    print("[EDA] Most delayed routes (n > 500):")
    (df.groupBy("origin","dest")
       .agg(F.avg("arr_delay").alias("avg_arr_delay"),
            F.count("*").alias("n"))
       .filter(F.col("n") > 500)
       .orderBy(F.desc("avg_arr_delay"))
       .show(20, truncate=False))
else:
    print("[EDA] Skipped route analysis (missing columns).")

# 8.4 Cancellations breakdown
if "cancellation_code" in df.columns:
    print("[EDA] Cancellations by code:")
    (df.groupBy("cancellation_code")
       .count()
       .orderBy(F.desc("count"))
       .show(truncate=False))

# ---------------------------
# 9) Write curated Parquet
# ---------------------------
partition_cols = ["flight_date"] if "flight_date" in df.columns else []
writer = df.write.mode("overwrite")
if partition_cols:
    writer = writer.partitionBy(*partition_cols)

writer.parquet(CURATED_DIR)
print(f"[OK] Curated parquet written to: {CURATED_DIR}")

# ---------------------------
# 10) Optional: save seasonal plot
# ---------------------------
if eda_month is not None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        pdf = eda_month.toPandas()
        if len(pdf) > 0:
            os.makedirs("docs/assets", exist_ok=True)
            plt.figure()
            plt.plot(pdf["month"], pdf["avg_arr_delay"])
            plt.xlabel("Month")
            plt.ylabel("Avg Arrival Delay (min)")
            plt.title("Seasonal Delays (All Flights)")
            out_path = "docs/assets/seasonal_delays.png"
            plt.savefig(out_path, bbox_inches="tight")
            print(f"[OK] Saved plot: {out_path}")
    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")

spark.stop()
