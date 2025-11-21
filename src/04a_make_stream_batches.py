# src/04a_make_stream_batches.py
import os, shutil, argparse
from uuid import uuid4
from pyspark.sql import SparkSession, functions as F, Window

def write_single_csv(dframe, out_csv_path: str):
    tmp_dir = f"{out_csv_path}.tmp_{uuid4().hex}"
    (dframe.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir))
    part = None
    for f in os.listdir(tmp_dir):
        if f.startswith("part-") and (f.endswith(".csv") or f.endswith(".csv.gz")):
            part = os.path.join(tmp_dir, f)
            break
    if not part:
        raise RuntimeError(f"No CSV part file written under {tmp_dir}")
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    shutil.move(part, out_csv_path)
    shutil.rmtree(tmp_dir)

def main():
    ap = argparse.ArgumentParser(description="Chunk last N months into stream micro-batches")
    ap.add_argument("--curated_dir", default=os.getenv("CURATED_DIR","./outputs/curated"))
    ap.add_argument("--stream_dir",  default=os.getenv("STREAM_DIR","./data/stream"))
    ap.add_argument("--months", type=int, default=3, help="How many most-recent months to include")
    ap.add_argument("--rows_per_file", type=int, default=20000, help="Rows per micro-batch CSV")
    args = ap.parse_args()

    os.makedirs(args.stream_dir, exist_ok=True)

    spark = (SparkSession.builder
             .appName("Make-Stream-Batches")
             .config("spark.sql.shuffle.partitions","200")
             .getOrCreate())

    df = spark.read.parquet(args.curated_dir)

    # Ensure flight_date exists as date; derive year/month if needed
    if "flight_date" not in df.columns:
        raise ValueError("curated data must contain 'flight_date' column")
    df = df.withColumn("flight_date", F.to_date("flight_date"))
    df = df.filter(F.col("flight_date").isNotNull())

    if "year" not in df.columns:
        df = df.withColumn("year", F.year("flight_date"))
    if "month" not in df.columns:
        df = df.withColumn("month", F.month("flight_date"))
    if "day_of_week" not in df.columns:
        df = df.withColumn("day_of_week", F.date_format("flight_date", "E"))

    # find the most recent date and back off N months
    max_date = df.agg(F.max("flight_date").alias("maxd")).collect()[0]["maxd"]
    if not max_date:
        raise ValueError("No rows with a valid flight_date in curated data.")

    # keep only last N months
    start_cut = F.add_months(F.lit(max_date), -args.months + 1)  # inclusive current month and N-1 back
    df_recent = df.filter(F.trunc("flight_date","month") >= F.trunc(start_cut, "month"))

    # keep all columns needed by the model
    keep_cols = [
        c for c in [
            "flight_date", "airline", "origin", "dest",
            "arr_delay", "dep_delay", "distance",
            "month", "year", "day_of_week"
        ]
        if c in df_recent.columns
    ]
    df_recent = df_recent.select(*keep_cols)

    # assign batch ids by flight_date order, fixed rows per file
    w = Window.orderBy(F.col("flight_date").asc())
    df_numbered = df_recent.withColumn("rn", F.row_number().over(w))
    df_batched  = df_numbered.withColumn("batch_id", ((F.col("rn")-1)/F.lit(args.rows_per_file)).cast("int"))

    # write each batch_id to a single CSV file: batch_0001.csv, batch_0002.csv, ...
    batch_ids = [r["batch_id"] for r in df_batched.select("batch_id").distinct().orderBy("batch_id").collect()]
    for b in batch_ids:
        out_path = os.path.join(args.stream_dir, f"batch_{b+1:04d}.csv")
        write_single_csv(df_batched.filter(F.col("batch_id")==b).drop("rn","batch_id"), out_path)
        print(f"[OK] Wrote {out_path}")

    spark.stop()
    print(f"✅ Done. {len(batch_ids)} batch file(s) created in {args.stream_dir}")

if __name__ == "__main__":
    main()