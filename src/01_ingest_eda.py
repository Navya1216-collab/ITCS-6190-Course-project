# src/01_ingest_eda.py
import os, glob, logging, shutil
from uuid import uuid4
from dotenv import load_dotenv
from pyspark.sql import SparkSession, functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest-eda")

# ---------------------------
# 1) Env & defaults
# ---------------------------
load_dotenv()
RAW_DATA_GLOB = os.getenv("RAW_DATA_GLOB", "./data/raw/*.csv")
CURATED_DIR   = os.getenv("CURATED_DIR",   "./outputs/curated")
TABLE_DIR     = os.getenv("TABLE_DIR",     "./outputs/tables")  # <— tables here

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(CURATED_DIR, exist_ok=True)

log.info(f"RAW_DATA_GLOB={RAW_DATA_GLOB}")
log.info(f"CURATED_DIR={CURATED_DIR}")
log.info(f"TABLE_DIR={TABLE_DIR}")

# ---------------------------
# 2) Spark
# ---------------------------
spark = (
    SparkSession.builder
    .appName("Flight-Delay-01-Tables")
    .config("spark.sql.files.maxPartitionBytes", "134217728")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

# ---------------------------
# 3) Read CSVs
# ---------------------------
paths = glob.glob(RAW_DATA_GLOB)
if not paths:
    raise FileNotFoundError(f"No CSV files found at {RAW_DATA_GLOB}")
log.info(f"Reading {len(paths)} CSV file(s)")
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(paths)
)

# ---------------------------
# 4) Normalize headers & types
# ---------------------------
rename_map = {
    "FL_DATE": "flight_date",
    "AIRLINE": "airline",
    "OP_CARRIER": "airline",
    "ORIGIN": "origin",
    "DEST": "dest",
    "ARR_DELAY": "arr_delay",
    "DEP_DELAY": "dep_delay",
    "DISTANCE": "distance",
    "CANCELLATION_CODE": "cancellation_code",
    "CRS_DEP_TIME": "crs_dep_time",
    "DEP_TIME": "dep_time",
}
for old, new in rename_map.items():
    if old in df.columns and new not in df.columns:
        df = df.withColumnRenamed(old, new)

# dates
if "flight_date" in df.columns:
    patterns = ["yyyy-MM-dd", "M/d/yyyy", "MM/dd/yyyy"]
    expr = None
    for p in patterns:
        parsed = F.to_date("flight_date", p)
        expr = parsed if expr is None else F.coalesce(expr, parsed)
    df = df.withColumn("flight_date", expr)
    df = df.withColumn("year", F.year("flight_date")).withColumn("month", F.month("flight_date"))

# numeric casts
for c in ["arr_delay", "dep_delay", "distance"]:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("double"))

# trims
for c in ["airline", "origin", "dest", "cancellation_code"]:
    if c in df.columns:
        df = df.withColumn(c, F.trim(F.col(c)))

# hour-of-day (if we have scheduled departure as HHMM)
if "crs_dep_time" in df.columns:
    # if col is string like "0550" or int 550, convert to hour = floor(val/100)
    df = df.withColumn("dep_hour", (F.col("crs_dep_time").cast("int")/100).cast("int"))

# weekday label
if "flight_date" in df.columns:
    df = df.withColumn("day_of_week", F.date_format("flight_date", "E"))

# basic cleaning
required = [c for c in ["flight_date","airline","origin","dest","arr_delay"] if c in df.columns]
if required:
    df = df.dropna(subset=required)
df = df.dropDuplicates()

# ---------------------------
# helper: write ONE csv file
# ---------------------------
def write_single_csv(dframe, out_csv_path: str):
    tmp_dir = f"{out_csv_path}.tmp_{uuid4().hex}"
    (dframe.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir))
    # find the part file and move/rename to .csv
    part = None
    for f in os.listdir(tmp_dir):
        if f.startswith("part-") and f.endswith(".csv"):
            part = os.path.join(tmp_dir, f)
            break
    if part is None:
        # spark sometimes writes .gz—handle that too
        for f in os.listdir(tmp_dir):
            if f.startswith("part-") and (f.endswith(".csv.gz") or f.endswith(".csv")):
                part = os.path.join(tmp_dir, f)
                break
    if part is None:
        raise RuntimeError(f"No CSV part file written under {tmp_dir}")
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    shutil.move(part, out_csv_path)
    shutil.rmtree(tmp_dir)

def save_table(name, dframe):
    csv_path = os.path.join(TABLE_DIR, f"{name}.csv")
    write_single_csv(dframe, csv_path)
    log.info(f"[OK] Table saved: {csv_path}")

# ---------------------------
# 5) TEN analyses → CSVs
# ---------------------------

tables_done = 0

# 1) Average arrival delay by airline
if {"airline","arr_delay"}.issubset(df.columns):
    a1 = (df.groupBy("airline")
            .agg(F.avg("arr_delay").alias("avg_arr_delay"),
                 F.count("*").alias("n"))
            .orderBy(F.desc("avg_arr_delay")))
    save_table("by_airline", a1); tables_done += 1

# 2) Monthly average delay (year, month)
if {"year","month","arr_delay"}.issubset(df.columns):
    a2 = (df.groupBy("year","month")
            .agg(F.avg("arr_delay").alias("avg_arr_delay"),
                 F.count("*").alias("n"))
            .orderBy("year","month"))
    save_table("by_month", a2); tables_done += 1

# 3) Most delayed routes (n>500)
if {"origin","dest","arr_delay"}.issubset(df.columns):
    a3 = (df.groupBy("origin","dest")
            .agg(F.avg("arr_delay").alias("avg_arr_delay"),
                 F.count("*").alias("n"))
            .filter(F.col("n") > 500)
            .orderBy(F.desc("avg_arr_delay")))
    save_table("routes_most_delayed", a3); tables_done += 1

# 4) Cancellations by code
if "cancellation_code" in df.columns:
    a4 = (df.groupBy("cancellation_code").count().orderBy(F.desc("count")))
    save_table("cancellations_by_code", a4); tables_done += 1

# 5) Top 15 origin airports by avg delay (n>1000)
if {"origin","arr_delay"}.issubset(df.columns):
    a5 = (df.groupBy("origin")
            .agg(F.avg("arr_delay").alias("avg_arr_delay"), F.count("*").alias("n"))
            .filter(F.col("n") > 1000)
            .orderBy(F.desc("avg_arr_delay"))
            .limit(15))
    save_table("origin_top_delay", a5); tables_done += 1

# 6) Top 15 destination airports by avg delay (n>1000)
if {"dest","arr_delay"}.issubset(df.columns):
    a6 = (df.groupBy("dest")
            .agg(F.avg("arr_delay").alias("avg_arr_delay"), F.count("*").alias("n"))
            .filter(F.col("n") > 1000)
            .orderBy(F.desc("avg_arr_delay"))
            .limit(15))
    save_table("dest_top_delay", a6); tables_done += 1

# 7) Arrival delay distribution summary (count/mean/stddev/min/25/50/75/max)
if "arr_delay" in df.columns:
    a7 = df.select("arr_delay").summary("count","mean","stddev","min","25%","50%","75%","max")
    save_table("arr_delay_summary", a7); tables_done += 1

# 8) Distance vs delay correlation (single row)
if {"distance","arr_delay"}.issubset(df.columns):
    corr = float(df.stat.corr("distance","arr_delay"))
    a8 = spark.createDataFrame([(corr,)], ["distance_arr_delay_corr"])
    save_table("distance_delay_correlation", a8); tables_done += 1

# 9) Departure vs Arrival delay by airline
if {"dep_delay","arr_delay","airline"}.issubset(df.columns):
    a9 = (df.groupBy("airline")
            .agg(F.avg("dep_delay").alias("avg_dep_delay"),
                 F.avg("arr_delay").alias("avg_arr_delay"),
                 F.count("*").alias("n"))
            .orderBy(F.desc("avg_arr_delay")))
    save_table("dep_vs_arr_by_airline", a9); tables_done += 1

# 10) Day-of-week trends
if {"day_of_week","arr_delay"}.issubset(df.columns):
    a10 = (df.groupBy("day_of_week")
            .agg(F.avg("arr_delay").alias("avg_arr_delay"),
                 F.count("*").alias("n")))
    save_table("by_day_of_week", a10); tables_done += 1

log.info(f"[OK] Wrote {tables_done} analysis table(s) to {TABLE_DIR}")

# ---------------------------
# 6) Curated Parquet (partitioned)
# ---------------------------
writer = df.write.mode("overwrite")
if {"year","month"}.issubset(df.columns):
    writer = writer.partitionBy("year","month")
writer.parquet(CURATED_DIR)
log.info(f"[OK] Curated parquet → {CURATED_DIR}")

spark.stop()