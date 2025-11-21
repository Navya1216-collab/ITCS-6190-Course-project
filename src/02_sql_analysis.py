# src/02_sql_analysis.py
#
# Spark SQL analysis on curated flight delays dataset.
# Produces several tables under outputs/tables/ using spark.sql(...).

import os
import shutil
from pyspark.sql import SparkSession

CURATED_DIR = os.getenv("CURATED_DIR", "./outputs/curated")
TABLES_DIR = "./outputs/tables"

os.makedirs(TABLES_DIR, exist_ok=True)


def write_single_csv(df, out_csv_path: str):
    """
    Write a Spark DF to a single CSV file with header.
    Uses a temp dir and moves the part file.
    """
    tmp_dir = out_csv_path + ".tmp"
    (df.coalesce(1)
       .write
       .mode("overwrite")
       .option("header", True)
       .csv(tmp_dir))

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
    spark = (
        SparkSession.builder
            .appName("FlightDelay-SQL-Analysis")
            .config("spark.sql.shuffle.partitions", "200")
            .getOrCreate()
    )

    print(f"[INFO] Reading curated data from {CURATED_DIR}")
    df = spark.read.parquet(CURATED_DIR)

    # Register temp view for SQL
    df.createOrReplaceTempView("flights")

    # 1) Avg arrival delay by airline
    by_airline_sql = """
        SELECT
            airline,
            COUNT(*)                AS num_flights,
            AVG(arr_delay)          AS avg_arr_delay,
            AVG(dep_delay)          AS avg_dep_delay
        FROM flights
        GROUP BY airline
        ORDER BY avg_arr_delay DESC
    """
    by_airline = spark.sql(by_airline_sql)
    write_single_csv(by_airline, os.path.join(TABLES_DIR, "by_airline_sql.csv"))
    print("[OK] Wrote by_airline_sql.csv")

    # 2) Monthly delay trend (year-month)
    by_month_sql = """
        SELECT
            year,
            month,
            COUNT(*)       AS num_flights,
            AVG(arr_delay) AS avg_arr_delay
        FROM flights
        GROUP BY year, month
        ORDER BY year, month
    """
    by_month = spark.sql(by_month_sql)
    write_single_csv(by_month, os.path.join(TABLES_DIR, "by_month_sql.csv"))
    print("[OK] Wrote by_month_sql.csv")

    # 3) Top 20 most delayed routes (origin-dest) by avg arrival delay
    by_route_sql = """
        SELECT
            origin,
            dest,
            COUNT(*)       AS num_flights,
            AVG(arr_delay) AS avg_arr_delay
        FROM flights
        GROUP BY origin, dest
        HAVING COUNT(*) >= 500
        ORDER BY avg_arr_delay DESC
        LIMIT 20
    """
    by_route = spark.sql(by_route_sql)
    write_single_csv(by_route, os.path.join(TABLES_DIR, "by_route_sql.csv"))
    print("[OK] Wrote by_route_sql.csv")

    # 4) Best and worst airports by on-time performance
    #    (share of flights with arr_delay <= 15 minutes)
    airport_otp_sql = """
        WITH per_airport AS (
            SELECT
                origin AS airport,
                COUNT(*) AS num_flights,
                AVG(CASE WHEN arr_delay <= 15 THEN 1.0 ELSE 0.0 END) AS on_time_rate
            FROM flights
            GROUP BY origin
        )
        SELECT
            airport,
            num_flights,
            on_time_rate
        FROM per_airport
        WHERE num_flights >= 1000
        ORDER BY on_time_rate DESC
    """
    airport_otp = spark.sql(airport_otp_sql)
    write_single_csv(airport_otp, os.path.join(TABLES_DIR, "airport_on_time_sql.csv"))
    print("[OK] Wrote airport_on_time_sql.csv")

    spark.stop()
    print("[DONE] Spark SQL analysis complete.")


if __name__ == "__main__":
    main()