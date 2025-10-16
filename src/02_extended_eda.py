# src/02_extended_eda.py
import os, argparse, logging
from pyspark.sql import SparkSession, functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extended-eda")

def parse_args():
    p = argparse.ArgumentParser(description="Generate 10 visualizations from curated data")
    p.add_argument("--curated_dir", default=os.getenv("CURATED_DIR","./outputs/curated"))
    p.add_argument("--plots_dir",   default=os.getenv("PLOTS_DIR","./outputs/plots"))
    p.add_argument("--sample_for_plots", type=int, default=250_000)
    return p.parse_args()

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def to_pandas_sample(df, limit):
    count = df.count()
    if count == 0:
        import pandas as pd
        return pd.DataFrame()
    frac = min(1.0, max(limit / count, 0.0))
    sdf = df.sample(False, frac, seed=42) if frac < 1.0 else df
    return sdf.limit(limit).toPandas()

# ----- plotting helpers (matplotlib, 1 chart per figure) -----
def plot_bar(x, y, title, xlabel, ylabel, out_path, rotate=0):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(x, y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if rotate: plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight")

def plot_hbar(x, y, title, xlabel, ylabel, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.barh(x, y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight")

def plot_line(x, y, title, xlabel, ylabel, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight")

def plot_scatter(x, y, title, xlabel, ylabel, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.6)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight")

def plot_pie(labels, sizes, title, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight")

def main():
    args = parse_args()
    ensure_dir(args.plots_dir)

    spark = (SparkSession.builder
             .appName("Flight-Delay-02-Plots")
             .config("spark.sql.shuffle.partitions", "200")
             .getOrCreate())

    log.info(f"Reading curated data from {args.curated_dir}")
    df = spark.read.parquet(args.curated_dir)

    # derived cols (if missing)
    if "flight_date" in df.columns and "year" not in df.columns:
        df = df.withColumn("year", F.year("flight_date"))
    if "flight_date" in df.columns and "month" not in df.columns:
        df = df.withColumn("month", F.month("flight_date"))
    if "flight_date" in df.columns and "day_of_week" not in df.columns:
        df = df.withColumn("day_of_week", F.date_format("flight_date", "E"))

    plots_done = 0
    out = lambda name: os.path.join(args.plots_dir, name)

    # 1) Airline avg delay (bar)
    if {"airline","arr_delay"}.issubset(df.columns):
        a = (df.groupBy("airline").agg(F.avg("arr_delay").alias("avg_arr_delay"))
                        .orderBy(F.desc("avg_arr_delay")))
        pdf = to_pandas_sample(a, 50)
        if not pdf.empty:
            plot_bar(pdf["airline"], pdf["avg_arr_delay"],
                     "Average Arrival Delay by Airline",
                     "Airline","Avg Delay (min)", out("avg_delay_airline.png"), rotate=45)
            plots_done += 1

    # 2) Monthly avg delay across years (line)
    if {"year","month","arr_delay"}.issubset(df.columns):
        a = (df.groupBy("year","month")
              .agg(F.avg("arr_delay").alias("avg_arr_delay"))
              .orderBy("year","month"))
        pdf = to_pandas_sample(a, 500)
        if not pdf.empty:
            pdf["ym"] = pdf["year"].astype(str)+"-"+pdf["month"].astype(int).astype(str).str.zfill(2)
            plot_line(pdf["ym"], pdf["avg_arr_delay"],
                      "Monthly Average Arrival Delay","Year-Month","Avg Delay (min)",
                      out("monthly_avg_delay.png"))
            plots_done += 1

    # 3) Top routes (hbar)
    if {"origin","dest","arr_delay"}.issubset(df.columns):
        a = (df.groupBy("origin","dest")
              .agg(F.avg("arr_delay").alias("avg_arr_delay"), F.count("*").alias("n"))
              .filter(F.col("n")>500).orderBy(F.desc("avg_arr_delay")).limit(20))
        pdf = to_pandas_sample(a, 20)
        if not pdf.empty:
            pdf["route"] = pdf["origin"] + "→" + pdf["dest"]
            plot_hbar(pdf["route"][::-1], pdf["avg_arr_delay"][::-1],
                      "Top Routes by Avg Arrival Delay (n>500)",
                      "Avg Delay (min)","Route", out("top_routes.png"))
            plots_done += 1

    # 4) Cancellations by code (pie)
    if "cancellation_code" in df.columns:
        a = df.groupBy("cancellation_code").count().orderBy(F.desc("count"))
        pdf = to_pandas_sample(a, 10)
        if not pdf.empty:
            plot_pie(pdf["cancellation_code"], pdf["count"],
                     "Cancellations by Code", out("cancellations_pie.png"))
            plots_done += 1

    # 5) Top origin airports (bar)
    if {"origin","arr_delay"}.issubset(df.columns):
        a = (df.groupBy("origin")
              .agg(F.avg("arr_delay").alias("avg_arr_delay"), F.count("*").alias("n"))
              .filter(F.col("n")>1000).orderBy(F.desc("avg_arr_delay")).limit(15))
        pdf = to_pandas_sample(a, 15)
        if not pdf.empty:
            plot_bar(pdf["origin"], pdf["avg_arr_delay"],
                     "Top 15 Origin Airports by Avg Delay (n>1000)",
                     "Origin","Avg Delay (min)", out("origin_delay.png"), rotate=45)
            plots_done += 1

    # 6) Top destination airports (bar)
    if {"dest","arr_delay"}.issubset(df.columns):
        a = (df.groupBy("dest")
              .agg(F.avg("arr_delay").alias("avg_arr_delay"), F.count("*").alias("n"))
              .filter(F.col("n")>1000).orderBy(F.desc("avg_arr_delay")).limit(15))
        pdf = to_pandas_sample(a, 15)
        if not pdf.empty:
            plot_bar(pdf["dest"], pdf["avg_arr_delay"],
                     "Top 15 Destination Airports by Avg Delay (n>1000)",
                     "Destination","Avg Delay (min)", out("dest_delay.png"), rotate=45)
            plots_done += 1

    # 7) Delay histogram
    if "arr_delay" in df.columns:
        import matplotlib.pyplot as plt
        pdf = to_pandas_sample(df.select("arr_delay").dropna(), args.sample_for_plots)
        if not pdf.empty:
            plt.figure()
            clipped = pdf["arr_delay"].clip(lower=-60, upper=300)
            plt.hist(clipped, bins=50, edgecolor="black")
            plt.title("Arrival Delay Distribution (clipped -60..300)")
            plt.xlabel("Arrival Delay (min)"); plt.ylabel("Frequency")
            plt.tight_layout(); plt.savefig(out("arr_delay_histogram.png"), bbox_inches="tight")
            plots_done += 1

    # 8) Distance vs delay (scatter, with corr in title)
    if {"distance","arr_delay"}.issubset(df.columns):
        corr = float(df.stat.corr("distance","arr_delay"))
        pdf = to_pandas_sample(df.select("distance","arr_delay").dropna(), args.sample_for_plots)
        if not pdf.empty:
            plot_scatter(pdf["distance"], pdf["arr_delay"],
                         f"Distance vs Arrival Delay (corr={corr:.3f})",
                         "Distance (miles)","Arrival Delay (min)",
                         out("distance_vs_delay.png"))
            plots_done += 1

    # 9) Dep→Arr delay propagation by airline (scatter)
    if {"dep_delay","arr_delay","airline"}.issubset(df.columns):
        a = (df.groupBy("airline")
              .agg(F.avg("dep_delay").alias("avg_dep_delay"),
                   F.avg("arr_delay").alias("avg_arr_delay")))
        pdf = to_pandas_sample(a, 60)
        if not pdf.empty:
            plot_scatter(pdf["avg_dep_delay"], pdf["avg_arr_delay"],
                         "Airlines: Avg Departure vs Avg Arrival Delay",
                         "Avg Departure Delay (min)","Avg Arrival Delay (min)",
                         out("dep_vs_arr_scatter.png"))
            plots_done += 1

    # 10) Day-of-week (bar, ordered)
    if {"day_of_week","arr_delay"}.issubset(df.columns):
        a = df.groupBy("day_of_week").agg(F.avg("arr_delay").alias("avg_arr_delay"))
        # order Mon..Sun
        dow = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        pdf = to_pandas_sample(a, 10)
        if not pdf.empty:
            pdf["order"] = pdf["day_of_week"].map({d:i for i,d in enumerate(dow)})
            pdf = pdf.sort_values("order")
            plot_bar(pdf["day_of_week"], pdf["avg_arr_delay"],
                     "Average Arrival Delay by Day of Week",
                     "Day of Week","Avg Delay (min)", out("weekday_delay.png"))
            plots_done += 1

    log.info(f"[OK] Wrote {plots_done} plot(s) to {args.plots_dir}")
    spark.stop()

if __name__ == "__main__":
    main()