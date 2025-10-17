# src/03_predictive_model.py
import os, glob, argparse, logging, shutil
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predictive")

def parse_args():
    p = argparse.ArgumentParser(description="Predictive analysis on flight delays")
    p.add_argument("--task", choices=["classify","regress"], default="classify",
                   help="classify: delayed > threshold; regress: predict arr_delay")
    p.add_argument("--algo", choices=["lr","rf","linreg","rfreg"], default="lr",
                   help="lr=LogisticRegression, rf=RandomForestClassifier, linreg=LinearRegression, rfreg=RandomForestRegressor")
    p.add_argument("--label_threshold", type=float, default=15.0,
                   help="Minutes cutoff for 'delayed' in classification")
    p.add_argument("--tree_max_bins", type=int, default=4096,
                   help="maxBins for tree-based models (rf/rfreg); must exceed highest category count")
    p.add_argument("--curated_dir", default=os.getenv("CURATED_DIR","./outputs/curated"),
                   help="Directory with curated Parquet (preferred)")
    p.add_argument("--raw_glob", default=None,
                   help="Fallback recursive glob for raw CSVs (e.g. ./data/raw/**/*.csv)")
    p.add_argument("--models_dir", default="./outputs/models", help="Where to save models/metrics")
    p.add_argument("--predictions_csv", default="./outputs/models/predictions_sample.csv",
                   help="CSV path for a small predictions sample")
    p.add_argument("--sample_rows", type=int, default=10000, help="Rows to save in predictions sample")
    return p.parse_args()

def read_curated_or_raw(spark, curated_dir, raw_glob):
    # try curated parquet first
    try:
        df = spark.read.parquet(curated_dir)
        _ = df.limit(1).count()
        log.info(f"Loaded curated dataset from {curated_dir}")
        return df
    except Exception:
        log.info(f"No readable Parquet at {curated_dir}")

    if not raw_glob:
        raise FileNotFoundError(
            f"No curated data at {curated_dir}. Provide --raw_glob to read CSVs directly."
        )

    # fallback: read raw CSVs
    paths = glob.glob(raw_glob, recursive=True)
    if not paths:
        raise FileNotFoundError(f"No CSVs matched raw_glob={raw_glob}")
    log.info(f"Reading {len(paths)} raw CSV file(s) from {raw_glob}")

    df = (spark.read.option("header", True).option("inferSchema", True).csv(paths))

    # normalize common headers
    rename_map = {
        "FL_DATE":"flight_date","AIRLINE":"airline","OP_CARRIER":"airline",
        "ORIGIN":"origin","DEST":"dest","ARR_DELAY":"arr_delay","DEP_DELAY":"dep_delay",
        "DISTANCE":"distance"
    }
    for old,new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.withColumnRenamed(old,new)

    # date fields
    if "flight_date" in df.columns:
        expr = None
        for pat in ["yyyy-MM-dd","M/d/yyyy","MM/dd/yyyy"]:
            parsed = F.to_date("flight_date", pat)
            expr = parsed if expr is None else F.coalesce(expr, parsed)
        df = df.withColumn("flight_date", expr)
        df = df.withColumn("year", F.year("flight_date")).withColumn("month", F.month("flight_date"))

    # numeric casts
    for c in ["arr_delay","dep_delay","distance"]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))

    # weekday
    if "flight_date" in df.columns and "day_of_week" not in df.columns:
        df = df.withColumn("day_of_week", F.date_format("flight_date","E"))

    # basic clean
    needed = [c for c in ["arr_delay","airline","origin","dest"] if c in df.columns]
    if needed:
        df = df.dropna(subset=needed)

    return df

def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    spark = (SparkSession.builder
             .appName("Flight-Delay-Predictive")
             .config("spark.sql.shuffle.partitions","200")
             .getOrCreate())

    # 1) load data
    df = read_curated_or_raw(spark, args.curated_dir, args.raw_glob)
    cols = set(df.columns)

    # derive simple features if missing
    if "flight_date" in cols and "year" not in cols:  df = df.withColumn("year", F.year("flight_date"))
    if "flight_date" in cols and "month" not in cols: df = df.withColumn("month", F.month("flight_date"))
    if "flight_date" in cols and "day_of_week" not in cols:
        df = df.withColumn("day_of_week", F.date_format("flight_date","E"))

    # 2) label
    if args.task == "classify":
        if "arr_delay" not in cols:
            raise ValueError("arr_delay needed to create classification label.")
        df = df.withColumn("delayed", F.when(F.col("arr_delay") > args.label_threshold, 1).otherwise(0))
        label_col = "delayed"
    else:
        label_col = "arr_delay"

    # 3) features
    cat_cols = [c for c in ["airline","origin","dest","day_of_week"] if c in df.columns]
    num_cols = [c for c in ["dep_delay","distance","month","year"] if c in df.columns]

    # index categoricals
    for c in cat_cols:
        idx = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        df = idx.fit(df).transform(df)

    if args.algo in ["lr","linreg"]:
        # one-hot for linear models
        enc = OneHotEncoder(
            inputCols=[f"{c}_idx" for c in cat_cols],
            outputCols=[f"{c}_oh" for c in cat_cols],
            handleInvalid="keep"
        )
        df = enc.fit(df).transform(df)
        feat_cols = [f"{c}_oh" for c in cat_cols] + num_cols
    else:
        # trees use indexed features (no one-hot)
        feat_cols = [f"{c}_idx" for c in cat_cols] + num_cols

    df = df.dropna(subset=[label_col])
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")
    data = assembler.transform(df).select("features", label_col)

    train, test = data.randomSplit([0.7, 0.3], seed=42)

    # 4) model train + eval
    metrics_text = ""
    if args.task == "classify":
        if args.algo == "lr":
            model = LogisticRegression(labelCol=label_col, featuresCol="features", maxIter=50).fit(train)
        else:  # rf
            model = RandomForestClassifier(labelCol=label_col, featuresCol="features",
                                           numTrees=80, maxBins=args.tree_max_bins).fit(train)
        pred = model.transform(test)
        auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC").evaluate(pred)
        metrics_text = f"TASK=classify, ALGO={args.algo}, THRESH={args.label_threshold}, ROC-AUC={auc:.4f}\n"
        print("[METRIC]", metrics_text.strip())
    else:
        if args.algo == "linreg":
            model = LinearRegression(labelCol=label_col, featuresCol="features", maxIter=50).fit(train)
        else:  # rfreg
            model = RandomForestRegressor(labelCol=label_col, featuresCol="features",
                                          numTrees=120, maxBins=args.tree_max_bins).fit(train)
        pred = model.transform(test)
        rmse = RegressionEvaluator(labelCol=label_col, metricName="rmse").evaluate(pred)
        r2   = RegressionEvaluator(labelCol=label_col, metricName="r2").evaluate(pred)
        metrics_text = f"TASK=regress, ALGO={args.algo}, RMSE={rmse:.4f}, R2={r2:.4f}\n"
        print("[METRIC]", metrics_text.strip())

    # 5) save model + metrics
    model_dirname = f"flight_delay_{'classify' if args.task=='classify' else 'regress'}_{args.algo}"
    model_path = os.path.join(args.models_dir, model_dirname)
    try: shutil.rmtree(model_path)
    except Exception: pass
    model.save(model_path)

    with open(os.path.join(args.models_dir, "metrics.txt"), "a") as f:
        f.write(metrics_text)

    # 6) optional predictions sample
    try:
        cols_out = [label_col, "prediction"]
        if args.task == "classify":
            cols_out.append("probability")
        (pred.select(*cols_out).limit(args.sample_rows)
             .toPandas()
             .to_csv(args.predictions_csv, index=False))
    except Exception as e:
        log.warning(f"Skipping predictions CSV: {e}")

    spark.stop()
    print(f"[OK] Model saved to {model_path}")
    print(f"[OK] Metrics appended to outputs/models/metrics.txt")
    print(f"[OK] Predictions sample (optional): {args.predictions_csv}")

if __name__ == "__main__":
    main()