# ITCS-6190: Flight Delay and Cancellation Analysis (2019–2023)

# GROUP 5
| Name	                 | Student ID  |
|------------------------|-------------|
| Navya Reddy Thadisana	 | 801425759   |
| Poojitha Jayareddygari | 801426875   |
| Sahit Ceeka	           | 801424751   |
| Sai Kiran Jagini	     | 801484665   |
| Jeevith Gowda	         | 801455831   |

# Project Overview

This project explores and predicts U.S. Flight Delays and Cancellations (2019–2023) using Apache Spark.
It covers the full data pipeline from data ingestion → cleaning → analysis → predictive modeling, and produces visualizations, tables, and model evaluation metrics.

We analyzed over 3 million flight records across multiple U.S. airlines to understand:

* What causes delays and cancellations?
* Which airlines and routes are most affected?
* Can we predict flight delays using machine learning?

# Dataset

* Source: Kaggle - Flight Delay and Cancellation Dataset (2019–2023)
* Period Covered: 2019–2023 (5 years)
* Size: Millions of rows (multi-GB CSVs)

## Key Features

| Column	              | Description                          |
|-----------------------|--------------------------------------|
| FL_DATE               |	Flight Date                          |
| OP_UNIQUE_CARRIER     |	Airline Carrier Code                 |
| ORIGIN, DEST	        | Departure and Destination Airports   |
| DEP_DELAY, ARR_DELAY  |	Delay in minutes (Departure/Arrival) |
| CANCELLED, DIVERTED	  | Status Flags                         |
| CANCELLATION_CODE	    | Reason for Cancellation              |
| DISTANCE	            | Flight Distance in Miles             |


# Environment Setup

## Create Virtual Environment

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install pyspark pandas matplotlib seaborn python-dotenv
```

## Environment Variables (.env)

```bash
RAW_DATA_GLOB=./data/raw/flights_sample_3m.csv
```
```bash
CURATED_DIR=./outputs/curated
```
```bash
MODELS_DIR=./outputs/models
```

## Run Scripts

```bash
# Data ingestion + EDA
./run.sh

# Predictive modeling
./run_predict.sh
```

# Project Structure
```java
ITCS-6190-Course-project/
├── data/
│   ├── raw/
│   └── curated/
├── outputs/
│   ├── tables/        ← Aggregated EDA results (.csv)
│   ├── plots/         ← Visual outputs (.png)
│   ├── models/        ← Model artifacts & metrics
├── src/
│   ├── 01_ingest_eda.py
│   ├── 02_extended_eda.py
│   └── 03_predictive_model.py
├── docs/
│   └── assets/
├── run.sh
├── run_predict.sh
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

# Data Ingestion & Cleaning

## Goals
* Load large CSV files into Spark DataFrames
* Clean inconsistent values and handle missing data
* Store optimized Parquet outputs

## Code Summary
```python
df = (spark.read.option("header", True).option("inferSchema", True).csv(RAW_DATA_GLOB))
df = df.withColumnRenamed("FL_DATE", "flight_date").withColumn("arr_delay", F.col("ARR_DELAY").cast("double"))
df = df.dropna(subset=["arr_delay", "dep_delay"])
df.write.mode("overwrite").parquet(CURATED_DIR)
```

Output: Curated dataset saved in outputs/curated

# Exploratory Data Analysis (EDA)

## Complex Queries (Spark SQL)

Below are representative queries we ran over the curated flights table (`flights_curated`).

**1) Multi-level aggregation with filtering & HAVING**
```sql
-- Average arrival delay by route & month, only busy routes (N > 100)
SELECT
  ORIGIN, DEST,
  month(flight_date) AS month,
  COUNT(*) AS flights,
  ROUND(AVG(ARR_DELAY), 2) AS avg_arr_delay
FROM flights_curated
WHERE CANCELLED = 0 AND ARR_DELAY IS NOT NULL
GROUP BY ORIGIN, DEST, month(flight_date)
HAVING COUNT(*) > 100
ORDER BY avg_arr_delay DESC
LIMIT 20;

**2) Window function: rank worst routes by airline**
WITH route_stats AS (
  SELECT
    OP_UNIQUE_CARRIER AS airline,
    ORIGIN, DEST,
    COUNT(*) AS n,
    AVG(ARR_DELAY) AS avg_arr_delay
  FROM flights_curated
  WHERE ARR_DELAY IS NOT NULL AND CANCELLED = 0
  GROUP BY OP_UNIQUE_CARRIER, ORIGIN, DEST
)
SELECT *
FROM (
  SELECT
    airline, ORIGIN, DEST, n, ROUND(avg_arr_delay,2) AS avg_arr_delay,
    DENSE_RANK() OVER (PARTITION BY airline ORDER BY avg_arr_delay DESC) AS rnk
  FROM route_stats
  WHERE n > 200
) t
WHERE rnk <= 3
ORDER BY airline, rnk;

**3) Join + time bucketing (by year/month)**
-- Year-month on-time performance by airline
SELECT
  OP_UNIQUE_CARRIER AS airline,
  date_format(flight_date, 'yyyy-MM') AS ym,
  COUNT(*) AS flights,
  ROUND(AVG(CASE WHEN ARR_DELAY <= 0 THEN 1 ELSE 0 END), 3) AS on_time_rate
FROM flights_curated
GROUP BY OP_UNIQUE_CARRIER, date_format(flight_date, 'yyyy-MM')
ORDER BY ym, airline;


We performed detailed EDA to uncover trends, seasonality, and performance patterns.

## Average Arrival Delay by Airline
| Airline	          | Avg Delay (min)	| Flights |
|-------------------|-----------------|---------|
| Allegiant Air	    | 13.28	          | 50,179  |
| JetBlue Airways	  | 12.28	          | 109,447 |
| Frontier Airlines	| 11.10	          | 62,711  |
| ExpressJet (aha!)	| 10.03	          | 17,951  |
| Spirit Air Lines	| 8.03	          | 93,200  |

Insight: Low-cost carriers show higher delay rates than full-service airlines like Delta.


## Monthly Delay Patterns
| Month |	Avg Delay (min) |	Flights |
|-------|-----------------|---------|
| 1	    | 2.19	          | 260,785 |
| 6	    | 10.06	          | 254,998 |
| 7	    | 9.49	          | 278,911 |
| 8	    | 6.45	          | 280,603 |
| 12	  | 6.67	          | 209,504 |

Insight: Delay spikes in summer (June–August) and holiday season (December).


## Most Delayed Routes (n > 500)
| Origin | Dest |	Avg Delay |	Flights |
|--------|------|-----------|---------|
| DEN	   | ASE	| 21.25	    | 910     |
| PNS	   | DFW	| 19.36	    | 721     |
| MCO	   | JFK	| 18.57	    | 1910    |
| FLL	   | JFK	| 18.30     |	1710    |
| DFW	   | HOU	| 18.28     |	947     |

Insight: Hub-to-hub routes (Denver, Dallas, Orlando) are prone to congestion.


## Delay Correlation

Observation:
* dep_delay and arr_delay have a strong correlation (ρ = 0.95)
* Distance has negligible correlation with delays


## Cancellation Reasons
| Reason Code |	Meaning           |	Percentage |
|-------------|-------------------|------------|
| A	          | Carrier Delay     |	28%        |
| B	          | Weather	          | 34%        |
| C	          | NAS (Air Traffic) |	22%        |
| D	          | Security          |	16%        |

Insight: Weather and NAS delays account for over 50% of cancellations.

## Additional EDA Tables
| File                           | Description                               |
|--------------------------------|-------------------------------------------|
| arr_delay_summary.csv          |	Summary statistics of delay distribution |
| by_airline.csv                 |	Average delays grouped by carrier        |
| by_day_of_week.csv             |	Delay trends by weekday                  |
| by_month.csv                   |	Monthly average delays                   |
| dest_top_delay.csv             |	Destinations with highest average delay  |
| routes_most_delayed.csv        |	Route-level performance                  |
| distance_delay_correlation.csv |	Delay vs Distance correlations           |

# Streaming Setup (Simulation)

We simulate “flight update events” in real time using a small Python generator that streams JSON over a socket (localhost:9999). Spark Structured Streaming ingests these events, parses them, and performs rolling window aggregations of delays.

**Flow:** `data_generator_stream.py` → (socket:9999) → `spark_streaming_flights.py` → console + CSV sink

**Why socket?** It’s lightweight and reproducible in Codespaces; we can later swap in Kafka.

# Generator (Terminal 1)

# Create streaming/data_generator_stream.py:
import json, socket, time, random, uuid
from datetime import datetime, timedelta

HOST, PORT = "localhost", 9999

def sample_event():
    airlines = ["AA","DL","UA","WN","B6","NK","F9","AS"]
    origins  = ["ATL","DFW","DEN","ORD","LAX","JFK","MCO","SEA"]
    dests    = ["LGA","IAH","SFO","MIA","PHX","CLT","BOS","MSP"]
    dep = random.randint(-5, 60)      # departure delay in minutes
    arr = dep + random.randint(-5, 20) # arrival delay correlated with dep
    return {
        "flight_id": str(uuid.uuid4()),
        "flight_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "airline": random.choice(airlines),
        "origin": random.choice(origins),
        "dest": random.choice(dests),
        "dep_delay": dep,
        "arr_delay": arr,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[generator] listening on {HOST}:{PORT} ...")
    conn, addr = s.accept()
    with conn:
        print(f"[generator] client connected: {addr}")
        while True:
            evt = sample_event()
            conn.sendall((json.dumps(evt) + "\n").encode("utf-8"))
            time.sleep(0.5)  # ~2 events/sec

# Spark Streaming job (Terminal 2)
# Create streaming/spark_streaming_flights.py:
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, avg, sum as _sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

HOST, PORT = "localhost", 9999

spark = (SparkSession.builder.appName("FlightsStreaming").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("flight_id", StringType(), True),
    StructField("flight_date", StringType(), True),
    StructField("airline", StringType(), True),
    StructField("origin", StringType(), True),
    StructField("dest", StringType(), True),
    StructField("dep_delay", IntegerType(), True),
    StructField("arr_delay", IntegerType(), True),
    StructField("timestamp", StringType(), True),
])

raw = (spark.readStream
       .format("socket")
       .option("host", HOST).option("port", PORT)
       .load())

parsed = raw.select(from_json(col("value"), schema).alias("j")).select("j.*") \
            .withColumn("event_time", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")) \
            .withWatermark("event_time", "1 minute")

# Windowed KPIs: rolling 1-min window, slide 30s (fast demo)
kpis = (parsed
        .groupBy(window(col("event_time"), "1 minute", "30 seconds"),
                 col("airline"))
        .agg(_sum("arr_delay").alias("sum_arr_delay"),
             avg("arr_delay").alias("avg_arr_delay"))
        .select(col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("airline"), col("sum_arr_delay"), col("avg_arr_delay")))

# Console sink (for demo) + File sink (evidence)
console_q = (kpis.writeStream
             .outputMode("append")
             .format("console")
             .option("truncate","false")
             .start())

file_q = (kpis.writeStream
          .outputMode("append")
          .option("header","true")
          .option("checkpointLocation","checkpoints/stream_kpis")
          .format("csv")
          .start("outputs/stream_kpis"))

console_q.awaitTermination()
file_q.awaitTermination()

## Streaming Demo & Observations

We streamed synthetic flight events at ~2 events/sec. Spark computed rolling 1-minute windows (slide 30s) with airline-level sum/avg arrival delay. Example console output:

+-------------------+-------------------+-------+-------------+-----------+
|window_start |window_end |airline|sum_arr_delay|avg_arr_delay|
+-------------------+-------------------+-------+-------------+-----------+
|2025-10-21 01:23:00|2025-10-21 01:24:00|DL | 112| 9.33|
|2025-10-21 01:23:30|2025-10-21 01:24:30|B6 | 87| 7.25|

**Early observations:**
- Airlines with higher simulated dep_delay quickly show higher windowed arr_delay.
- Sliding windows provide smoother trend visibility than single micro-batches.

# Predictive Modeling

Objective
Predict flight delay likelihood and delay duration using Spark MLlib.

## Algorithms Implemented
| Type	          | Model	                  | Purpose                 |
|-----------------|-------------------------|-------------------------|
| Classification  |	Logistic Regression     |	Predict delayed/on-time |
| Classification  |	Random Forest	          | Feature importance      |
| Regression      |	Linear Regression       |	Predict delay minutes   |
| Regression      |	Random Forest Regressor |	Nonlinear regression    |

# Commands Used
### Logistic Regression
```
python src/03_predictive_model.py --task classify --algo lr --curated_dir ./outputs/curated --models_dir ./outputs/models
```
### Random Forest Classifier
```
python src/03_predictive_model.py --task classify --algo rf --tree_max_bins 4096 --curated_dir ./outputs/curated --models_dir ./outputs/models
```
### Linear Regression
```
python src/03_predictive_model.py --task regress --algo linreg --curated_dir ./outputs/curated --models_dir ./outputs/models
```


# Model Outputs
| File                       |	Description                  |
|----------------------------|-------------------------------|
| metrics.txt                |	Evaluation metrics           |
| predictions_sample.csv     |	True vs Predicted samples    |
| feature_importances_rf.csv | Random Forest feature weights |
| confusion_matrix.csv       |	Classification performance   |
| model_card.json	           | Metadata summary              |
| roc_curve_lr.png           |	ROC visualization            |


# Sample: predictions_sample.csv
| flight_id |	true_delay | predicted_delay | probability |
|-----------|------------|-----------------|-------------|
| 001       |	0	         | 0	             | 0.04        |
| 002       |	1          | 1	             | 0.91        |
| 003       |	0          | 0	             | 0.13        |
| 004       |	1          | 1	             | 0.87        |


# Model Performance

Logistic Regression
| Metric           | Value  |
|------------------|--------|
| ROC-AUC	         | 0.9334 |
| Accuracy         | 89%    |
| Train/Test Split | 70/30  |

Linear Regression
| Metric | Value |
|--------|-------|
| RMSE   | 8.26  |
| R²     | 0.78  |

Feature Importance (Random Forest)
| Feature    | Importance |
|------------|------------|
| dep_delay  |	0.82      |
| distance   |	0.07      |
| month      |	0.06      |
| origin_idx |	0.03      |
| dest_idx   |	0.02      |


# Visualization Samples
Interpretation: Logistic Regression shows strong discrimination (AUC > 0.93).
Departure delay dominates prediction, followed by month and distance.


# Errors & Fixes
| Error                             |	Cause	                        | Solution                         |
|-----------------------------------|-------------------------------|----------------------------------|
| No readable Parquet               |	Curated folder missing        |	Added fallback CSV reader        |
| event not found in .gitignore     |	Bash parsing issue            |	Quoted exclamation marks         |
| maxBins < categorical cardinality |	RandomForest feature overflow |	Added --tree_max_bins 4096       |
| kaggle: command not found         |	CLI missing                   |	Installed via pip install kaggle |
| Git LFS blank CSVs                |	Small files tracked by LFS    |	Updated .gitattributes           |

## Streaming Challenges & Fixes

| Issue | Cause | Resolution |
|-------|--------|------------|
| **Socket port conflict (9999)** | A previous `data_generator_stream.py` process was still running, keeping the port busy | Stopped old process using `pkill -f data_generator_stream.py` before starting a new one |
| **Empty output folders** | Spark Structured Streaming waits for the first complete window before writing data | Extended runtime (~2 min) or reduced window size (1 min window + 30 s slide) for faster feedback |
| **Checkpoint lock error** | Old checkpoint metadata from prior runs | Deleted `checkpoints/stream_kpis` before re-starting the stream |
| **Data schema mismatch** | JSON field types differed from expected schema | Enforced consistent schema with `StructType` and cast columns appropriately |
| **Slow batch writes in Codespaces** | Limited I/O performance in container | Reduced shuffle partitions and used `coalesce(1)` for smaller output files |

# Key Findings

## Operational Trends
* Summer months → highest delays
* Major hubs → greater congestion
* Departure delays → propagate to arrivals

## Predictive Modeling
* Logistic Regression and Random Forest outperform others
* Achieved AUC = 0.93 and R² = 0.78

## Impact
* Airlines can use model insights to anticipate high-delay routes or schedules
* Seasonal and route-based planning can reduce delays

## Next Steps toward ML Integration

We now plan to connect the **streaming pipeline** with our **predictive modeling** stage to enable real-time delay forecasting.

### 🔗 Integration Roadmap
1. **Feature Engineering from Stream:**  
   Aggregate rolling delay statistics (e.g., average departure/arrival delay per airline, route, or 5-minute window) from the live stream.  
   These aggregated metrics will serve as dynamic input features for the trained ML model.

2. **Model Deployment in Spark Structured Streaming:**  
   Load the saved classification/regression models from `outputs/models` using Spark MLlib’s `PipelineModel.load()`.  
   Apply the model to each micro-batch to generate *real-time predictions* of delay likelihood or duration.

3. **Online Inference Outputs:**  
   Write prediction results to a new sink (`outputs/stream_predictions/`), and optionally visualize them in the console or dashboard.

4. **Monitoring & Evaluation:**  
   Track prediction accuracy and drift over time to retrain models when significant performance degradation is detected.

5. **Dashboard Integration (optional):**  
   Build a lightweight dashboard (Streamlit or Plotly Dash) to display current average delays, predicted risks, and top delayed routes in near real-time.

# Future Enhancements
* Apply Spark Streaming for real-time monitoring
* Build interactive dashboard (Streamlit/Plotly)

Extend ML with Gradient Boosted Trees or XGBoost

🏁 Conclusion

This project demonstrates end-to-end Big Data Processing, Visualization, and Predictive Modeling using Apache Spark.
It successfully transforms massive flight datasets into actionable insights and delay predictions.
