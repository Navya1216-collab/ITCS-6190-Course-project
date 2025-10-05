# ITCS-6190-Course-project

**GROUP 5: Details**
Cloud Computing course project:

* Navya Reddy Thadisana – 801425759
* Poojitha Jayareddygari – 801426875
* Sahit Ceeka – 801424751
* Sai Kiran Jagini – 801484665
* Jeevith Gowda

---

## Dataset

* **Link:** [Flight Delay and Cancellation Dataset (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
* **Coverage:** 2019 – 2023 (5 years)
* **Size:** Millions of rows, several GB in raw CSV format

**Key Features:**

* `FL_DATE` – Flight date
* `OP_UNIQUE_CARRIER` – Airline carrier code
* `ORIGIN`, `DEST` – Origin and destination airports
* `DEP_DELAY`, `ARR_DELAY` – Departure and arrival delays (in minutes)
* `CANCELLED`, `DIVERTED` – Indicators for cancellations and diversions
* Additional operational details (scheduled times, elapsed times, tail number, etc.)

---

## Project Overview

This project analyzes the U.S. Flight Delay and Cancellation Dataset (2019–2023) from Kaggle.
The dataset contains millions of flight records across multiple U.S. airlines, including details about departure/arrival delays, cancellations, and diversions.

Our goal is to use **Apache Spark** to explore, analyze, and model flight delays and cancellations, identifying key factors that contribute to these events and building predictive insights.

---

## Objectives

1. **Data Ingestion & Cleaning**

   * Load multi-GB dataset into Spark DataFrames
   * Handle missing values, inconsistent formats
   * Optimize storage with Parquet

2. **Exploratory Data Analysis (EDA)**

   * Distribution of delays by airline, airport, route
   * Average delays by month, season, and day of week
   * Trends in cancellations and diversions

3. **Predictive Modeling**

   * Build models to predict delay/cancellation likelihood
   * Explore external weather data correlation

4. **Streaming Simulation (Stretch Goal)**

   * Use Spark Structured Streaming to simulate real-time monitoring of flight delays

---

## Tools & Technologies

* Apache Spark (PySpark, Spark SQL, MLlib, Structured Streaming)
* Python (Pandas, Matplotlib for EDA)
* Parquet (optimized storage format)
* Kaggle API (for dataset access)

---

## Project Progress

* Dataset identified and downloaded
* Spark environment set up in Codespaces
* Data Ingestion + Initial EDA completed ✅

---

## Data Ingestion & EDA

### Steps Performed

**1. Environment Setup**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyspark pandas matplotlib python-dotenv
```

**2. Environment Variables (`.env`)**

```bash
RAW_DATA_GLOB=/workspaces/data/flights_sample_3m.csv
CURATED_DIR=/workspaces/data/curated
SAMPLE_DATA_GLOB=./samples/*.csv
```

**3. Run Script with `run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs) || true
fi

python src/01_ingest_eda.py
```

Run with:

```bash
./run.sh
```

**4. Ingestion Script (`src/01_ingest_eda.py`)**

* Reads raw CSV(s) into Spark DataFrames
* Renames important columns (`FL_DATE → flight_date`, `AIRLINE → airline`, etc.)
* Casts numeric fields (`arr_delay`, `dep_delay`, `distance`)
* Drops nulls and duplicates
* Writes **partitioned Parquet files** to `$CURATED_DIR`

```python
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(RAW_DATA_GLOB))

# Rename + cast
df = df.withColumnRenamed("FL_DATE", "flight_date") \
       .withColumnRenamed("AIRLINE", "airline") \
       .withColumnRenamed("ORIGIN", "origin") \
       .withColumnRenamed("DEST", "dest") \
       .withColumnRenamed("ARR_DELAY", "arr_delay") \
       .withColumnRenamed("DEP_DELAY", "dep_delay")

df = df.withColumn("arr_delay", F.col("arr_delay").cast("double")) \
       .withColumn("dep_delay", F.col("dep_delay").cast("double")) \
       .withColumn("distance", F.col("distance").cast("double"))
```

**5. EDA Queries**

* Null counts:

```python
nulls = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
nulls.show(truncate=False)
```
Insight: Most columns have negligible missing data, confirming strong dataset integrity.
The primary nulls appear in CANCELLATION_CODE, which is expected since non-cancelled flights naturally have NULL codes.
This validates that cancellations are correctly encoded rather than missing.

* Average arrival delay by airline:

```python
df.groupBy("airline").agg(
    F.avg("arr_delay").alias("avg_arr_delay"),
    F.count("*").alias("n")
).orderBy(F.desc("avg_arr_delay")).show(20, truncate=False)
```

Insight: Airlines such as Allegiant Air and JetBlue Airways show the highest mean delays,
suggesting that their routes or schedules face heavier congestion.
In contrast, Southwest and Delta maintain comparatively lower averages, reflecting stronger on-time reliability.

* Monthly average delays:

```python
df.withColumn("month", F.month("flight_date")) \
  .groupBy("month") \
  .agg(F.avg("arr_delay").alias("avg_arr_delay"), F.count("*").alias("n")) \
  .orderBy("month").show(12, truncate=False)
```
Insight: Delay levels rise sharply during June–August, the summer travel period.
Weather disruptions and vacation traffic likely explain this seasonal trend.
Winter months show shorter delays, consistent with reduced passenger volume.

* Most delayed routes:

```python
df.groupBy("origin","dest") \
  .agg(F.avg("arr_delay").alias("avg_arr_delay"),
       F.count("*").alias("n")) \
  .filter(F.col("n") > 500) \
  .orderBy(F.desc("avg_arr_delay")).show(20, truncate=False)
```
Insight: Routes like DEN → ASE, MCO → JFK, and DFW → HOU exhibit consistently high delay averages.
These are typically busy hub-to-hub or weather-sensitive corridors, implying that airport congestion,
rather than flight distance, is the main driver of late arrivals.

---------------------------------------------------------------------------------------------------------------------------------------------------------
**6. Visualization**

```python
plt.plot(pdf["month"], pdf["avg_arr_delay"])
plt.xlabel("Month")
plt.ylabel("Avg Arrival Delay (min)")
plt.title("Seasonal Delays (All Flights)")
plt.savefig("docs/assets/seasonal_delays.png", bbox_inches="tight")
```

---

(a) Average Delay by Airline

```python
# Average delay by airline
pdf = df.groupBy("airline").agg(F.avg("arr_delay").alias("avg_delay")).toPandas()
pdf = pdf.sort_values("avg_delay", ascending=False)

plt.figure(figsize=(10,5))
plt.bar(pdf["airline"], pdf["avg_delay"], color="skyblue")
plt.title("Average Arrival Delay by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Delay (min)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("docs/assets/avg_delay_airline.png", bbox_inches="tight")

Insight: Airlines such as Allegiant Air and JetBlue Airways experience the highest average delays, while Southwest and Delta show better on-time performance.
This indicates differences in operational efficiency and scheduling resilience among carriers.

(b) Correlation Heatmap – Delay vs Distance
# Correlation heatmap for delay metrics
pdf_corr = df.select("arr_delay", "dep_delay", "distance").toPandas().corr()

plt.figure(figsize=(5,4))
sns.heatmap(pdf_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap – Arrival vs Departure Delay")
plt.tight_layout()
plt.savefig("docs/assets/delay_corr_heatmap.png", bbox_inches="tight")

Insight: A strong positive correlation (~0.95) exists between dep_delay and arr_delay, confirming that late departures nearly always lead to late arrivals.
The weak correlation with distance suggests flight length has minimal influence on delay duration.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Outputs Generated

**Dataset Size**

* Rows: `3,000,000`
* Columns: `32`

**Average Arrival Delay by Airline (Top 5)**

| Airline           | Avg Delay (min) | Flights |
| ----------------- | --------------: | ------: |
| Allegiant Air     |           13.28 |  50,179 |
| JetBlue Airways   |           12.28 | 109,447 |
| Frontier Airlines |           11.10 |  62,711 |
| ExpressJet (aha!) |           10.03 |  17,951 |
| Spirit Air Lines  |            8.03 |  93,200 |

Interpretation: Carriers such as Allegiant Air and JetBlue experience higher delay minutes on average,
while major airlines like American and Delta maintain steadier operations.
This indicates efficiency differences that can be modeled later for prediction.

**Monthly Average Arrival Delay**

| Month | Avg Delay (min) | Flights |
| ----: | --------------: | ------: |
|     1 |            2.19 | 260,785 |
|     6 |           10.06 | 254,998 |
|     7 |            9.49 | 278,911 |
|     8 |            6.45 | 280,603 |
|    12 |            6.67 | 209,504 |

Interpretation: Delays increase during summer and holiday seasons (June–August, December),
demonstrating clear temporal patterns that can improve time-aware models.

**Most Delayed Routes (n > 500)**

| Origin | Dest | Avg Delay (min) | Flights |
| :----: | :--: | --------------: | ------: |
|   DEN  |  ASE |           21.25 |     910 |
|   PNS  |  DFW |           19.36 |     721 |
|   MCO  |  JFK |           18.57 |   1,910 |
|   FLL  |  JFK |           18.30 |   1,710 |
|   DFW  |  HOU |           18.28 |     947 |

Interpretation: High-delay routes cluster around major connecting airports.
This supports further feature engineering based on route congestion and hub classification.

**Cancellations**

* `cancellation_code=NULL` count: **2,913,802** (majority not cancelled)
Interpretation: Over 97% of flights are not cancelled (cancellation_code=NULL),
confirming class imbalance for predictive modeling and the need for resampling or weighted evaluation metrics

**Visualization**

* Seasonal delays plot generated → `docs/assets/seasonal_delays.png`

---
##  Weekly Check-in Summary

###  Progress Since Last Meeting
- Completed Spark-based ingestion pipeline (CSV → Parquet conversion).
- Cleaned, renamed, and casted columns for consistency.
- Performed detailed EDA on 3 million flight records:
  - Delay trends by airline, month, and route.
  - Correlation analysis between departure and arrival delays.
  - Generated three visualizations for seasonal, airline, and correlation patterns.
- Documented all findings with insights and interpretation in the README.

###  Current Challenges / Blockers
- Dataset size (multi-GB) causes long load times in Codespaces.
- Need to join external weather data for richer analysis.
- Some route-level granularity leads to high variance in averages.
- Storage optimization and caching improvements under review.

###  Plan for Next Week
- Integrate external **weather dataset** to explore delay correlation.
- Apply feature engineering (e.g., route congestion score, seasonal encoding).
- Begin baseline **predictive modeling** for delay likelihood using Spark MLlib.
- Add additional visuals (e.g., boxplots or trend lines) for model explainability.
- Prepare slides for the next progress presentation.

- Apply feature engineering (e.g., route congestion score, seasonal encoding).
- Begin baseline **predictive modeling** for delay likelihood using Spark MLlib.
- Add additional visuals (e.g., boxplots or trend lines) for model explainability.
- Prepare slides for the next progress presentation.

