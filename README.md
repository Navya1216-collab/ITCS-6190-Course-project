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

# Future Enhancements
* Apply Spark Streaming for real-time monitoring
* Build interactive dashboard (Streamlit/Plotly)

Extend ML with Gradient Boosted Trees or XGBoost

🏁 Conclusion

This project demonstrates end-to-end Big Data Processing, Visualization, and Predictive Modeling using Apache Spark.
It successfully transforms massive flight datasets into actionable insights and delay predictions.
