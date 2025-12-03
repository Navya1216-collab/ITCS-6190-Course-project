# **ITCS-6190: Flight Delay & Cancellation Analysis (2019–2023)**

### **Group 5 — Final Project Report**

| Name                   | Student ID |
| ---------------------- | ---------- |
| Navya Reddy Thadisana  | 801425759  |
| Poojitha Jayareddygari | 801426875  |
| Sahit Ceeka            | 801424751  |
| Sai Kiran Jagini       | 801484665  |
| Jeevith Doddalingegowda Rama          | 801455831  |

---

# **Project Overview**

This project implements a **complete big-data pipeline using Apache Spark**, covering:

**Data Ingestion → Cleaning → Extended EDA → SQL Analysis → Predictive ML Modeling → Structured Streaming → Real-Time Predictions → Visualization**

We processed **3+ million U.S. domestic flight records (2019–2023)** to discover:

* Which airlines/routes have the highest delays
* What features influence delays
* Whether delays can be predicted
* How to simulate streaming flight data
* How ML models perform in a real-time environment

---

# **Dataset Information**

**Source:** Kaggle – Domestic US Flight Delay & Cancellation Data (2019–2023)
**Size:** ~3 million rows (>2 GB)
**Period:** Jan 2019 – Dec 2023

### Key Columns Used

| Column               | Description      |
| -------------------- | ---------------- |
| FL_DATE              | Flight date      |
| OP_UNIQUE_CARRIER    | Airline          |
| ORIGIN, DEST         | Airports         |
| DEP_DELAY, ARR_DELAY | Delay minutes    |
| CANCELLED, DIVERTED  | Event flags      |
| DISTANCE             | Distance (Miles) |

---

# **Project Structure**

```
ITCS-6190-Course-Project/
├── data/raw/                        # Raw Kaggle CSV
├── outputs/
│   ├── curated/                     # Clean Parquet
│   ├── tables/                      # EDA tables (CSV)
│   ├── plots/                       # Visualizations (PNG)
│   ├── models/                      # Saved models + metrics
│   ├── stream_out/                  # Streaming demo outputs
├── src/
│   ├── 01_ingest_eda.py
│   ├── 02_extended_eda.py
│   ├── 02_sql_analysis.py
│   ├── 03_predictive_model.py
│   ├── 04a_make_stream_batches.py
│   ├── 04_stream_predict.py
│   ├── viz_model.py
│   ├── viz_stream.py
│   └── viz_stream_live.py
├── run.sh                           # Full pipeline (one command)
└── run_stream_predict.sh            # Streaming + live prediction runner
```

---

# **Environment Setup**

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

OR automatic installation via:

```bash
chmod +x run.sh 
./run.sh
```

### Configure `.env` (Optional)

```
RAW_DATA_GLOB=./data/raw/*.csv
CURATED_DIR=./outputs/curated
MODELS_DIR=./outputs/models
```

---

# **ONE-COMMAND FULL PIPELINE**

The entire project (Ingestion → EDA → SQL → ML Model → Streaming-ready batches) runs via:

```bash
chmod +x run.sh 
./run.sh
```

---

#  **Step 1 — Data Ingestion & Cleaning (01_ingest_eda.py)**

### Actions Performed

* Load multi-GB raw CSV using Spark
* Normalize column names
* Type casting & missing value handling
* Add derived features (year, month, day_of_week)
* Save optimized Parquet

### Output Files

Located in → `outputs/curated/`

### Sample Table Output (`arr_delay_summary.csv`)

```
month,avg_arr_delay,total_flights
1,2.19,260785
6,10.06,254998
7,9.49,278911
12,6.67,209504
```

---

# **Step 2 — Extended EDA (02_extended_eda.py)**

Generates **tables + visualizations**.

---

## **EDA Tables (Actual CSV Outputs)**

(All stored in `outputs/tables/`)

### **Top Delay Airlines (`by_airline.csv`)**

```
airline,avg_arr_delay,num_flights
Allegiant Air,13.28,50179
JetBlue Airways,12.28,109447
Frontier Airlines,11.10,62711
```

### **Monthly Delay Patterns (`by_month.csv`)**

```
month,avg_arr_delay,num_flights
1,2.19,260785
6,10.06,254998
7,9.49,278911
```

### **Route Delay Summary (`by_route_sql.csv`)**

*(Generated using Spark SQL)*

```
origin,dest,num_flights,on_time_rate
DFW,LAX,13904,0.612
DEN,PHX,12011,0.654
ORD,LGA,8789,0.589
```

---

## **EDA Plots (Actual PNG Outputs)**

(All stored in `outputs/plots/`)

* `arr_delay_histogram.png`
* `avg_delay_airline.png`
* `monthly_avg_delay.png`
* `weekday_delay.png`
* `top_routes.png`
* `cancellations_pie.png`
* `dep_vs_arr_scatter.png`
* `distance_vs_delay.png`
* `origin_delay.png`
* `dest_delay.png`

Example Insight (from actual plots):
**Departure Delay correlates extremely strongly with Arrival Delay**
(ρ ≈ 0.95)

---

# **Step 3 — SQL Analysis (02_sql_analysis.py)**

Spark SQL was used to extract insights.

### ✈ Airport On-Time Performance (`airport_on_time_sql.csv`)

```
airport,num_flights,on_time_rate
ATL,128209,0.892
DFW,110982,0.866
DEN,102614,0.842
```

---

# **Step 4 — Predictive Modeling (03_predictive_model.py)**

Models used:

### Classification

* Logistic Regression
* Random Forest Classifier

### Regression

* Linear Regression
* Random Forest Regressor

---

## **Actual Model Metrics (`metrics.txt`)**

```
TASK=classify, ALGO=lr, THRESH=15.0, ROC-AUC=0.9334
TASK=classify, ALGO=rf, THRESH=15.0, ROC-AUC=0.9108
TASK=regress, ALGO=rfreg, RMSE=22.5989, R2=0.8139
```

---

## **Confusion Matrix (`confusion_matrix.csv`)**

```
y,yhat,count
0,0,780073
0,1,12576
1,0,42413
1,1,112058
```

Interpretation:

* True Negative: 780,073
* True Positive: 112,058

Accuracy = 89%

---

## **Feature Importances (`feature_importances_rf.csv`)**

```
dep_delay,0.9109
year,0.0047
distance,0.0015
month,0.0007
airline_idx,0.00055
origin_idx,0.00050
day_of_week_idx,0.00038
dest_idx,0.00028
```

**Conclusion:**
`dep_delay` dominates → If a flight departs late, it *will* arrive late.

---

## **Sample Predictions (`predictions_sample.csv`)**

```
delayed,prediction,probability
0,0,"[0.88,0.12]"
0,0,"[0.98,0.01]"
1,0,"[0.97,0.02]"
1,1,"[0.12,0.88]"
```

---

# **Step 5 — Streaming Simulation (04a_make_stream_batches.py)**

This script converts curated data into **micro-batches** like a live airline feed.

**Outputs stored in:**
`data/stream/batch_0001.csv`, `batch_0002.csv`, …

---

# **Step 6 — Structured Streaming + Live Predictions (04_stream_predict.py)**

### What happens in this stage:

* Spark watches `data/stream/`
* Each new batch triggers:

  * ML model loading
  * Feature engineering
  * Real-time predictions
  * Saving results to Parquet

---

## Real Streaming Output (Actual Parquet → CSV screenshot)

Example batch result:

```
airline,avg_arr_delay
JetBlue Airways,31.4
Frontier Airlines,30.9
Spirit Airlines,25.4
Republic Airways,5.0
```

Stored in:

```
outputs/stream_out/predictions/batch=0/part-00000.snappy.parquet
```

---

## Live Visualization (viz_stream_live.py)

Continuously updates a scatter/line graph of predictions over time.

Example usage:

```bash
python src/viz_stream_live.py
```

---

# **Final Integrated Insights**

### **Operational Findings**

* Summer months = highest delays
* Low-cost carriers → more delays
* DFW, ORD, ATL have busiest & most delayed routes

### **ML Findings**

* Logistic Regression AUC = **0.93+**
* Random Forest AUC = **0.91+**
* *Departure delay is the strongest predictor*

### **Streaming Findings**

* Successfully simulated real-time airline feed
* Predictions are generated instantly per micro-batch
* Visual dashboard displays continuously updated results

---

# **How to Reproduce Everything**

### Run entire pipeline

```bash
chmod +x run.sh 
./run.sh
```

### Start live streaming predictions

```bash
chmod +x run_stream_predict.sh
./run_stream_predict.sh
```

### View live plot dashboard

```bash
python src/viz_stream_live.py
```

---
# Final Deliverables

- 📄 Final Report: `Documentation/Cloud Course Project Dcoumentation.pdf`
- 📊 Final Slides: `Documentation/Group 5 - Flight Delay & Cancellation Analysis.pdf`
- 🎥 Technical Video (~20 min): `Documentation/Cloud Course Project Recording.mov`

# Limitations & Future Work

- Our models focus on a subset of years/routes and may not generalize to all airports or future disruptions.
- Weather and ATC constraints are not modeled explicitly; they are only indirectly captured via delays.

**Future Work**
- Incorporate richer weather and congestion features.
- Explore gradient-boosted trees / XGBoost and hyperparameter tuning at scale.
- Deploy the streaming prediction job to a managed Spark cluster (e.g., EMR / Dataproc) instead of local mode.

# **Conclusion**

This project demonstrates a complete **end-to-end big-data system**:

* Large-scale distributed ingestion
* Optimized data engineering
* Deep EDA + SQL analytics
* Machine learning modeling
* Real-time stream processing
* Predictive analytics in streaming context

It showcases the real capabilities of **Apache Spark**, both batch & streaming, combined with **MLlib** for prediction.
