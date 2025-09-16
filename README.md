# ITCS-6190-Course-project

GROUP 5:Details
Cloud Computing course project:
Navya Reddy Thadisana-801425759
Poojitha Jayareddygari 801426875
SAHIT CEEKA 801424751
Sai Kiran Jagini 801484665
Jeevith Gowda

Dataset link:
https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023

Flight Delay and Cancellation Analysis (2019–2023)
Project Overview
This project analyzes the U.S. Flight Delay and Cancellation Dataset (2019–2023) from Kaggle. The dataset contains millions of flight records across multiple U.S. airlines, including details about departure/arrival delays, cancellations, and diversions.
Our goal is to use Apache Spark to explore, analyze, and model flight delays and cancellations, identifying key factors that contribute to these events and building predictive insights.

Dataset Information
Source: Kaggle – Flight Delay and Cancellation Dataset 2019–2023
Coverage: 2019 – 2023 (5 years)
Size: Millions of rows, several GB in raw CSV format

Key Features:
FL_DATE – Flight date
OP_UNIQUE_CARRIER – Airline carrier code
ORIGIN, DEST – Origin and destination airports
DEP_DELAY, ARR_DELAY – Departure and arrival delays (in minutes)
CANCELLED, DIVERTED – Indicators for cancellations and diversions
Additional operational details (scheduled times, elapsed times, tail number, etc.)

Objectives
Data Ingestion & Cleaning
Load multi-GB dataset into Spark DataFrames.
Handle missing values, inconsistent formats, and optimize storage with Parquet.
Exploratory Data Analysis (EDA)
Distribution of delays by airline, airport, route.
Average delays by month, season, and day of the week.
Trends in cancellations and diversions.
Predictive Modeling
Build models to predict flight delay or cancellation likelihood.
Explore external weather data for correlation.
Streaming Simulation (Stretch Goal)
Use Spark Structured Streaming to simulate real-time monitoring of flight delays.
                                                                                                                                                                                                                                              Tools & Technologies
Apache Spark (PySpark, Spark SQL, MLlib, Structured Streaming)
Python (Pandas, Matplotlib/Seaborn for EDA)
Parquet (optimized storage format)
Kaggle API (for dataset access)

Project Progress
Dataset identified and downloaded.
Next step: Set up Spark environment and ingest dataset.
Plan: Perform initial exploration (schema, null values, basic distributions).

References
Dataset: Kaggle – Flight Delay and Cancellation Dataset 2019–2023
