from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression

# ---------------------------------------------------
# 1. START SPARK SESSION
# ---------------------------------------------------
spark = SparkSession.builder.appName("FlightDelayPipeline").getOrCreate()
print("🚀 Spark session started successfully!")

# ---------------------------------------------------
# 2. LOAD CLEANED CSV FILES FOR ANALYSIS
# ---------------------------------------------------
print("📂 Loading cleaned EDA CSV files...")

df_airline = spark.read.csv("data/eda/by_airline_quick_csv/*.csv", header=True, inferSchema=True)
df_month   = spark.read.csv("data/eda/by_month_quick_csv/*.csv", header=True, inferSchema=True)
df_cancel  = spark.read.csv("data/eda/cancellations_quick_csv/*.csv", header=True, inferSchema=True)
df_routes  = spark.read.csv("data/eda/routes_quick_csv/*.csv", header=True, inferSchema=True)

print("✅ All datasets loaded successfully!")

# ---------------------------------------------------
# 3. MERGE DATA FOR ML MODEL (WE USE BY_MONTH DATA)
# ---------------------------------------------------
df = df_month.select(
    col("Month").alias("MONTH"),
    col("Airline").alias("AIRLINE"),
    col("Total_Flights"),
    col("Total_Delays"),
    col("Average_Arrival_Delay")
)

df = df.na.drop()
print("🧹 Cleaned dataset ready for ML modeling!")

# ---------------------------------------------------
# 4. BASIC EDA (EXAMPLE OUTPUT)
# ---------------------------------------------------
print("📊 Top 5 months with highest total delays:")
df.orderBy(desc("Total_Delays")).show(5)

# ---------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------
# Convert airline string → numeric
indexer = StringIndexer(
    inputCol="AIRLINE",
    outputCol="AIRLINE_INDEX"
)
df = indexer.fit(df).transform(df)

# Assemble ML features
assembler = VectorAssembler(
    inputCols=["MONTH", "Total_Flights", "Total_Delays", "AIRLINE_INDEX"],
    outputCol="features"
)

df_ml = df.withColumn("label", col("Average_Arrival_Delay"))
df_ml = assembler.transform(df_ml).select("features", "label")

print("🧱 Feature engineering complete!")

# ---------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ---------------------------------------------------
train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

# ---------------------------------------------------
# 7. TRAIN ML MODEL (PREDICT ARRIVAL DELAY)
# ---------------------------------------------------
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(train)

predictions = model.transform(test)

print("📈 Model Evaluation:")
print(f"RMSE = {model.summary.rootMeanSquaredError}")
print(f"R2   = {model.summary.r2}")

# ---------------------------------------------------
# 8. DONE
# ---------------------------------------------------
print("🎉 End-to-End Spark Pipeline Completed Successfully!")
spark.stop()
