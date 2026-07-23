"""
PySpark ETL & feature engineering for the Chewy churn project.
Run:  python spark_etl.py  (expects ../data/chewy_customers.csv)
Outputs: ../data/chewy_features.parquet
"""
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder
         .appName("ChewyChurnETL")
         .config("spark.driver.memory", "4g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ---- Load -----------------------------------------------------------------
sdf = spark.read.csv("../data/chewy_customers.csv", header=True, inferSchema=True)
print(f"Loaded {sdf.count():,} rows")

# ---- Clean: duplicates + median imputation --------------------------------
before = sdf.count()
sdf = sdf.dropDuplicates(["customer_id"])
print(f"Removed {before - sdf.count():,} duplicate rows")

for c in ["email_open_rate", "avg_delivery_days", "app_sessions_per_month"]:
    median = sdf.approxQuantile(c, [0.5], 0.001)[0]
    sdf = sdf.fillna({c: median})
    print(f"Imputed nulls in {c} with median {median:.3f}")

# ---- Feature engineering (Spark SQL expressions) --------------------------
sdf = (sdf
    .withColumn("spend_per_pet",    F.round(F.col("total_spend_12m") / F.col("num_pets"), 2))
    .withColumn("order_gap_ratio",  F.round(F.col("days_since_last_order") /
                                            (90.0 / F.greatest(F.col("orders_per_quarter"), F.lit(0.3))), 3))
    .withColumn("engagement_score", F.round(0.6 * F.col("email_open_rate") +
                                            0.4 * F.col("app_sessions_per_month") / 60.0, 4))
    .withColumn("service_friction", F.col("unresolved_tickets") + F.col("refunds_12m") +
                                    (F.col("avg_delivery_days") > 4).cast("int"))
)

# ---- Distributed EDA ------------------------------------------------------
print("\nChurn rate by Autoship plan:")
(sdf.groupBy("plan")
    .agg(F.count("*").alias("customers"), F.round(F.avg("churned") * 100, 1).alias("churn_%"))
    .orderBy("churn_%").show())

# ---- Save -----------------------------------------------------------------
sdf.write.mode("overwrite").parquet("../data/chewy_features.parquet")
print("Saved ../data/chewy_features.parquet")
spark.stop()
