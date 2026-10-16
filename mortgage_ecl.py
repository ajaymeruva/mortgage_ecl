"""
Mortgage Default Prediction with Expected Credit Loss (ECL) Calculations

This script performs mortgage default prediction using PySpark MLlib's Logistic Regression
and calculates Expected Credit Loss (ECL) using PD (Probability of Default), 
LGD (Loss Given Default), and EAD (Exposure at Default).
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf, when, lit
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("MortgageDefaultPrediction").getOrCreate()

data = [
    (1, 750, 300000, 5000, 1, 0.03, 0.7, 0.9),  # Customer 1: Good credit, paid well, etc.
    (2, 650, 150000, 3000, 0, 0.05, 0.6, 0.8),  # Customer 2: Average credit, defaulted
    (3, 680, 250000, 4000, 1, 0.04, 0.65, 0.85), # Customer 3: Good credit, paid well, etc.
    (4, 620, 100000, 2000, 0, 0.1, 0.5, 0.7),   # Customer 4: Poor credit, defaulted
    (5, 700, 400000, 6000, 1, 0.02, 0.75, 0.95)  # Customer 5: Good credit, paid well
]

columns = ['Customer_ID', 'Credit_Score', 'Loan_Amount', 'Income', 'Default_Flag', 'Interest_Rate', 'LTV', 'Loan_Term']

df = spark.createDataFrame(data, columns)

df.show()

assembler = VectorAssembler(inputCols=['Credit_Score', 'Loan_Amount', 'Income', 'Interest_Rate', 'LTV', 'Loan_Term'],
                            outputCol='features')

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(labelCol='Default_Flag', featuresCol='features')

pipeline = Pipeline(stages=[assembler, lr])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

# Show predictions
predictions.select('Customer_ID', 'prediction', 'probability').show()

extract_pd = udf(lambda v: float(v[1]) if v is not None else None, DoubleType())
predictions = predictions.withColumn('PD', extract_pd(col('probability')))

predictions.select('Customer_ID', 'PD').show()

# Assuming a fixed LGD rate based on the credit score.
def calculate_LGD(credit_score):
    if credit_score < 650:
        return 0.9  # High loss for low credit score
    elif credit_score < 700:
        return 0.7  # Moderate loss for average credit score
    else:
        return 0.5  # Lower loss for good credit score
    
lgd_udf = udf(calculate_LGD, DoubleType())

predictions = predictions.withColumn('LGD', 
                                     when(col('Credit_Score').isNotNull(), lgd_udf(col('Credit_Score')))
                                     .otherwise(lit(0)))  # Default to 0 if Credit_Score is null

predictions = predictions.withColumn('EAD', col('Loan_Amount'))

# Step 10: Calculate ECL (Expected Credit Loss)
predictions = predictions.withColumn('ECL', col('PD') * col('LGD') * col('EAD'))

# Step 11: Show the final results with PD, LGD, EAD, and ECL
predictions.select('Customer_ID', 'PD', 'LGD', 'EAD', 'ECL').show()

# Stop the Spark session
spark.stop()