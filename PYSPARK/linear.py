from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("IstanbulStockExchangeModel").getOrCreate()

# Load dataset
file_path = "./istanbul_stock_exchange.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
print(df.columns)

# Display schema and first few rows
df.printSchema()
df.show(5)

# Drop rows with missing values
df = df.dropna()

# Define feature columns and target column
feature_cols = ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
target_col = "ISE_TL"  # Assuming ISE is the Istanbul Stock Exchange return column

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Scale features (optional)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Select relevant columns
df = df.select(col("scaled_features").alias("features"), col(target_col).alias("label"))

# Split dataset into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train a linear regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Model summary
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Predict on test data
predictions = lr_model.transform(test_data)
predictions.select("features", "label", "prediction").show(5)

# Evaluate model performance
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Stop Spark session
spark.stop()
