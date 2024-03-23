from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize a Spark session
spark = SparkSession.builder.appName("DistributedML").getOrCreate()

# Load and preprocess data
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Assemble feature vectors (if needed)
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")

# Initialize classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

# Set up the pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Train the model
model = pipeline.fit(data)

# Make predictions
predictions = model.transform(data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Stop the Spark session
spark.stop()
