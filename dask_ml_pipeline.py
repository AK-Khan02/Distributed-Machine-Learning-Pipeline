from dask.distributed import Client
import dask.array as da
from dask_ml.model_selection import train_test_split
from dask_ml.xgboost import XGBClassifier
from dask_ml.metrics import accuracy_score

# Start a Dask client
client = Client()

# Generate or load data
X, y = da.random.random((10000, 20)), da.random.randint(0, 2, size=10000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the classifier
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc}")

# Close the Dask client
client.close()
