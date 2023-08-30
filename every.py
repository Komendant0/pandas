import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate random data
np.random.seed(42)
num_samples = 1000
features = np.random.rand(num_samples, 5)
coefficients = np.array([3.2, -1.5, 2.8, 0.7, -0.9])
noise = np.random.normal(0, 0.2, num_samples)
target = np.dot(features, coefficients) + noise

# Create a DataFrame
data = pd.DataFrame(data=np.column_stack((features, target)),
                    columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'target'])

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(train_data.drop('target', axis=1), train_data['target'])

# Make predictions
test_predictions = model.predict(test_data.drop('target', axis=1))

# Calculate Mean Squared Error
mse = mean_squared_error(test_data['target'], test_predictions)
print(f"Mean Squared Error: {mse}")

# Save the model
import joblib
joblib.dump(model, 'linear_regression_model.pkl')

# Visualize data and predictions (requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(test_data['target'], test_predictions, alpha=0.5)
plt.plot([min(test_data['target']), max(test_data['target'])], [min(test_data['target']), max(test_data['target'])], linestyle='--', color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.show()
