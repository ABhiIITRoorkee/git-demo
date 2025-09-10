
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Dataset: House Size vs Price
X = np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000]).reshape(-1, 1)
y = np.array([15, 18, 21, 24, 27, 30, 33, 36])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model parameters
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Best-Fit Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (in lakhs)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()
