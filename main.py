
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Example dataset: Hours studied vs Exam score
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # Hours studied
y = np.array([15, 25, 35, 45, 50, 60])           # Exam score

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Best-fit line")
plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.legend()
plt.show()
