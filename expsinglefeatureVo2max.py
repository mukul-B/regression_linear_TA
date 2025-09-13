import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulate data
np.random.seed(42)
# Oxygen saturation (SpO2) in %
oxygen = np.random.uniform(85, 100, 30)  # independent variable
# VO2 max (ml/kg/min), assume linear relation + some noise
vo2_max = 0.8 * oxygen - 30 + np.random.normal(0, 2, len(oxygen))  # dependent variable

# Visualize raw data
plt.scatter(oxygen, vo2_max, color='blue')
plt.xlabel("Blood Oxygen (%)")
plt.ylabel("VO2 Max (ml/kg/min)")
plt.title("VO2 Max vs Blood Oxygen")
plt.show()


# Reshape data for sklearn
X = oxygen.reshape(-1, 1)
y = vo2_max

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Fitted line: VO2_max = {slope:.2f} * Oxygen + {intercept:.2f}")

# Make predictions
y_pred = model.predict(X)

plt.scatter(oxygen, vo2_max, color='blue', label="Data")
plt.plot(oxygen, y_pred, color='red', label="Linear Regression Fit")
plt.xlabel("Blood Oxygen (%)")
plt.ylabel("VO2 Max (ml/kg/min)")
plt.title("Linear Regression: VO2 Max vs Blood Oxygen")
plt.legend()
plt.show()
