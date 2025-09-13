import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Simulate data
np.random.seed(42)
n_samples = 50

oxygen = np.random.uniform(85, 100, n_samples)       # Feature 1
heart_rate = np.random.uniform(60, 180, n_samples)   # Feature 2

# VO2 max depends on both features + some noise
vo2_max = 0.5 * oxygen - 0.2 * heart_rate + 50 + np.random.normal(0, 2, n_samples)

# Combine features into X matrix
X = np.column_stack((oxygen, heart_rate))
y = vo2_max

model = LinearRegression()
model.fit(X, y)

# Get coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict VO2 max
y_pred = model.predict(X)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(oxygen, heart_rate, vo2_max, color='blue', label='Data')

# Create meshgrid for regression plane
O2_grid, HR_grid = np.meshgrid(
    np.linspace(min(oxygen), max(oxygen), 10),
    np.linspace(min(heart_rate), max(heart_rate), 10)
)
VO2_plane = (model.coef_[0] * O2_grid + 
             model.coef_[1] * HR_grid + 
             model.intercept_)

# Plot regression plane
ax.plot_surface(O2_grid, HR_grid, VO2_plane, color='red', alpha=0.5)

ax.set_xlabel('Blood Oxygen (%)')
ax.set_ylabel('Heart Rate (bpm)')
ax.set_zlabel('VO2 Max (ml/kg/min)')
ax.set_title('Linear Regression: VO2 Max vs Oxygen & Heart Rate')
plt.show()
