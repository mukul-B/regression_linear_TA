import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Load diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data[:, [2, 8]]  # BMI and Age
y = diabetes.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Kernel Ridge Regression with RBF kernel
model = KernelRidge(kernel="rbf", alpha=1.0, gamma=10)  # adjust gamma for smoothness
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Create grid for surface plot
bmi_range = np.linspace(X[:,0].min(), X[:,0].max(), 50)
age_range = np.linspace(X[:,1].min(), X[:,1].max(), 50)
BMI_grid, Age_grid = np.meshgrid(bmi_range, age_range)
X_grid = np.c_[BMI_grid.ravel(), Age_grid.ravel()]
y_grid_pred = model.predict(X_grid).reshape(BMI_grid.shape)

# Plot 3D surface
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

# Scatter actual data
ax.scatter(X_test[:,0], X_test[:,1], y_test, color="black", label="Data Points")

# Regression surface
ax.plot_surface(BMI_grid, Age_grid, y_grid_pred, color="red", alpha=0.5)

ax.set_xlabel("BMI")
ax.set_ylabel("Age")
ax.set_zlabel("Disease Progression")
ax.set_title("Kernel Ridge Regression (RBF) with 2D Features")
plt.show()
