import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 3]  # BMI
y = diabetes.target

# Split the data
X_train = X[:-20]
X_test = X[-20:]
y_train = y[:-20]
y_test = y[-20:]

# Train linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# Compute distances
distances = np.abs(y_test - y_pred)

# Get indices of top 5 largest distances
top_idx = np.argsort(distances)[-5:]

# Plot regression line
plt.figure(figsize=(8,6))
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')

# Scatter all points
plt.scatter(X_test, y_test, color='black', s=50, label='Data Points')

# Highlight and annotate top distance points
for i in top_idx:
    # Vertical line
    plt.plot([X_test[i], X_test[i]], [y_test[i], y_pred[i]], color='red', linestyle='dashed', alpha=0.7)
    # Highlight point
    plt.scatter(X_test[i], y_test[i], color='red', s=60)
    # Annotate distance with small offset
    plt.text(X_test[i]+0.01, (y_test[i]+y_pred[i])/2, f"{distances[i]:.1f}", color='red', fontsize=9)

plt.title('Linear Regression: Disease Progression vs BMI (Top Distances Highlighted)')
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.legend()
plt.show()

# Metrics
print(f"Coefficients: {regr.coef_}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")

