import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data[:, [2,8]]  # BMI and Age
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# Compute distances
distances = np.abs(y_test - y_pred)

# Indices of top 5 largest distances
top_idx = np.argsort(distances)[-5:]
max_idx = top_idx[-1]  # maximum distance

# Print performance
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))

# 3D plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Regression plane
BMI_grid, Age_grid = np.meshgrid(
    np.linspace(X[:,0].min(), X[:,0].max(), 10),
    np.linspace(X[:,1].min(), X[:,1].max(), 10)
)
plane = regr.coef_[0] * BMI_grid + regr.coef_[1] * Age_grid + regr.intercept_
ax.plot_surface(BMI_grid, Age_grid, plane, color='red', alpha=0.5)

# Scatter all points (blue)
ax.scatter(X_test[:,0], X_test[:,1], y_test, color='blue', s=50, label='Data Points')

# Highlight top distances
for i in top_idx:
    line_color = 'red' if i == max_idx else 'green'
    point_color = 'red' if i == max_idx else 'green'
    
    # Vertical dashed line to plane
    ax.plot([X_test[i,0], X_test[i,0]],
            [X_test[i,1], X_test[i,1]],
            [y_test[i], y_pred[i]], color=line_color, linestyle='dashed', alpha=0.7)
    
    # Highlight point
    ax.scatter(X_test[i,0], X_test[i,1], y_test[i], color=point_color, s=60)
    
    # Annotate distance with small offset
    mid_z = (y_test[i] + y_pred[i])/2
    ax.text(X_test[i,0]+0.01, X_test[i,1]+0.01, mid_z, f"{distances[i]:.1f}", color='black', fontsize=9)

ax.set_xlabel('BMI')
ax.set_ylabel('Age')
ax.set_zlabel('Disease Progression')
ax.set_title('Top Distances to Regression Plane (Max in Red)')
plt.legend()
plt.show()
