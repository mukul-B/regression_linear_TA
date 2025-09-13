# Use multiple features: BMI, age, and blood pressure
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()

# feature_list =  [random.randint(0, 9) for _ in range(5)]
# feature_list = random.sample(range(5), 5)

# feature_list = [7,8,2,3]
feature_list = [8]
print(feature_list)
X = diabetes.data[:, feature_list]  # BMI, age, and blood pressure
y = diabetes.target


X_train_multi, X_test_multi, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f' Training size: {len(y_train)} Test : {len(y_test)}')
# Create linear regression object
regr_multi = linear_model.LinearRegression()

# Train the model using the training sets
regr_multi.fit(X_train_multi, y_train)

# Make predictions using the testing set
y_pred_multi = regr_multi.predict(X_test_multi)

# Print coefficients
print(f"Coefficients: {regr_multi.coef_}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred_multi):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred_multi):.2f}")
