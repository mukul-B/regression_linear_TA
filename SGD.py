from sklearn.linear_model import SGDRegressor
import numpy as np

# Fake data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# SGD Regressor (linear regression with SGD)
model = SGDRegressor(loss="squared_error", learning_rate="constant", eta0=0.01, max_iter=1000)
model.fit(X, y)

print("Weights:", model.coef_, "Intercept:", model.intercept_)