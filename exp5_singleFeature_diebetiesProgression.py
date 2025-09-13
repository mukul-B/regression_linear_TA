import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load diabetes dataset
diabetes = datasets.load_diabetes()

def get_feature_desc(feature_name):
    feature_descriptions = {
        "age": "Age (years)",
        "sex": "Sex",
        "bmi": "Body Mass Index",
        "bp": "Average Blood Pressure",
        "s1": "Total Serum Cholesterol (TC)",
        "s2": "Low-Density Lipoproteins (LDL)",
        "s3": "High-Density Lipoproteins (HDL)",
        "s4": "TCH (Total Cholesterol / HDL)",
        "s5": "Log of Serum Triglycerides (LTG)",
        "s6": "Blood Sugar Level (GLU)"
    }
    
    return feature_descriptions.get(feature_name, feature_name.upper())

def plot_linear_regression(feature_name, top_n=5):
    """
    Plots linear regression for a chosen feature of the diabetes dataset.
    
    Parameters:
    - feature_name: str, name of the feature (e.g., 'bmi', 'age', 'bp')
    - top_n: int, number of points with largest distances to highlight
    """
    # Find feature index
    if feature_name not in diabetes.feature_names:
        raise ValueError(f"Feature '{feature_name}' not found in dataset.")
    idx = diabetes.feature_names.index(feature_name)
    
    # Prepare data
    # X = diabetes.data[:, np.newaxis, idx]
    X = diabetes.data[:, [idx]]
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
    
    # Fit linear regression
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    
    # Compute distances
    distances = np.abs(y_test - y_pred)
    
    # Get indices of top N largest distances
    top_idx = np.argsort(distances)[-top_n:]
    
    # Print metrics
    print(f"Feature: {feature_name.upper()}")
    print(f"Coefficient: {regr.coef_[0]:.3f}")
    print(f"Intercept: {regr.intercept_:.3f}")
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
    MSE_error = round(mean_squared_error(y_test, y_pred),2)
    # Plot regression line
    plt.figure(figsize=(8,6))
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
    plt.scatter(X_test, y_test, color='black', s=50, label='Data Points')
    
    # Highlight and annotate top distances
    for i in top_idx:
        plt.plot([X_test[i], X_test[i]], [y_test[i], y_pred[i]], color='red', linestyle='dashed', alpha=0.7)
        plt.scatter(X_test[i], y_test[i], color='red', s=60)
        plt.text(X_test[i]+0.01, (y_test[i]+y_pred[i])/2, f"{distances[i]:.1f}", color='red', fontsize=9)
    
    plt.title(f'Linear Regression: Disease Progression vs {feature_name.upper()} (Total error: {MSE_error})')
    plt.xlabel(get_feature_desc(feature_name))
    plt.ylabel('Disease Progression')
    plt.legend()
    plt.savefig(f'single_feature_{idx}.png')
    # plt.show()
    
    return MSE_error
    
    
    
# Example usage
if __name__ == '__main__':
    
    variables = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
    mse_list = [plot_linear_regression(vr, top_n=5) for vr in variables]
    
    plt.figure(figsize=(10,6))
    plt.bar(variables, mse_list, color='skyblue')
    plt.title("MSE of Linear Regression for Each Feature")
    plt.xlabel("Feature")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=45)
    plt.grid(True, linewidth=0.3)
    
    save_path = "singleFeaturePerformance.png"
    plt.savefig(save_path)
    # plt.show()
    print(save_path)
    # plot_linear_regression('age', top_n=5)
