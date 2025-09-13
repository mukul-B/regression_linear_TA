import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

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

def plot_linear_regression(feature_names, top_n=5):
    """
    Plots linear regression for one or two chosen features of the diabetes dataset.
    """
    # Ensure list
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Validate
    idx = []
    for feature_name in feature_names:
        if feature_name not in diabetes.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in dataset.")
        idx.append(diabetes.feature_names.index(feature_name))
    
    # Prepare data
    X = diabetes.data[:, idx]
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
    top_idx = np.argsort(distances)[-top_n:]
    
    # Print metrics
    print(f"Features: {[f.upper() for f in feature_names]}")
    print(f"Coefficients: {regr.coef_}")
    print(f"Intercept: {regr.intercept_:.3f}")
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
    MSE_error = round(mean_squared_error(y_test, y_pred), 2)

    # Plot
    if len(feature_names) == 1:
        # --- 2D PLOT ---
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(X_test, y_test, color="blue", s=50, label="Data Points")
        ax.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
        
        # Highlight outliers
        for i in top_idx:
            ax.vlines(X_test[i], y_test[i], y_pred[i], color="red", linestyle="dashed")
            ax.scatter(X_test[i], y_test[i], color="red", s=60)
            ax.text(X_test[i], (y_test[i]+y_pred[i])/2, f"{distances[i]:.1f}", fontsize=9)

        ax.set_title(f"Linear Regression: Disease Progression vs {get_feature_desc(feature_names[0])} (MSE={MSE_error})")
        ax.set_xlabel(get_feature_desc(feature_names[0]))
        ax.set_ylabel("Disease Progression")
        ax.legend()
        plt.savefig(f'single_feature_{idx[0]}.png')
        # plt.show()

    elif len(feature_names) == 2:
        # --- 3D PLOT ---
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d")

        # Create regression plane
        x1, x2 = np.meshgrid(
            np.linspace(X[:,0].min(), X[:,0].max(), 10),
            np.linspace(X[:,1].min(), X[:,1].max(), 10)
        )
        plane = regr.coef_[0]*x1 + regr.coef_[1]*x2 + regr.intercept_
        ax.plot_surface(x1, x2, plane, color="red", alpha=0.5)

        # Scatter points
        ax.scatter(X_test[:,0], X_test[:,1], y_test, color="blue", s=50, label="Data Points")

        # Highlight top distances (draw perpendicular lines to plane)
        for i in top_idx:
            x_point = X_test[i,0]
            y_point = X_test[i,1]
            z_actual = y_test[i]
            z_pred = y_pred[i]

            ax.plot([x_point, x_point], [y_point, y_point], [z_actual, z_pred],
                    color="red", linestyle="dashed")
            ax.scatter(x_point, y_point, z_actual, color="red", s=60)
            ax.text(x_point, y_point, (z_actual+z_pred)/2, f"{distances[i]:.1f}", fontsize=9)

        ax.set_title(f"Linear Regression: {get_feature_desc(feature_names[0])} + {get_feature_desc(feature_names[1])} (MSE={MSE_error})")
        ax.set_xlabel(get_feature_desc(feature_names[0]))
        ax.set_ylabel(get_feature_desc(feature_names[1]))
        ax.set_zlabel("Disease Progression")
        plt.legend()
        plt.savefig(f'two_feature_{idx[0]}_{idx[1]}.png')
        # plt.show()

    return MSE_error

# Example usage
if __name__ == "__main__":
    variables = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
    # variables = [["s5","bmi"]]
    # variables = [["s5","bmi","s4"] ]  
    # variables = [["s5","bmi","s4","bp"] ]
    # variables = [["s5","bmi","s4","sex"] ]
    # variables = [["s5","bmi","s4","s3"] ]
    
    mse_list = [plot_linear_regression(vr, top_n=5) for vr in variables]

    plt.figure(figsize=(10,6))
    labels = [str(vr) for vr in variables]
    plt.bar(labels, mse_list, color="skyblue")
    plt.title("MSE of Linear Regression for Features")
    plt.xlabel("Feature(s)")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, linewidth=0.3)
    plt.xticks(rotation=45)
    save_path = "singleFeaturePerformance.png"
    plt.savefig(save_path)
    # plt.show()
    print(save_path)
    # plt.show()
