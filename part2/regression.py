#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    subplot,
    title,
    xlabel,
    ylabel,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
import torch
from dtuimldmtools import train_neural_net

from scipy.stats import ttest_rel


# In[18]:


# Function for preprocessing
def preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Check for missing values and print counts
    print(f"Missing values in 'Age': {data['Age'].isna().sum()}")
    print(f"Missing values in 'Pclass': {data['Pclass'].isna().sum()}")
    print(f"Missing values in 'Sex': {data['Sex'].isna().sum()}")
    print(f"Missing values in 'Embarked': {data['Embarked'].isna().sum()}")
    print(f"Missing values in 'Parch': {data['Parch'].isna().sum()}")
    print(f"Missing values in 'SibSp': {data['SibSp'].isna().sum()}")
    print(f"Missing values in 'Fare': {data['Fare'].isna().sum()}")
    
    # Impute missing values
    age_imputer = SimpleImputer(strategy="median")
    data["Age"] = age_imputer.fit_transform(data[["Age"]])
    
    embarked_imputer = SimpleImputer(strategy="most_frequent")
    data["Embarked"] = embarked_imputer.fit_transform(data[["Embarked"]]).ravel()

    data['Fare'] = np.log1p(data['Fare'])

    # Drop irrelevant columns
    data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
    
    # Encode categorical variables (keep all classes)
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
    data = pd.get_dummies(data, columns=["Pclass"], drop_first=False)

    # Split features and target
    X = data.drop(columns=["Fare"])
    y = data["Fare"]

    # Store feature names before scaling
    feature_names = list(data.drop(columns=["Fare"]).columns)
    print("Feature Names:", feature_names)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


# In[19]:


# Preprocess the data
X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocess_data("titanic.csv")

# Define attribute names with intercept term
attributeNames = ["Intercept"] + feature_names
print("Attribute Names:", attributeNames)


# In[20]:


# Regression task setup
def regression_task(X, y):
    # Number of features
    M = X.shape[1]  # Number of features (already scaled and ready)
    print(f"Number of features: {M}")

    # Set up cross-validation
    K = 10  # Number of folds
    CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

    # Lambda range for regularization
    lambdas = np.power(10.0, range(-2, 6))

    print("Regression task setup complete. Ready for cross-validation.")
    return M, CV, lambdas

# Set up regression task
try:
    M, CV, lambdas = regression_task(X_train_scaled, y_train)
except Exception as e:
    print(f"An error occurred: {e}")
    raise


# In[21]:


# Assume `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`, and `feature_names` come from preprocess_data
X = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])  # Add a bias (intercept) term
y = y_train.values if isinstance(y_train, pd.Series) else y_train  # Ensure y is a NumPy array

# Define the number of features
M = X.shape[1]  # Number of features (including the intercept)
print(f"Number of features: {M}")

# Define the number of folds for cross-validation
K = 10  # Number of cross-validation folds
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=42)

# Define range of lambda values (regularization parameters)
lambdas = np.power(10.0, range(-2, 6))  # Lambda values from 10^-2 to 10^5

# Initialize arrays to store training and test errors for unregularized regression
Error_train = np.empty((K, 1))  # Training errors for unregularized regression
Error_test = np.empty((K, 1))   # Test errors for unregularized regression

# Initialize arrays to store training and test errors for regularized regression
Error_train_rlr = np.empty((K, 1))  # Training errors for regularized regression
Error_test_rlr = np.empty((K, 1))   # Test errors for regularized regression

# Initialize array to store weights for regularized regression
w_rlr = np.empty((M, K))  # Shape: (number of features, number of folds)

# Initialize array to store weights for unregularized regression
w_noreg = np.empty((M, K))  # Shape: (number of features, number of folds)

# Store errors across lambda values for plotting
train_err_vs_lambda = []  # Training errors for each lambda
test_err_vs_lambda = []   # Test errors for each lambda

print(f"Initialized arrays for cross-validation with {K} folds.")


# In[22]:


def rlr_validate(X, y, lambdas, cvf=10):
    CV = model_selection.KFold(cvf, shuffle=True, random_state=42)
    M = X.shape[1]  # Number of features (including intercept)
    
    # Initialize storage
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    
    f = 0  # Fold counter
    y = y.squeeze()  # Ensure y is 1D if provided as 2D
    
    # Cross-validation loop
    for train_index, test_index in CV.split(X, y):
        # Split data into training and test folds
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        # Precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Regularization loop
        for l, lambda_val in enumerate(lambdas):
            # Regularization term, excluding bias term
            lambdaI = lambda_val * np.eye(M)
            lambdaI[0, 0] = 0  # Exclude bias regularization
            
            # Compute ridge regression weights
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            
            # Calculate train/test errors
            train_error[f, l] = np.mean(np.square(y_train - X_train @ w[:, f, l]))
            test_error[f, l] = np.mean(np.square(y_test - X_test @ w[:, f, l]))
        
        f += 1
    
    # Calculate mean errors and optimal lambda
    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.mean(w, axis=1)
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda


# In[23]:


k = 0  # Cross-validation fold counter

# Perform outer cross-validation
for train_index, test_index in CV.split(X, y):
    # Extract training and test sets for the current fold
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    internal_cross_validation = 10  # Number of inner folds for lambda selection

    # Inner cross-validation to find the optimal lambda
    (
        opt_val_err,         # Minimum validation error across all tested lambdas
        opt_lambda,          # Lambda value with the minimum validation error
        mean_w_vs_lambda,    # Mean weight values across lambdas
        train_err_vs_lambda, # Training error for each lambda
        test_err_vs_lambda,  # Validation error for each lambda
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Debug to identify attributes with unusually high coefficients
    avg_coefs = np.mean(np.abs(mean_w_vs_lambda), axis=1)  # Calculate mean coefficient magnitude across lambdas
    print(f"Cross-validation fold {k+1}:")
    for i, attribute_name in enumerate(attributeNames[1:]):  # Skip the bias term
        print(f"{attribute_name}: Average coefficient magnitude: {avg_coefs[i]:.4f}")
    print()

    # Precompute XtX and Xty for the training set
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Regularized regression: Estimate weights for optimal lambda
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Compute errors with regularization
    Error_train_rlr[k] = np.mean(np.square(y_train - X_train @ w_rlr[:, k]))
    Error_test_rlr[k] = np.mean(np.square(y_test - X_test @ w_rlr[:, k]))

    # Unregularized regression: Estimate weights without regularization
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()

    # Compute errors without regularization
    Error_train[k] = np.mean(np.square(y_train - X_train @ w_noreg[:, k]))
    Error_test[k] = np.mean(np.square(y_test - X_test @ w_noreg[:, k]))

    # Visualization for the last cross-validation fold
    if k == K - 1:
        figure(figsize=(12, 8))

        # Plot mean weight values as a function of lambda
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Exclude bias term
        xlabel("Regularization factor (lambda)")
        ylabel("Mean Coefficient Values")
        grid()
        legend(attributeNames[1:], loc="best")  # Skip "Intercept" in legend

        # Plot train and validation errors as a function of lambda
        subplot(1, 2, 2)
        title(f"Optimal lambda: {opt_lambda:.2e}")
        loglog(
            lambdas, train_err_vs_lambda.T, "b.-",
            lambdas, test_err_vs_lambda.T, "r.-"
        )
        xlabel("Regularization factor (lambda)")
        ylabel("Mean Squared Error (Cross-validation)")
        legend(["Train error", "Validation error"])
        grid()

    k += 1  # Increment fold counter


# In[24]:


# Display results for linear regression
print("Linear regression without regularization:")
print(f"- Average training error: {Error_train.mean():.4f}")
print(f"- Average test error:     {Error_test.mean():.4f}")

# Display results for regularized linear regression
print("\nRegularized linear regression:")
print(f"- Average training error: {Error_train_rlr.mean():.4f}")
print(f"- Average test error:     {Error_test_rlr.mean():.4f}")

# Display weights in the last fold
print("\nWeights in the last fold:")
print(f"{'Attribute':>20} {'Weight (w_rlr)':>20}")
print("-" * 40)
# Start from the second index to skip the bias term (Intercept)
for m in range(1, M):
    print(f"{attributeNames[m]:>20} {w_rlr[m, -1]:>20.2f}")


# In[41]:


# Parameters
outer_folds = 10
kf_outer = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

# Define hyperparameter ranges
lambdas = np.power(10.0, range(-2, 6))  # Regularization parameters for Ridge
hidden_layer_sizes = [1, 3, 5, 10]  # Number of hidden units for ANN
max_iter = 2000
n_replicates = 1

# Initialize storage for results
results_table = []

# Outer Loop: For each fold in outer cross-validation
for i, (train_index_outer, test_index_outer) in enumerate(kf_outer.split(X, y)):
    print(f"\nOuter Fold {i + 1}/{outer_folds}")

    # Split data into outer training and testing sets
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

    # ===============================
    # 1. Baseline Model
    # ===============================
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train_outer, y_train_outer)
    baseline_predictions = baseline_model.predict(X_test_outer)
    baseline_error = np.mean((y_test_outer - baseline_predictions) ** 2)

    # ===============================
    # 2. Regularized Linear Regression
    # ===============================
    best_lambda = None
    best_ridge_error = float("inf")
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)  # Inner folds

    for lambda_value in lambdas:
        inner_errors = []
        for train_index_inner, val_index_inner in kf_inner.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_index_inner], X_train_outer[val_index_inner]
            y_train_inner, y_val_inner = y_train_outer[train_index_inner], y_train_outer[val_index_inner]

            # Train Ridge Regression on inner training set
            ridge_model = Ridge(alpha=lambda_value)
            ridge_model.fit(X_train_inner, y_train_inner)
            val_predictions = ridge_model.predict(X_val_inner)
            inner_error = np.mean((y_val_inner - val_predictions) ** 2)
            inner_errors.append(inner_error)

        # Calculate mean error for this lambda
        mean_inner_error = np.mean(inner_errors)
        if mean_inner_error < best_ridge_error:
            best_ridge_error = mean_inner_error
            best_lambda = lambda_value

    # Train Ridge Regression on outer training set with best lambda
    ridge_model = Ridge(alpha=best_lambda)
    ridge_model.fit(X_train_outer, y_train_outer)
    ridge_predictions = ridge_model.predict(X_test_outer)
    ridge_error = np.mean((y_test_outer - ridge_predictions) ** 2)

    # ===============================
    # 3. Artificial Neural Network
    # ===============================
    best_hidden_layer = None
    best_ann_error = float("inf")

    for hidden_layer in hidden_layer_sizes:
        inner_errors = []
        for train_index_inner, val_index_inner in kf_inner.split(X_train_outer):
            X_train_inner = torch.Tensor(X_train_outer[train_index_inner])
            y_train_inner = torch.Tensor(y_train_outer[train_index_inner]).unsqueeze(1)
            X_val_inner = torch.Tensor(X_train_outer[val_index_inner])
            y_val_inner = torch.Tensor(y_train_outer[val_index_inner]).unsqueeze(1)

            # Define ANN model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X_train_inner.shape[1], hidden_layer),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_layer, 1),
            )
            loss_fn = torch.nn.MSELoss()

            # Train ANN
            net, _, _ = train_neural_net(
                model, loss_fn, X_train_inner, y_train_inner, max_iter=max_iter
            )

            # Validate ANN
            val_predictions = net(X_val_inner)
            inner_error = torch.mean((val_predictions - y_val_inner) ** 2).item()
            inner_errors.append(inner_error)

        # Calculate mean error for this hidden layer size
        mean_inner_error = np.mean(inner_errors)
        if mean_inner_error < best_ann_error:
            best_ann_error = mean_inner_error
            best_hidden_layer = hidden_layer

    # Train ANN on outer training set with best hidden layer size
    X_train_outer_tensor = torch.Tensor(X_train_outer)
    y_train_outer_tensor = torch.Tensor(y_train_outer).unsqueeze(1)
    X_test_outer_tensor = torch.Tensor(X_test_outer)
    net, _, _ = train_neural_net(
        lambda: torch.nn.Sequential(
            torch.nn.Linear(X_train_outer_tensor.shape[1], best_hidden_layer),
            torch.nn.Tanh(),
            torch.nn.Linear(best_hidden_layer, 1),
        ),
        torch.nn.MSELoss(),
        X_train_outer_tensor,
        y_train_outer_tensor,
        max_iter=max_iter,
    )
    ann_predictions = net(X_test_outer_tensor)
    ann_error = torch.mean((ann_predictions - torch.Tensor(y_test_outer).unsqueeze(1)) ** 2).item()

    # Append results for this fold
    results_table.append({
        "Outer Fold": i + 1,
        "Linear Regression (λ*)": best_lambda,
        "Linear Regression (E_test)": ridge_error,
        "ANN (hidden_layer*)": best_hidden_layer,
        "ANN (E_test)": ann_error,
        "Baseline (E_test)": baseline_error,
    })


# In[42]:


# Convert results into a DataFrame
results_df = pd.DataFrame(results_table)

# Display the table
print("\nTwo-level Cross-Validation Table:")
print(results_df)

# Visualize Test Errors for Each Model
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        results_df["Linear Regression (E_test)"],
        results_df["ANN (E_test)"],
        results_df["Baseline (E_test)"],
    ],
    labels=["Linear Regression", "ANN", "Baseline"],
)
plt.ylabel("Test Error (MSE)")
plt.title("Model Comparison: Test Errors")
plt.grid()
plt.show()


# In[43]:


# Extract test errors from the results table
ridge_errors = results_df["Linear Regression (E_test)"]
ann_errors = results_df["ANN (E_test)"]
baseline_errors = results_df["Baseline (E_test)"]

# Pairwise t-tests
print("Paired t-tests:")
ann_vs_ridge_t, ann_vs_ridge_p = ttest_rel(ann_errors, ridge_errors)
print(f"ANN vs. Linear Regression: t={ann_vs_ridge_t:.4f}, p={ann_vs_ridge_p:.4f}")

ann_vs_baseline_t, ann_vs_baseline_p = ttest_rel(ann_errors, baseline_errors)
print(f"ANN vs. Baseline: t={ann_vs_baseline_t:.4f}, p={ann_vs_baseline_p:.4f}")

ridge_vs_baseline_t, ridge_vs_baseline_p = ttest_rel(ridge_errors, baseline_errors)
print(f"Linear Regression vs. Baseline: t={ridge_vs_baseline_t:.4f}, p={ridge_vs_baseline_p:.4f}")

# Interpret results
alpha = 0.05
print("\nInterpretation:")
print(f"ANN vs. Linear Regression: {'Significant' if ann_vs_ridge_p < alpha else 'Not significant'}")
print(f"ANN vs. Baseline: {'Significant' if ann_vs_baseline_p < alpha else 'Not significant'}")
print(f"Linear Regression vs. Baseline: {'Significant' if ridge_vs_baseline_p < alpha else 'Not significant'}")


# In[44]:


# Calculate differences between test errors
diff_ann_vs_ridge = ann_errors - ridge_errors
diff_ann_vs_baseline = ann_errors - baseline_errors
diff_ridge_vs_baseline = ridge_errors - baseline_errors

# Compute mean and standard error of differences
mean_diff_ann_ridge = np.mean(diff_ann_vs_ridge)
se_diff_ann_ridge = np.std(diff_ann_vs_ridge, ddof=1) / np.sqrt(len(diff_ann_vs_ridge))

mean_diff_ann_baseline = np.mean(diff_ann_vs_baseline)
se_diff_ann_baseline = np.std(diff_ann_vs_baseline, ddof=1) / np.sqrt(len(diff_ann_vs_baseline))

mean_diff_ridge_baseline = np.mean(diff_ridge_vs_baseline)
se_diff_ridge_baseline = np.std(diff_ridge_vs_baseline, ddof=1) / np.sqrt(len(diff_ridge_vs_baseline))

# Compute 95% confidence intervals
ci_ann_ridge = (mean_diff_ann_ridge - 1.96 * se_diff_ann_ridge, mean_diff_ann_ridge + 1.96 * se_diff_ann_ridge)
ci_ann_baseline = (mean_diff_ann_baseline - 1.96 * se_diff_ann_baseline, mean_diff_ann_baseline + 1.96 * se_diff_ann_baseline)
ci_ridge_baseline = (mean_diff_ridge_baseline - 1.96 * se_diff_ridge_baseline, mean_diff_ridge_baseline + 1.96 * se_diff_ridge_baseline)

# Print results
print("\nConfidence Intervals (Setup II):")
print(f"ANN vs. Linear Regression: mean={mean_diff_ann_ridge:.4f}, 95% CI={ci_ann_ridge}")
print(f"ANN vs. Baseline: mean={mean_diff_ann_baseline:.4f}, 95% CI={ci_ann_baseline}")
print(f"Linear Regression vs. Baseline: mean={mean_diff_ridge_baseline:.4f}, 95% CI={ci_ridge_baseline}")

