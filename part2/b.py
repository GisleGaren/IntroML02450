import numpy as np
import pandas as pd
import torch
from dtuimldmtools import train_neural_net
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


def main():
    # Fetch data and clean data in the same manner as in `analysis.ipynb`.
    df = pd.read_csv('dataset/train.csv')
    df_clean = df.drop(columns=[
        'PassengerId', 'Name', 'Ticket', 'Cabin','Survived', 'Fare',
    ])
    df_clean['Age'] = df_clean['Age'].fillna(df['Age'].median())
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df['Embarked'].mode()[0])
    df_clean = pd.get_dummies(df_clean, columns=['Sex', 'Embarked'], drop_first=True)
    df_clean = pd.get_dummies(df_clean, columns=['Pclass'])
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_clean)

    # Prepare data for two fold cross validation
    X = np.hstack([np.ones((df_standardized.shape[0], 1)), df_standardized])
    y = df['Fare'].values
    k1 = 10
    k2 = 10

    # Model parameters
    ann_parameters = [1, 5, 10, 15]
    reg_parameters = np.power(10.0, range(-2, 6))

    # Do two level cross validation
    errors = two_level_cv(k1, k2, X, y, ann_parameters, reg_parameters)
    # Report errors
    table = make_table(k1, *errors)
    with open('table.tex', 'w') as fp:
        fp.write(table)
    print(table)


def two_level_cv(k1, k2, X, y, ann_parameters, reg_parameters):
    # Number of features
    M = X.shape[1]
    # Error calculation
    mean_squared_error = lambda y, y_est: sum((y - y_est) ** 2) / len(y)

    # Data for two level cross validation of ANN
    ## model parameters
    hidden_units_parameters = ann_parameters
    ## Training parameters
    n_replicates = 3
    max_iter = 10_000
    loss_fn = torch.nn.MSELoss()
    ## The number of models to test
    s = len(hidden_units_parameters)
    ## The list of models to test
    ann_models = [
        lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden_units, 1),
        ) for n_hidden_units in hidden_units_parameters
    ]

    # Data for two level cross validation of regularized linear regression
    ## model parameters
    regularization_parameters = reg_parameters
    ## The number of models to test
    t = len(regularization_parameters)

    # The test errors and parameters
    ann_test_error = np.empty((k1, 2))
    reg_test_error = np.empty((k1, 2))
    baseline_test_error = np.empty(k1)

    # The outer cross fold iterator
    outer_cv = model_selection.KFold(k1, shuffle=True)
    outer_cv = enumerate(outer_cv.split(X, y))
    # The outer cross validation loop
    for outer_fold, (outer_train_index, outer_test_index) in outer_cv:
        print(f'Running outer fold n. {outer_fold}')

        # The outer train/test split
        X_outer_train = X[outer_train_index, :]
        y_outer_train = y[outer_train_index]
        X_outer_test = X[outer_test_index, :]
        y_outer_test = y[outer_test_index]

        # The validation errors per ANN model
        ann_validation_error = np.empty((s, k2))
        # The validation errors per regression model
        reg_validation_error = np.empty((t, k2))

        # The inner cross fold iterator
        inner_cv = model_selection.KFold(k2, shuffle=True)
        inner_cv = enumerate(inner_cv.split(X_outer_train, y_outer_train))
        # The inner cross validation loop
        for inner_fold, (inner_train_index, inner_test_index) in inner_cv:
            print(f'Running inner fold n. {inner_fold} '
                  f'in outer fold n. {outer_fold}')
            # The inner train/test split
            X_inner_train = X_outer_train[inner_train_index, :]
            y_inner_train = y_outer_train[inner_train_index]
            X_inner_test = X_outer_train[inner_test_index, :]
            y_inner_test = y_outer_train[inner_test_index]

            # ANN Inner Fold
            print('Training ANN\'s')
            ## Train each model and validate it on the inner train/test split
            for i, model in enumerate(ann_models):
                net, _, _ = train_neural_net(
                    model,
                    loss_fn,
                    X=torch.Tensor(X_inner_train),
                    y=torch.Tensor(y_inner_train).unsqueeze(1),
                    n_replicates=n_replicates,
                    max_iter=max_iter,
                )
                y_estimates = net(torch.Tensor(X_inner_test))
                y_estimates = y_estimates.flatten().float().data.numpy()
                mse = mean_squared_error(y_inner_test, y_estimates)
                ann_validation_error[i, inner_fold] = mse

            # Regression Inner Fold
            print('Training regression')
            ## Precompute terms involving the design matrix
            Xty = X_inner_train.T @ y_inner_train
            XtX = X_inner_train.T @ X_inner_train
            ## Train each model and validate it on the inner train/test split
            for i, regularization_parameter in enumerate(regularization_parameters):
                # Prepare matrix of regularization terms
                lambdas = regularization_parameter * np.eye(M)
                lambdas[0, 0] = 0
                # Solve for model weights
                w = np.linalg.solve(XtX + lambdas, Xty).squeeze()
                # Determine validation error
                y_estimates = X_inner_test @ w.T
                mse = mean_squared_error(y_inner_test, y_estimates)
                reg_validation_error[i, inner_fold] = mse

        # ANN Outer Fold
        print(f'Training optimal ANN for outer fold n. {outer_fold}')
        ## Extract the optimal ANN model
        optimal_index = ann_validation_error.sum(axis=1).argmin()
        optimal_n_hidden_units = hidden_units_parameters[optimal_index]
        model = ann_models[optimal_index]
        ## Train the optimal ANN model
        net, _, _ = train_neural_net(
            model,
            loss_fn,
            X=torch.Tensor(X_outer_train),
            y=torch.Tensor(y_outer_train).unsqueeze(1),
            n_replicates=n_replicates,
            max_iter=max_iter,
        )
        ## Determine test error
        y_estimates = net(torch.Tensor(X_outer_test))
        y_estimates = y_estimates.flatten().float().data.numpy()
        mse = mean_squared_error(y_outer_test, y_estimates)
        ann_test_error[outer_fold, 0] = optimal_n_hidden_units
        ann_test_error[outer_fold, 1] = mse

        # Regression Outer Fold
        print(f'Training optimal regression for outer fold n. {outer_fold}')
        ## Precompute terms involving the design matrix
        Xty = X_outer_train.T @ y_outer_train
        XtX = X_outer_train.T @ X_outer_train
        ## Extract optimal regression model
        optimal_index = reg_validation_error.sum(axis=1).argmin()
        optimal_regularization_parameter = regularization_parameters[optimal_index]
        ## Prepare matrix of regularization terms
        lambdas = optimal_regularization_parameter * np.eye(M)
        lambdas[0, 0] = 0
        ## Solve for model weights
        w = np.linalg.solve(XtX + lambdas, Xty).squeeze()
        ## Determine test error
        y_estimates = X_outer_test @ w.T
        mse = mean_squared_error(y_outer_test, y_estimates)
        reg_test_error[outer_fold, 0] = optimal_regularization_parameter
        reg_test_error[outer_fold, 1] = mse

        # Baseline Outer Fold
        print(f'Training baseline model for outer fold n. {outer_fold}')
        baseline_model = lambda x: y_outer_train.mean()
        y_estimates = baseline_model(X_outer_test)
        mse = mean_squared_error(y_outer_test, y_estimates)
        baseline_test_error[outer_fold] = mse

    print('Two layer cross validation finished for all models')

    return ann_test_error, reg_test_error, baseline_test_error


def make_table(k1, ann_test_error, reg_test_error, baseline_test_error):
    rows = []
    for i in range(k1):
        row = (
            f'{k1} '
            f'& {ann_test_error[i, 0]} & {ann_test_error[i, 1]} '
            f'& {reg_test_error[i, 0]} & {reg_test_error[i, 1]} '
            f'& {baseline_test_error[i]} \\\\'
        )
        rows.append(row)
    return '\n'.join(rows)


if __name__ == '__main__':
    main()
