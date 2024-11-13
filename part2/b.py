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
    k1 = 2
    k2 = 2
    models = ['baseline', 'ann']

    print(ann_two_level_cv(k1, k2, X, y))

    # return two_level_cross_validation(X, y, k1, k2, models)


def two_level_cross_validation(X, y, k1, k2, models):
    s = len(models)
    test_error = np.empty(k1)
    outer_cv = model_selection.KFold(k1, shuffle=True)
    outer_split = enumerate(outer_cv.split(X, y))

    print('| Fold number '
          '| ANN test error '
          '| Linear regression test error '
          '| Baseline test error |')
    print('|---|-----|-----|---------|')
    for outer_fold, (outer_train_index, outer_test_index) in outer_split:
        X_outer_train = X[outer_train_index]
        y_outer_train = y[outer_train_index]
        X_outer_test = X[outer_test_index]
        y_outer_test = y[outer_test_index]

        validation_error = np.empty((s, k2))
        inner_cv = model_selection.KFold(k2, shuffle=True)
        inner_split = enumerate(inner_cv.split(X_outer_train, y_outer_train))

        for inner_fold, (inner_train_index, inner_test_index) in inner_split:
            X_inner_train = X_outer_train[inner_train_index]
            y_inner_train = y_outer_train[inner_train_index]
            X_inner_test = X_outer_train[inner_test_index]
            y_inner_test = y_outer_train[inner_test_index]

            for i, model in enumerate(models):
                model = train(model, X_inner_train, y_inner_train)
                error = test(model, X_inner_test, y_inner_test)
                error /= X_outer_train.shape[0]
                error *= X_inner_train.shape[0]
                validation_error[i, inner_fold] = error

        optimal_index = validation_error.sum(axis=1).argmin()
        model = models[optimal_index]
        model = train(model, X_outer_train, y_outer_train)
        error = test(model, X_outer_test, y_outer_test)
        error /= X.shape[0]
        error *= X_outer_train.shape[0]
        print(f'| {outer_fold} | N/A | N/A | {error:7.2f} |')
        test_error[outer_fold] = error

    return test_error.sum()


def train(model, X, y):
    if model == 'ann':
        return train_ann(X, y)
    elif model == 'regression':
        assert False
    elif model == 'baseline':
        return train_baseline(X, y)
    else:
        assert False


def ann_two_level_cv(k1, k2, X, y):
    # Number of features
    M = X.shape[1]
    # Error calculation
    mean_squared_error = lambda y, y_est: sum((y - y_est) ** 2) / len(y)

    # Data for two level cross validation of ANN
    ## model parameters
    hidden_units_parameters = [1]
    ## Training parameters
    n_replicates = 1
    max_iter = 10_000
    loss_fn = torch.nn.MSELoss()
    ## The number of models to test
    s = len(hidden_units_parameters)
    ## The list of models to test
    models = [
        lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden_units, 1),
        ) for n_hidden_units in hidden_units_parameters
    ]

    # Data for two level cross validation of regularized linear regression
    ## model parameters
    regularization_parameters = [0]
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
            for i, model in enumerate(models):
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

        # ANN Outer Fold
        print(f'Training optimal ANN for outer fold n. {outer_fold}')
        ## Extract the optimal ANN model
        optimal_index = ann_validation_error.sum(axis=1).argmin()
        optimal_n_hidden_units = hidden_units_parameters[optimal_index]
        model = models[optimal_index]
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

        # Baseline Outer Fold
        print(f'Training baseline model for outer fold n. {outer_fold}')
        baseline_model = lambda x: y_outer_train.mean()
        y_estimates = baseline_model(X_outer_test)
        mse = mean_squared_error(y_outer_test, y_estimates)
        baseline_test_error[outer_fold] = mse

    print('Two layer cross validation finished for all models')

    return ann_test_error, baseline_test_error


def train_baseline(X, y):
    return lambda x: y.mean()


def test(model, X, y):
    error_accumulator = 0.0
    for i, x in enumerate(X):
        y_hat = model(x)
        error_accumulator += (y[i] - y_hat) * (y[i] - y_hat)
    return error_accumulator / y.shape[0]
