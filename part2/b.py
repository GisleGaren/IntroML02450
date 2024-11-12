import pandas as pd
import numpy as np
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
    models = [None]

    return two_level_cross_validation(X, y, k1, k2, models)


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
    # baseline
    return lambda x: y.mean()


def test(model, X, y):
    error_accumulator = 0.0
    for i, x in enumerate(X):
        y_hat = model(x)
        error_accumulator += (y[i] - y_hat) * (y[i] - y_hat)
    return error_accumulator / y.shape[0]
