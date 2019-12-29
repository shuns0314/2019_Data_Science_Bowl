import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)


def main():
    """main."""
    args = parser.parse_args()
    train_df = pd.read_csv(f"data/processed/{args.train_csv}.csv", index_col=0)
    # print(train_df.head())
    y = train_df['accuracy_group']
    x = train_df.drop('accuracy_group', axis=1)
    lgb_regression(x, y)


def lgb_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)
    print(x_train.columns)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
    }

    model = lgb.train(params=lgb_params,
                      train_set=lgb_train,
                      valid_sets=lgb_val)

    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(rmse)
    print(y_pred)


if __name__ == "__main__":
    main()
