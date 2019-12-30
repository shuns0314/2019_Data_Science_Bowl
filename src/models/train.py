import argparse
import os
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

from loss_function import qwk
from post_process import post_processing, threshold


parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)
parser.add_argument('--name', type=str)


def main():
    """main."""
    args = parser.parse_args()
    train_df = pd.read_csv(f"data/processed/{args.train_csv}.csv", index_col=0)
    # print(train_df.head())
    y = train_df['accuracy_group']
    x = train_df.drop('accuracy_group', axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15)
    model, params = lgb_regression(x_train, y_train)
    y_pred = model.predict(x_val)
    # print(type(y_pred))
    func = np.frompyfunc(threshold, 2, 1)
    post_pred = func(y_pred, params)
    loss = qwk(y_val, post_pred)
    print(f"val_loss: {loss}")

    if args.name is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'lgb_{args.train_csv}_{now}'

    if not os.path.exists(f'models/{args.name}'):
        os.makedirs(f'models/{args.name}')

    # modelのsave
    joblib.dump(model, f'models/{args.name}/model_{args.name}.pkl')

    # paramsのsave
    with open(f'models/{args.name}/params_{args.name}.pkl', 'wb') as handle:
        pickle.dump(params, handle)


def lgb_regression(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
    }

    model = lgb.train(params=lgb_params,
                      train_set=lgb_train,
                      valid_sets=lgb_val)

    y_pred = model.predict(x_val, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    print(rmse)
    print(y_pred)

    params = post_processing(y_val, y_pred)

    return model, params


if __name__ == "__main__":
    main()
