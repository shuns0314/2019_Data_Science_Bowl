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
from closs_validation import stratified_group_k_fold


parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)
parser.add_argument('--test_csv', type=str, default=None)
parser.add_argument('--name', type=str)


def main():
    """main."""
    args = parser.parse_args()
    train_df = pd.read_csv(f"data/processed/{args.train_csv}.csv", index_col=0)

    # folderの作成
    if args.name is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'lgb_{args.train_csv}_{now}'
    if not os.path.exists(f'models/{args.name}'):
        os.makedirs(f'models/{args.name}')

    if args.test_csv == 'None':
        model, params, pred_df = lgb_regression(train_df)
    else:
        test_df = pd.read_csv(f"data/processed/{args.test_csv}.csv", index_col=0)
        model, params, pred_df = lgb_regression(train_df, test_df)
        pred_df.to_csv(f'models/{args.name}/submission.csv')

    # modelのsave
    joblib.dump(model, f'models/{args.name}/model_{args.name}.pkl')

    # paramsのsave
    with open(f'models/{args.name}/params_{args.name}.pkl', 'wb') as handle:
        pickle.dump(params, handle)


def lgb_regression(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> pd.DataFrame:

    num_fold = 5

    y = train_df['accuracy_group']
    x = train_df.drop('accuracy_group', axis=1)
    groups = np.array(x['installation_id'])

    lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
        }

    x = x.drop('installation_id', axis=1)
    total_pred = np.zeros(y.shape)

    func = np.frompyfunc(threshold, 2, 1)

    if test_df is not None:
        test_x = test_df.drop('accuracy_group', axis=1)
        test_x = test_x.drop('installation_id', axis=1)
        total_test_pred = np.zeros([test_df.shape[0], num_fold])
        print(total_test_pred.shape)

    all_params = []

    for fold_ind, (train_ind, val_ind, test_ind) in enumerate(
            stratified_group_k_fold(X=x, y=y, groups=groups, k=num_fold, seed=77)):
        # print(dev_ind)
        x_train = x.iloc[train_ind]
        y_train = y.iloc[train_ind]
        x_val = x.iloc[val_ind]
        y_val = y.iloc[val_ind]
        x_test = x.iloc[test_ind]
        # y_test = y.iloc[test_ind]

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)

        model = lgb.train(params=lgb_params,
                          train_set=lgb_train,
                          valid_sets=lgb_val)

        y_val_pred = model.predict(x_val, num_iteration=model.best_iteration)

        params = post_processing(y_val, y_val_pred)
        all_params.append(params)

        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        y_pred = func(y_pred, params)
        total_pred[test_ind] = y_pred

        if test_df is not None:
            total_test_pred[:, fold_ind] = model.predict(test_x, num_iteration=model.best_iteration)

    loss = qwk(y, total_pred)
    print(f"val_loss: {loss}")

    if test_df is None:
        return model, all_params
    else:
        return model, all_params, total_test_pred


if __name__ == "__main__":
    main()
