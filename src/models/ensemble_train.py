import argparse
import os
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

from models.loss_function import qwk, lgb_qwk
from models.post_process import post_processing, threshold
from models.closs_validation import stratified_group_k_fold
from features.preprocess_feature_main import PreprocessTime
from lgb import lgb_model


parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)
parser.add_argument('--test_csv', type=str, default=None)
parser.add_argument('--name', type=str)


PARAMS = {
        'threshold_0': 1.12,
        'threshold_1': 1.62,
        'threshold_2': 2.20
        }


def main():
    """main."""
    args = parser.parse_args()
    train_df = pd.read_csv(f"data/processed/{args.train_csv}.csv", index_col=0)
    train_df.columns = train_df.columns.str.replace(',', '')
    test_df = pd.read_csv(f"data/processed/{args.test_csv}.csv", index_col=0)
    test_df.columns = test_df.columns.str.replace(',', '')

    # folderの作成
    if args.name is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'lgb_{args.train_csv}_{now}'
    if not os.path.exists(f'models/{args.name}'):
        os.makedirs(f'models/{args.name}')

    dir_path = f'models/{args.name}'

    # 書き込み用のファイルの作成
    with open(f'{dir_path}/loss.txt', mode='w') as f:
        f.write(f"train_csv: {args.name}\n")

    num_fold = 4
    total_val_pred = np.zeros([train_df.shape[0], ])
    test_list = []
    importance_list = []

    kfold = KFold(num_fold, random_state=77)
    seed_list = [77, 5, 8, 16, 42, 10,
                 777, 98, 356, 561, 36,
                 56, 1, 212, 444, 612]
    for fold_ind, (train_ind, val_ind) in enumerate(
            kfold.split(train_df)):

        loss_list = []
        val_list = []

        for seed in seed_list:
            print(train_df.iloc[train_ind].shape)
            print(train_df.iloc[val_ind].shape)
            print(test_df.shape)
            loss, val_pred, test_pred, all_importance = train_and_predict(
                train_df.iloc[train_ind], train_df.iloc[val_ind], test_df,
                seed=seed, path=dir_path)
            print(loss)
            loss_list.append(loss)
            val_list.append(val_pred)
            test_list.append(test_pred)
            importance_list.append(all_importance)

        loss = np.mean(loss_list)
        with open(f'models/{args.name}/loss.txt', mode='a') as f:
            f.write(f"fold_{fold_ind}_seed_average_loss: {loss}\n")
        val_pred = pd.concat(val_list, axis=1)
        total_val_pred[val_ind] = val_pred.mean(axis=1)

    test_pred = pd.concat(test_list, axis=1)
    test_pred.to_csv(f'models/{args.name}/check_cv.csv')
    test_pred = test_pred.mean(axis=1)

    func = np.frompyfunc(threshold, 2, 1)
    final_val_pred = func(total_val_pred, PARAMS)
    loss = qwk(final_val_pred, train_df['accuracy_group'])
    print(loss)
    with open(f'models/{args.name}/loss.txt', mode='a') as f:
        f.write(f"seed_brending_loss: {loss}\n")
    pred_df = func(test_pred, PARAMS)
    pred_df.to_csv(f'models/{args.name}/submission.csv', header=False)

    all_importance = pd.concat(importance_list, axis=1)
    all_importance.to_csv(f'models/{args.name}/all_importance.csv')


def train_and_predict(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      test_df: pd.DataFrame, seed: int, path, model: str) -> pd.DataFrame:
    num_fold = 4
    groups = np.array(train_df['installation_id'])

    y = train_df['accuracy_group'].reset_index(drop=True)
    print(y)
    x = train_df.drop('accuracy_group', axis=1).reset_index(drop=True)
    x = x.drop('installation_id', axis=1)

    # val_y = val_df['accuracy_group']
    val_x = val_df.drop('accuracy_group', axis=1)
    val_x = val_x.drop('installation_id', axis=1)

    test_x = test_df.drop('accuracy_group', axis=1)
    test_x = test_x.drop('installation_id', axis=1)

    if model in ['NN', 'kNN']:
        prerpocess

    total_inner_val_pred = np.zeros([y.shape[0], ])
    total_outer_val_pred = np.zeros([val_x.shape[0], num_fold])
    total_test_pred = np.zeros([test_df.shape[0], num_fold])

    all_importance = []
    inner_val_index_all = []

    for fold_ind, (train_ind, inner_val_ind) in enumerate(
            stratified_group_k_fold(
                X=x, y=y, groups=groups, k=num_fold, seed=seed)):

        # # lgb
        lgb_train_model, y_pred, importance, test_x, val_x = lgb_model(
            x, y, train_ind, inner_val_ind, test_x, seed=seed, val_x=val_x)
        total_inner_val_pred[inner_val_ind] = y_pred

        val_pred = lgb_train_model.predict(
            val_x, num_iteration=lgb_train_model.best_iteration)
        total_outer_val_pred[:, fold_ind] = val_pred

        test_pred = lgb_train_model.predict(
            test_x, num_iteration=lgb_train_model.best_iteration)
        total_test_pred[:, fold_ind] = test_pred

        all_importance.append(importance)
        inner_val_index_all += inner_val_ind

    all_importance = pd.concat(all_importance, axis=1)

    # print(total_val_pred)
    func = np.frompyfunc(threshold, 2, 1)
    print(y)
    print(type(y))
    y = y[inner_val_index_all].values
    total_val_pred = total_inner_val_pred[inner_val_index_all]

    loss = qwk(func(total_val_pred, PARAMS), y)

    with open(f'{path}/loss.txt', mode='a') as f:
        f.write(f"lgb_loss: {loss}\n")

    total_outer_val_pred = pd.DataFrame(total_outer_val_pred).mean(axis=1)
    total_test_pred = pd.DataFrame(total_test_pred).mean(axis=1)

    return loss, total_outer_val_pred, total_test_pred, all_importance


if __name__ == "__main__":
    main()
