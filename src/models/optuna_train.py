import argparse
import os
from datetime import datetime
import uuid
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import optuna

from loss_function import qwk, lgb_qwk
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
    train_df.columns = train_df.columns.str.replace(',', '')

    # folderの作成
    if args.name is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'lgb_{args.train_csv}_{now}'
    if not os.path.exists(f'models/{args.name}'):
        os.makedirs(f'models/{args.name}')

    if args.test_csv == 'None':
        model, all_importance, pred_df, loss = lgb_regression(train_df)
    else:
        # coefficient = train_df['accuracy_group'].value_counts(sort=False)/len(train_df['accuracy_group'])
        test_df = pd.read_csv(f"data/processed/{args.test_csv}.csv", index_col=0)
        lgb_regression(train_df, test_df)
        # model, all_importance, pred_df, loss = lgb_regression(train_df, test_df)
        # all_importance.to_csv(f'models/{args.name}/all_importance.csv')
        # pred_df.to_csv(f'models/{args.name}/check_cv.csv')
        # pred_df = pred_df.apply(lambda x: x.mode()[0] if len(x.mode()) == 1 else coefficient[x.mode()].idxmax(), axis=1)
        # pred_df.to_csv(f'models/{args.name}/submission.csv', header=False)

    # with open(f'models/{args.name}/loss.txt', mode='w') as f:
    #     print(f"val_loss: {loss}")
    #     f.write(f"val_loss: {loss}\n")
    #     f.write(f"train_csv: {args.name}\n")

    # # modelのsave
    # joblib.dump(model, f'models/{args.name}/model_{args.name}.pkl')

    # paramsのsave
    # with open(f'models/{args.name}/params_{args.name}.pkl', 'wb') as handle:
    #     pickle.dump(params, handle)


def lgb_regression(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> pd.DataFrame:

    def objective(trial):
        # 試行にUUIDを設定
        trial_uuid = str(uuid.uuid4())
        trial.set_user_attr("uuid", trial_uuid)

        num_fold = 8

        y = train_df['accuracy_group']
        x = train_df.drop('accuracy_group', axis=1)
        groups = np.array(x['installation_id'])

        lgb_params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "metric": 'rmse',
            "verbosity": 0,
            "early_stopping_round": 50,
            "max_depth": trial.suggest_int('max_depth', 3, 9),
            "learning_rate": 0.1,
            "num_leaves": trial.suggest_int("num_leaves", 10, 500),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.0, 1.0),
            "subsample": trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1),
            "min_child_weight": trial.suggest_loguniform('min_child_weight', 1e-1, 1e+1),
            "colsample_bytree": trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1),
            "min_gain_to_split": trial.suggest_loguniform('min_gain_to_split', 1e-8, 1e-0)
            }

        # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'rmse')

        x = x.drop('installation_id', axis=1)
        # total_pred = np.zeros(y.shape)
        total_loss = []
        func = np.frompyfunc(threshold, 2, 1)

        if test_df is not None:
            test_x = test_df.drop('accuracy_group', axis=1)
            test_x = test_x.drop('installation_id', axis=1)
            total_test_pred = np.zeros([test_df.shape[0], num_fold])
            print(total_test_pred.shape)

        all_importance = []

        for fold_ind, (train_ind, test_ind) in enumerate(
                stratified_group_k_fold(X=x, y=y, groups=groups, k=num_fold, seed=77)):
            # print(dev_ind)
            x_train = x.iloc[train_ind]
            y_train = y.iloc[train_ind]
            # x_val = x.iloc[val_ind]
            # y_val = y.iloc[val_ind]
            x_test = x.iloc[test_ind]
            y_test = y.iloc[test_ind]

            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)

            model = lgb.train(params=lgb_params,
                              train_set=lgb_train,
                              valid_sets=lgb_val,
                              feval=lgb_qwk,
                              num_boost_round=1000,
                              early_stopping_rounds=50)

            # y_val_pred = model.predict(x_test, num_iteration=model.best_iteration)

            params = {
                'threshold_0': 1.12,
                'threshold_1': 1.62,
                'threshold_2': 2.20
                },
            # params = post_processing(y_val, y_val_pred)
            # all_params.append(params)

            y_pred = model.predict(x_test, num_iteration=model.best_iteration)
            all_importance.append(
                pd.DataFrame(model.feature_importance('gain'), index=x_train.columns))
            y_pred = func(y_pred, params)

            total_loss.append(qwk(y_pred, y_test))
            # total_pred[test_ind] = y_pred

            if test_df is not None:
                test_pred = model.predict(
                    test_x, num_iteration=model.best_iteration)
                test_pred = func(test_pred, params)
                total_test_pred[:, fold_ind] = test_pred
        all_importance = pd.concat(all_importance, axis=1)

        loss = np.mean(total_loss)

        # モデルの保存
        if not os.path.exists("lgb_output"):
            os.mkdir("lgb_output")
        with open("lgb_output/"+f"{trial_uuid}.pkl", "wb") as fp:
            pickle.dump(model, fp)

        return loss

    study = optuna.create_study(direction='maximize')

    # https://optuna.readthedocs.io/en/stable/reference/study.html#optuna.study.Study.optimize
    study.optimize(func=objective,  # 実行する関数
                   n_trials=200,  # 試行回数
                   timeout=None,  # 与えられた秒数後に学習を中止します。default=None
                   n_jobs=-1)  # 並列実行するjob数

    # if test_df is None:
    #     return model, all_importance, loss
    # else:
    #     return model, all_importance, pd.DataFrame(total_test_pred), loss


if __name__ == "__main__":
    main()
