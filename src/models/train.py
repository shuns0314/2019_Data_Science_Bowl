import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna

from loss_function import qwk
from sklearn.externals import joblib


# load model
# gbm_pickle = joblib.load('lgb.pkl')


parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)


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
    y_pred = threshold(y_pred, params)

    func = np.frompyfunc(threshold, 2, 1)
    post_pred = func(y_pred, params)
    loss = qwk(y_val, post_pred)
    print(f"val_loss* {loss}")

    # joblib.dump(model, 'lgb.pkl')


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


def post_processing(y_test, y_pred):

    def objectives(trial):
        params = {
            'threshold_0': trial.suggest_uniform('threshold_0', 0.0, 3.0),
            'threshold_1': trial.suggest_uniform('threshold_1', 0.0, 3.0),
            'threshold_2': trial.suggest_uniform('threshold_2', 0.0, 3.0),
        }
        func = np.frompyfunc(threshold, 2, 1)
        post_pred = func(y_pred, params)
        loss = qwk(y_test, post_pred)

        return loss

    study = optuna.create_study(direction='maximize')
    study.optimize(objectives, n_trials=100)

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')

    print(f'  Params: ')
    for key, value in trial.params.ite2ms():
        print(f'    {key}: {value}')

    return trial.params.ite2ms()


def threshold(x, params):
    if x < params['threshold_0']:
        y = 0
    elif x < params['threshold_1']:
        y = 1
    elif x < params['threshold_2']:
        y = 2
    else:
        y = 3
    return y


if __name__ == "__main__":
    main()
