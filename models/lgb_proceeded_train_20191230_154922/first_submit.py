from typing import Tuple

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# /code/src/features/make_feature.py
RAW_PATH = "/code/data/raw"


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               train_labels_df: pd.DataFrame):
    """前処理のメイン関数."""
    activities_map = encode_title(train_df, test_df)
    assert len(activities_map) == 44, f'想定値: 44, 入力値: {len(activities_map)}'

    train_df['title'] = train_df['title'].map(activities_map)
    test_df['title'] = test_df['title'].map(activities_map)
    train_labels_df['title'] = train_labels_df['title'].map(activities_map)
    win_code = make_event_code(activities_map)
    assert len(win_code) == 44, f'想定値: 44, 入力値: {len(win_code)}'

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    compile_history = CompileHistory(win_code=win_code)
    compiled_train = compile_history.compile_history_data(train_df)

    compile_history = CompileHistory(win_code=win_code, test_set=True)
    compiled_test = compile_history.compile_history_data(test_df)

    return compiled_train, compiled_test


def encode_title(train_df: pd.DataFrame,
                 test_df: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """各DataFrameのtitleを番号に変換するためのmethod.

    return:
    activites_map = {'Sandcastle Builder (Activity)': 0,
      'Dino Drink': 1,
      'Mushroom Sorter (Assessment)': 2,
      'Crystal Caves - Level 2': 3,...}
    """
    # setに入れる重複がなくなる.
    train_title = set(train_df['title'].unique())
    test_title = set(test_df['title'].unique())
    # unionは和集合
    list_of_user_activities = list(train_title.union(test_title))
    activities_map = dict(
        zip(list_of_user_activities, np.arange(len(list_of_user_activities)))
        )
    return activities_map


def make_event_code(activities_map: dict):
    """
    return:
    event_code={0: 4100,
    1: 4100,
    2: 4100,...} <- {title: event_code}
    """
    win_code = dict(
        zip(
            activities_map.values(),
            (4100*np.ones(len(activities_map))).astype('int'))
        )
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    return win_code


class GetData():
    """各installation_idのおける過去のゲームの実績をまとめるmethod."""

    def __init__(self, win_code, test_set=False):
        self.win_code = win_code
        self.user_activities_count = {
            'Clip': 0,
            'Activity': 0,
            'Assessment': 0,
            'Game': 0
            }
        self.last_activity = 0
        self.test_set = test_set
        self.count_actions = 0

    def process(self, user_sample):
        all_assessments = []

        get_assesments = GetAssessmentFeature(self.win_code,
                                              test_set=self.test_set)

        # まずgame_sessionでgroupbyする
        for i, session in user_sample.groupby('game_session', sort=False):
            session_type = session['type'].iloc[0]

            # session数が1以下を省く
            if self.test_set is True:
                second_condition = True
            else:
                if len(session) > 1:
                    second_condition = True
                else:
                    second_condition = False

            features: dict = self.user_activities_count.copy()

            # session typeがAssessmentのやつだけ、カウントする。
            if (session_type == 'Assessment') & (second_condition):
                features = get_assesments.process(session, features)

                if features is not None:
                    # 特徴量に前回までのゲームの回数を追加
                    features['count_actions'] = self.count_actions
                    all_assessments.append(features)

            self.count_actions += len(session)

            # second_conditionがFalseのときは、user_activities_countのみ増える。
            if self.last_activity != session_type:
                self.user_activities_count[session_type] += 1
                self.last_activitiy = session_type

        if self.test_set:
            return all_assessments[-1]
        return all_assessments


# make_feature
class GetAssessmentFeature:

    def __init__(self, win_code, test_set=False):
        self.test_set = test_set
        self.win_code = win_code
        self.accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
        self.mean_accuracy_group = 0  # accuracy_groupsの平均
        self.count_accuracy = 0
        self.count_correct_attempts = 0  # 成功
        self.count_uncorrect_attempts = 0  # 失敗
        self.counter = 0
        self.durations = []
        self.true_attempts = 0
        self.false_attempts = 0

    def process(self, session, features):
        all_attempts = session.query(
                f"event_code == {self.win_code[session['title'].iloc[0]]}"
                )
        assert type(all_attempts) == pd.DataFrame

        features['session_title'] = session['title'].iloc[0]

        # 特徴量に前回までの正解数と失敗数追加
        features = self.add_count_attempts(features, all_attempts)

        # 特徴量に前回までのaccuracyを追加
        count_acc = self.count_accuracy/self.counter if self.counter > 0 else 0
        features['count_accuracy'] = count_acc
        accuracy = self.calc_accuracy(self.true_attempts, self.false_attempts)
        self.count_accuracy += accuracy

        # 特徴量に前回までの平均ゲーム時間を追加
        features = self.add_duration_mean(features, session)

        # 特徴量に今回のacc_groupを追加
        features = self.add_accuracy_group(features, accuracy)

        # 特徴量に前回までのacc_groupの平均を追加
        if self.counter > 0:
            mean_acc_gp = self.mean_accuracy_group/self.counter
        else:
            mean_acc_gp = 0
        features['mean_accuracy_group'] = mean_acc_gp

        # 特徴量に前回までのacc_groupの数を追加
        features.update(self.accuracy_groups)
        self.mean_accuracy_group += features['accuracy_group']
        self.accuracy_groups[features['accuracy_group']] += 1

        self.counter += 1

        # trainで試行回数が0のものを除外する。
        if self.test_set is True:
            return features
        else:
            if self.true_attempts + self.false_attempts == 0:
                # print(0)
                pass
            else:
                # print(features)
                return features

    def add_duration_mean(self, df, session):
        if self.durations == []:
            df['duration_mean'] = 0
        else:
            df['duration_mean'] = np.mean(self.durations)

        self.durations.append(
            (session.iloc[-1, 2] - session.iloc[0, 2]).seconds
            )
        return df

    def add_accuracy_group(self,
                           df: pd.DataFrame,
                           accuracy: int) -> pd.DataFrame:
        if accuracy == 0:
            df['accuracy_group'] = 0
        elif accuracy == 1:
            df['accuracy_group'] = 3
        elif accuracy == 0.5:
            df['accuracy_group'] = 2
        else:
            df['accuracy_group'] = 1
        return df

    def add_count_attempts(self,
                           df: pd.DataFrame,
                           all_attempts: pd.DataFrame):
        """result: 'true' or 'false'."""
        # correct
        df['count_correct_attempts'] = self.count_correct_attempts
        self.true_attempts = all_attempts['event_data'].str.contains('true').sum()
        self.count_correct_attempts += self.true_attempts

        # uncorrect
        df['count_uncorrect_attempts'] = self.count_uncorrect_attempts
        self.false_attempts = all_attempts['event_data'].str.contains('false').sum()
        self.count_uncorrect_attempts += self.false_attempts
        return df

    def calc_accuracy(self, true, false):
        accuracy = true/(true+false) if (true+false) != 0 else 0
        return accuracy


class CompileHistory:
    def __init__(self, win_code, test_set: bool = False):
        self.win_code = win_code
        self.test_set = test_set

    def compile_history_data(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """過去のデータを、installation_idごとのデータにまとめる."""
        get_data = GetData(win_code=self.win_code, test_set=self.test_set)
        compiled_data = Parallel(n_jobs=-1)(
            [delayed(self.get_data_for_sort)(
                user_sample, i, get_data
                ) for i, (_, user_sample) in enumerate(
                    df.groupby('installation_id', sort=False)
                    )]
        )
        compiled_data.sort(key=lambda x: x[1])
        compiled_data = [t[0] for t in compiled_data]
        if self.test_set is False:
            compiled_data = [data for inner_data in compiled_data for data in inner_data]
        # print(compiled_data)

        return pd.DataFrame(compiled_data)

    def get_data_for_sort(self,
                          data: pd.DataFrame,
                          i: int,
                          get_data: GetData) -> Tuple[pd.DataFrame, int]:
        compiled_data = get_data.process(data)
        # print(f"compiled_data: {compiled_data}")
        return compiled_data, i


def qwk(a1, a2):
    """
    Source:
    https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


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
    print(trial.params)
    for param in trial.params:
        print(f'    {param}: {trial.params[param]}')

    return trial.params


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


# train
def train_main(train_df):
    """main."""
    y = train_df['accuracy_group']
    x = train_df.drop('accuracy_group', axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15)
    model, params = lgb_regression(x_train, y_train)
    y_pred = model.predict(x_val)
    func = np.frompyfunc(threshold, 2, 1)
    post_pred = func(y_pred, params)
    loss = qwk(y_val, post_pred)
    print(f"val_loss: {loss}")
    return model, params


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


def predict_main(test_df, model, params):
    test_df = test_df.drop('accuracy_group', axis=1)

    pred_df = model.predict(test_df)
    func = np.frompyfunc(threshold, 2, 1)
    post_pred = func(pred_df, params)
    submission = pd.read_csv(f'{RAW_PATH}/sample_submission.csv')
    submission['accuracy_group'] = post_pred.astype(int)
    print(submission.dtypes)
    submission.to_csv(f'submission.csv', index=False)


def main():
    """main."""
    print('--read_csv--')
    train_df = pd.read_csv(f"{RAW_PATH}/train.csv")
    test_df = pd.read_csv(f"{RAW_PATH}/test.csv")
    train_labels_df = pd.read_csv(f"{RAW_PATH}/train_labels.csv")

    print('--compile--')
    compiled_train, compiled_test = preprocess(train_df,
                                               test_df,
                                               train_labels_df)
    print('--train--')
    model, params = train_main(compiled_train)

    print('--predict--')
    predict_main(compiled_test, model, params)


if __name__ == "__main__":
    main()
