# CV: 0.553 LB: 0.519 num_fold = 10
# CV: 0.555 LB: 0.527 num_fold = 8
from typing import Tuple
import random
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import optuna
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# /code/src/features/make_feature.py
RAW_PATH = "/code/data/raw"


def preprocess(train: pd.DataFrame,
               test: pd.DataFrame,
               train_labels: pd.DataFrame):
    """前処理のメイン関数."""
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    # print(list_of_event_code)
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    print('title_column')
    print(train.head())
    print('asses_title')
    print(assess_titles)
    print('list_of_event_code')
    print(list_of_event_code)
    print('list_of_event_id')
    print(list_of_event_id)
    print('activities_labels')
    print(activities_labels)
    print('all_title_event_code')
    print(all_title_event_code)

    compile_history = CompileHistory(
        win_code=win_code,
        assess_titles=assess_titles,
        list_of_event_code=list_of_event_code,
        list_of_event_id=list_of_event_id,
        activities_labels=activities_labels,
        all_title_event_code=all_title_event_code
        )
    compiled_train = compile_history.compile_history_data(train)

    compile_history = CompileHistory(
        win_code=win_code,
        assess_titles=assess_titles,
        list_of_event_code=list_of_event_code,
        list_of_event_id=list_of_event_id,
        activities_labels=activities_labels,
        all_title_event_code=all_title_event_code,
        test_set=True)
    compiled_test = compile_history.compile_history_data(test)

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

    def __init__(self,
                 win_code,
                 assess_titles,
                 list_of_event_code,
                 list_of_event_id,
                 activities_labels,
                 all_title_event_code,
                 test_set=False):

        self.win_code = win_code
        self.assess_titles = assess_titles
        self.list_of_event_code = list_of_event_code
        self.list_of_event_id = list_of_event_id
        self.activities_labels = activities_labels
        self.all_title_event_code = all_title_event_code

        self.user_activities_count = {
            'Clip': 0,
            'Activity': 0,
            'Assessment': 0,
            'Game': 0
            }
        self.last_activity = 0
        self.test_set = test_set
        self.count_actions = 0

        # print(self.list_of_event_code)
        self.event_code_count: Dict[str, int] = {ev: 0 for ev in self.list_of_event_code}
        self.event_id_count: Dict[str, int] = {eve: 0 for eve in self.list_of_event_id}
        self.title_count: Dict[str, int] = {eve: 0 for eve in self.activities_labels.values()}
        self.title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in self.all_title_event_code}

    def process(self, user_sample, installation_id):

        all_assessments = []
        get_assesments = GetAssessmentFeature(self.win_code,
                                              self.assess_titles,
                                              self.list_of_event_code,
                                              self.list_of_event_id,
                                              self.activities_labels,
                                              self.all_title_event_code,
                                              test_set=self.test_set)
        # まずgame_sessionでgroupbyする
        for i, session in user_sample.groupby('game_session', sort=False):
            session_type = session['type'].iloc[0]
            # print(session_type)
            # game_session数が1以下を省く
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
                    features['installation_id'] = installation_id
                    # 特徴量に前回までのゲームの回数を追加
                    features['count_actions'] = self.count_actions

                    features.update(self.event_code_count.copy())
                    features.update(self.event_id_count.copy())
                    features.update(self.title_count.copy())
                    features.update(self.title_event_code_count.copy())

                    all_assessments.append(features)

            self.count_actions += len(session)
            self.event_code_count = self.update_counters(
                session, self.event_code_count, "event_code")
            self.event_id_count = self.update_counters(
                session, self.event_id_count, "event_id")
            self.title_count = self.update_counters(
                session, self.title_count, 'title')
            self.title_event_code_count = self.update_counters(
                session, self.title_event_code_count, 'title_event_code')

            # second_conditionがFalseのときは、user_activities_countのみ増える。
            if self.last_activity != session_type:
                self.user_activities_count[session_type] += 1
                self.last_activitiy = session_type

        if self.test_set:
            return all_assessments[-1]
        return all_assessments

    def update_counters(self, session, counter: dict, col: str):
        num_of_session_count = Counter(session[col])
        for k in num_of_session_count.keys():
            x = k
            if col == 'title':
                x = self.activities_labels[k]
            counter[x] += num_of_session_count[k]
        return counter


# make_feature
class GetAssessmentFeature:

    def __init__(self,
                 win_code,
                 assess_titles,
                 list_of_event_code,
                 list_of_event_id,
                 activities_labels,
                 all_title_event_code,
                 test_set=False):

        self.list_of_event_code = list_of_event_code
        self.list_of_event_id = list_of_event_id
        self.activities_labels = activities_labels
        self.all_title_event_code = all_title_event_code
        self.assess_titles = assess_titles

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

        self.last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

        self.so_cool = 0
        self.greatjob = 0

    def process(self, session, features):
        all_attempts = session.query(
                f"event_code == {self.win_code[session['title'].iloc[0]]}"
                )
        assert type(all_attempts) == pd.DataFrame

        features['session_title'] = session['title'].iloc[0]
        session_title = session['title'].iloc[0]
        # print(self.activities_labels[session_title])
        session_title_text = self.activities_labels[session_title]
        # 特徴量に前回までの正解数と失敗数追加
        features = self.add_count_attempts(features, all_attempts)

        # 特徴量に前回までのaccuracyを追加
        count_acc = self.count_accuracy/self.counter if self.counter > 0 else 0
        features['count_accuracy'] = count_acc
        accuracy = self.calc_accuracy(self.true_attempts, self.false_attempts)
        self.count_accuracy += accuracy

        features.update(self.last_accuracy_title.copy())
        # print(accuracy)
        self.last_accuracy_title['acc_' + session_title_text] = accuracy

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
    def __init__(self,
                 win_code,
                 assess_titles,
                 list_of_event_code,
                 list_of_event_id,
                 activities_labels,
                 all_title_event_code,
                 test_set: bool = False,):

        self.win_code = win_code
        self.assess_titles = assess_titles
        self.list_of_event_code = list_of_event_code
        self.list_of_event_id = list_of_event_id
        self.activities_labels = activities_labels
        self.all_title_event_code = all_title_event_code
        self.test_set = test_set

    def compile_history_data(self,
                             df: pd.DataFrame) -> pd.DataFrame:
        """過去のデータを、installation_idごとのデータにまとめる."""
        print(self.list_of_event_code)
        get_data = GetData(
            win_code=self.win_code,
            assess_titles=self.assess_titles,
            list_of_event_code=self.list_of_event_code,
            list_of_event_id=self.list_of_event_id,
            activities_labels=self.activities_labels,
            all_title_event_code=self.all_title_event_code,
            test_set=self.test_set,
            )
        compiled_data = Parallel(n_jobs=-1, verbose=10)(
            [delayed(self.get_data_for_sort)(
                user_sample, i, get_data, installation_id
                ) for i, (installation_id, user_sample) in enumerate(
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
                          get_data: GetData,
                          installation_id: str) -> Tuple[pd.DataFrame, int]:
        compiled_data = get_data.process(data, installation_id)
        # print(f"compiled_data: {compiled_data}")
        return compiled_data, i


def lgb_qwk(preds, data):
    params = {
        'threshold_0': 1.12,
        'threshold_1': 1.62,
        'threshold_2': 2.20
        },

    func = np.frompyfunc(threshold, 2, 1)
    test_pred = func(preds, params)
    loss = qwk(test_pred, data.get_label())
    return 'qwk', loss, True


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


def stratified_group_k_fold(X, y, groups, k, seed=None):
    np.random.seed(seed)
    # ラベルの数をカウント
    labels_num = np.max(y) + 1
    # 各グループのラベルの数をカウントする
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1
    # 各フォールドのラベルの数をカウント
    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)

    for i in range(k):
        test_k = i
        # val_k = i+1 if i+1 != k else 0
        # print(val_k)
        train_groups = all_groups - groups_per_fold[test_k]  #  - groups_per_fold[val_k]
        # val_groups = groups_per_fold[val_k]
        test_groups = groups_per_fold[test_k]
        # print(test_groups)
        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        # val_indices = [i for i, g in enumerate(groups) if g in val_groups]
        # test_indices = {str(g): [i for i, g in enumerate(groups) if g in test_groups]}

        test_indices = []
        n_g = None
        test_list = []
        for i, g in enumerate(groups):
            if g in test_groups:
                if n_g is not None and n_g != g:
                    test_indices.append(test_list)
                    test_list = []
                test_list.append(i)
                n_g = g

        test_indices = [np.random.choice(i) for i in test_indices]
        yield train_indices, test_indices  # val_indices,


def get_distribution(y_vals):
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]


def post_processing(y_test, y_pred):

    def objectives(trial):
        params = {
            'threshold_0': trial.suggest_uniform('threshold_0', 0.0, 1.5),
            'threshold_1': trial.suggest_uniform('threshold_1', 1.2, 2.0),
            'threshold_2': trial.suggest_uniform('threshold_2', 2.0, 3.0),
        }
        func = np.frompyfunc(threshold, 2, 1)
        post_pred = func(y_pred, params)
        loss = qwk(y_test, post_pred)

        return loss

    study = optuna.create_study(direction='maximize')
    study.optimize(objectives, n_trials=150)

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
def train_main(train_df, test_df):
    """main."""
    _, pred_df = lgb_regression(train_df, test_df)
    coefficient = train_df['accuracy_group'].value_counts(sort=False)/len(train_df['accuracy_group'])
    print(coefficient)
    pred_df = pred_df.apply(lambda x: x.mode()[0] if len(x.mode()) == 1 else coefficient[x.mode()].idxmax(), axis=1)
    return pred_df


def lgb_regression(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> pd.DataFrame:

    num_fold = 8

    y = train_df['accuracy_group']
    x = train_df.drop('accuracy_group', axis=1)
    groups = np.array(x['installation_id'])

    lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
        }

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
                          feval=lgb_qwk)

        # y_val_pred = model.predict(x_test, num_iteration=model.best_iteration)

        params = {
            'threshold_0': 1.12,
            'threshold_1': 1.62,
            'threshold_2': 2.20
            },
        # params = post_processing(y_val, y_val_pred)
        # all_params.append(params)

        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        all_importance.append(pd.DataFrame(model.feature_importance('gain'), index=x_train.columns))
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
    print(f"val_loss: {loss}")

    return model, pd.DataFrame(total_test_pred)


def predict_main(pred_df):
    submission = pd.read_csv(f'{RAW_PATH}/sample_submission.csv')
    submission['accuracy_group'] = pred_df.astype(int)
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
    print(compiled_train.shape, compiled_test.shape)
    print('--train--')
    compiled_train.columns = compiled_train.columns.str.replace(',', '')
    compiled_test.columns = compiled_test.columns.str.replace(',', '')
    pred_df = train_main(compiled_train, compiled_test)

    print('--predict--')
    predict_main(pred_df)


if __name__ == "__main__":
    main()
