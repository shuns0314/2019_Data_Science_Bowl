import json
from collections import Counter
from typing import Dict

import pandas as pd
import numpy as np


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
        # self.title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in self.all_title_event_code}

        self.total_duration = 0
        self.frequency = 0

        self.game_mean_event_count = 0
        self.accumulated_game_miss = 0

        self.game_round = []
        self.game_duration = []
        self.game_level = []

        self.so_cool = 0
        self.greatjob = 0

    def process(self, user_sample, installation_id):

        all_assessments = []
        get_assesments = GetAssessmentFeature(self.win_code,
                                              self.assess_titles,
                                              self.list_of_event_code,
                                              self.list_of_event_id,
                                              self.activities_labels,
                                              self.all_title_event_code,
                                              test_set=self.test_set)
        first_session = user_sample.iloc[0, 2]
        # まずgame_sessionでgroupbyする
        for i, (session_id, session) in enumerate(
                user_sample.groupby('game_session', sort=False)):
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

            # gameを初めて開始してからの時間を計測
            # if i == 0:
            features: dict = self.user_activities_count.copy()

            if session_type == "Game":

                self.game_mean_event_count = (self.game_mean_event_count + session['event_count'].iloc[-1])/2.0

                game_s = session[session.event_code == 2030]
                misses_cnt = self.count_miss(game_s)
                self.accumulated_game_miss += misses_cnt

                try:
                    game_round_ = json.loads(session['event_data'].iloc[-1])["round"]
                    self.game_round.append(game_round_)

                except:
                    pass

                try:
                    game_duration_ = json.loads(session['event_data'].iloc[-1])["duration"]
                    self.game_duration.append(game_duration_)
                except:
                    pass

                try:
                    game_level_ = json.loads(session['event_data'].iloc[-1])["level"]
                    self.game_level.append(game_level_)
                except:
                    pass

            if session_type == 'Activity':
                # 特徴量に前回までのcoolとgreatの数を追加
                so_cool = session['event_data'].str.contains('SoCool').sum()
                self.so_cool += so_cool

                greatjob = session['event_data'].str.contains('GreatJob').sum()
                self.greatjob += greatjob

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
                    # features.update(self.title_event_code_count.copy())

                    features['total_duration'] = self.total_duration
                    features['frequency'] = self.frequency

                    features['game_mean_event_count'] = self.game_mean_event_count
                    features['accumulated_game_miss'] = self.accumulated_game_miss
                    features['mean_game_round'] = np.mean(self.game_round) if len(self.game_round) != 0 else 0
                    features['max_game_round'] = np.max(self.game_round) if len(self.game_round) != 0 else 0
                    # features['sum_game_round'] = np.max(self.game_round) if len(self.game_round) != 0 else 0
                    features['mean_game_duration'] = np.mean(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['max_game_duration'] = np.max(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['sum_game_duration'] = np.sum(self.game_duration) if len(self.game_duration) != 0 else 0
                    features['mean_game_level'] = np.mean(self.game_level) if len(self.game_level) != 0 else 0
                    # features['max_game_level'] = np.max(self.game_level) if len(self.game_level) != 0 else 0
                    # features['sum_game_level'] = np.sum(self.game_level) if len(self.game_level) != 0 else 0

                    features['so_cool'] = self.so_cool
                    features['greatjob'] = self.greatjob

                    all_assessments.append(features)

            self.total_duration = (session.iloc[-1, 2] - first_session).seconds
            self.count_actions += len(session)
            if self.total_duration == 0:
                self.frequency = 0
            else:
                self.frequency = self.count_actions / self.total_duration

            self.event_code_count = self.update_counters(
                session, self.event_code_count, "event_code")
            self.event_id_count = self.update_counters(
                session, self.event_id_count, "event_id")
            self.title_count = self.update_counters(
                session, self.title_count, 'title')
            # self.title_event_code_count = self.update_counters(
            #     session, self.title_event_code_count, 'title_event_code')

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

    def count_miss(self, df):
        cnt = 0
        for e in range(len(df)):
            x = df['event_data'].iloc[e]
            y = json.loads(x)['misses']
            cnt += y
        return cnt


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
            df['duration_max'] = 0
        else:
            df['duration_mean'] = np.mean(self.durations)
            df['duration_max'] = np.max(self.durations)

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
