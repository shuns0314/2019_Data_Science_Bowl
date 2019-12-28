
import pandas as pd
import numpy as np


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
                print(features)
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
