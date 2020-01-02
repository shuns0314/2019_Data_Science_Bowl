
import pandas as pd
import numpy as np


class GetEventData():
    """各installation_idのおける過去のゲームの実績をまとめるmethod."""

    def __init__(self, win_code, test_set=False):
        self.win_code = win_code
        self.test_set = test_set

        # event_data
        self.version = 0
        self.castles_placed = 0
        self.molds = 0
        self.sand = 0
        self.filled = 0
        self.movie_id = 0
        self.options = 0
        self.animals = 0
        self.round_target_size = 0
        self.round_target_type = 0
        self.round_target_animal = 0
        self.item_type = 0
        self.position = 0
        self.animal = 0
        self.correct = 0
        self.misses = 0
        self.holding_shell = 0
        self.has_water = 0
        self.shells = 0
        self.holes = 0
        self.shell_size = 0
        self.hole_position = 0
        self.cloud = 0
        self.cloud_size = 0
        self.water_level = 0
        self.time_played = 0
        # elf.houses = 0
        # self.dinosaurs = 0
        # self.dinosaur = 0
        self.dinosaurs_placed = 0
        self.house_size = 0
        self.house_position = 0


    def process(self, user_sample, installation_id):
        all_assessments = []

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

            features = dict()

            # session typeがAssessmentのやつだけ、カウントする。
            if (session_type == 'Assessment') & (second_condition):

                if features is not None:
                    features['installation_id'] = installation_id

                    features['version'] = self.version
                    features['castles_placed'] = self.castles_placed
                    features['molds'] = self.molds
                    features['sand'] = self.sand
                    features['filled'] = self.filled
                    features['movie_id'] = self.movie_id
                    features['options'] = self.options
                    features['animals'] = self.animals
                    features['round_target.size'] = self.round_target_size
                    features['round_target.type'] = self.round_target_type
                    features['round_target.animal'] = self.round_target_animal
                    features['item_type'] = self.item_type
                    features['position'] = self.position
                    features['animal'] = self.animal
                    features['correct'] = self.correct
                    features['misses'] = self.misses
                    features['holding_shell'] = self.holding_shell
                    features['has_water'] = self.has_water
                    features['shells'] = self.shells
                    features['holes'] = self.holes
                    features['shell_size'] = self.shell_size
                    features['hole_position'] = self.hole_position
                    features['cloud'] = self.cloud
                    features['cloud_size'] = self.cloud_size
                    features['water_level'] = self.water_level
                    features['time_played'] = self.time_played
                    # features['houses'] = self.houses
                    # features['dinosaurs'] = self.dinosaurs
                    # features['dinosaur'] = self.dinosaur
                    features['dinosaurs_placed'] = self.dinosaurs_placed
                    features['house.size'] = self.house_size
                    features['house.position'] = self.house_position

                    all_assessments.append(features)

            self.castles_placed += session['castles_placed'].sum()
            self.molds += session['molds'].sum()
            self.sand += session['sand'].sum()
            self.filled += session['filled'].sum()
            self.movie_id += session['movie_id'].sum()
            self.options += session['options'].sum()
            self.animals += session['animals'].sum()
            self.round_target_size += session['round_target.size'].sum()
            self.round_target_type += session['round_target.type'].sum()
            self.round_target_animal += session['round_target.animal'].sum()
            self.item_type += session['item_type'].sum()
            self.position += session['position'].sum()
            self.animal += session['animal'].sum()
            self.correct += session['correct'].sum()
            self.misses += session['misses'].sum()
            self.holding_shell += session['holding_shell'].sum()
            self.has_water += session['has_water'].sum()
            self.shells += session['shells'].sum()
            self.holes += session['holes'].sum()
            self.shell_size += session['shell_size'].sum()
            self.hole_position += session['hole_position'].sum()
            self.cloud += session['cloud'].sum()
            self.cloud_size += session['cloud_size'].sum()
            self.water_level += session['water_level'].sum()
            self.time_played += session['time_played'].sum()
            # self.houses += session['houses'].sum()
            # self.dinosaurs += session['dinosaurs'].sum()
            # self.dinosaur += session['dinosaur']
            self.dinosaurs_placed += session['dinosaurs_placed'].sum()
            self.house_size += session['house.size'].sum()
            self.house_position += session['house.position'].sum()

        if self.test_set:
            return all_assessments[-1]
        return all_assessments


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


    def process(self, user_sample, installation_id):
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
                    features['installation_id'] = installation_id
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
