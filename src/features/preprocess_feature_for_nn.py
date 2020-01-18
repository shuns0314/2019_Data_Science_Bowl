import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


EFFECTIVE_FEATURES = [
    'session_title', 'label_session_title_count_accuracy_label',
    'count_accuracy', 'label_session_title_count_action_label', '2000',
    'mean_accuracy_group', '3afb49e6', 'acc_Bird Measurer (Assessment)',
    'Clip', 'mean_game_round',
    'label_mean_accuracy_group_label_description_val', 'good_comment_ratio',
    '4070', 'acc_Chest Sorter (Assessment)', '7372e1a5',
    'count_label_session_title_description_val', '04df9b66',
    'label_session_title_description_val', '3020', 'args_6'
    ]

NUMERICAL_FEATURES = [
    'args_6', 'count_accuracy', '2000', 'mean_accuracy_group', '3afb49e6',
    'acc_Bird Measurer (Assessment)', 'Clip', 'mean_game_round',
    'good_comment_ratio', '4070', 'acc_Chest Sorter (Assessment)', '7372e1a5',
    'count_label_session_title_description_val', '04df9b66', '3020'
    ]

CATEGORICAL_FEATURES = [
    'session_title', 'label_session_title_count_accuracy_label',
    'label_session_title_count_action_label',
    'label_mean_accuracy_group_label_description_val',
    'label_session_title_description_val'
    ]


class PreprocessForNN:

    def process(self, train: pd.DataFrame, test: pd.DataFrame):

        train = train[EFFECTIVE_FEATURES]
        test = test[EFFECTIVE_FEATURES]

        # 数値データをBox-Cox変換
        pt = PowerTransformer()
        pt.fit(train[NUMERICAL_FEATURES])
        train[NUMERICAL_FEATURES] = pt.transform(train[NUMERICAL_FEATURES])
        test[NUMERICAL_FEATURES] = pt.transform(test[NUMERICAL_FEATURES])

        # Category data を One hot encoding
        all_df = pd.concat([train, test])
        all_df = pd.get_dummies(all_df, columns=CATEGORICAL_FEATURES)
        train = all_df.iloc[:train.shape[0], :].reset_index(drop=True)
        test = all_df.iloc[:test.shape[0], :].reset_index(drop=True)

        # 欠損値埋め
        train = train.fillna(0)
        test = test.fillna(0)

        return train, test
