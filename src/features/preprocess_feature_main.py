import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


EVENT_ID_LIST = ["c51d8688", "d51b1749", "cf82af56", "30614231", "f93fc684", "1b54d27f", "a8a78786", "37ee8496", "51102b85", "2a444e03", "01ca3a3c", "acf5c23f", "86ba578b", "a8cc6fec", "160654fd", "65a38bf7", "c952eb01", "06372577", "d88e8f25", "0413e89d", "8d7e386c", "58a0de5c", "4bb2f698", "5be391b5", "ab4ec3a4", "b2dba42b", "2b9272f4", "7f0836bf", "5154fc30", "3afb49e6", "9ce586dd", "16dffff1", "2b058fe3", "a7640a16", "3bfd1a65", "bfc77bd6","003cd2ee","fbaf3456","19967db1","df4fe8b6","5290eab1","3bb91dda","6f8106d9","5f0eb72c","5c2f29ca","7d5c30a2","90d848e0","3babcb9b","53c6e11a","d9c005dd","15eb4a7d","15a43e5b","f71c4741","895865f3","b5053438","7da34a02","a5e9da97","a44b10dc","17ca3959","ab3136ba","76babcde","1375ccb7","67439901","363d3849","bcceccc6","90efca10","a6d66e51","90ea0bac","7ab78247","25fa8af4","dcaede90","65abac75","15f99afc","47f43a44","8fee50e2","51311d7a","c6971acf","56bcd38d","c74f40cd","d3268efa","c189aaf2","7ec0c298","9e6b7fb5","5c3d2b2f","e4d32835","0d1da71f","abc5811c","e37a2b78","6d90d394","3d63345e","b012cd7f","792530f8","26a5a3dd","d2e9262e","598f4598","00c73085","7cf1bc53","46b50ba8","ecc36b7f","363c86c9","5859dfb6","8ac7cce4","ecc6157f","daac11b0","4a09ace1","47026d5f","9b4001e4","67aa2ada","f7e47413","cf7638f3","a2df0760","e64e2cfd","119b5b02","1bb5fbdb","1beb320a","7423acbc","14de4c5d","587b5989","828e68f9","3d0b9317","4b5efe37","fcfdffb6","a76029ee","1575e76c","bdf49a58","b2e5b0f1","bd701df8","250513af","6bf9e3e1","7dfe6d8a","9e34ea74","6cf7d25c","1cc7cfca","e79f3763","5b49460a","37c53127","ea296733","b74258a0","02a42007","9b23e8ee","bb3e370b","6f4bd64e","05ad839b","611485c5","a592d54e","c7f7f0e1","857f21c0","6077cc36","3dcdda7f","5e3ea25a","e7561dd2","f6947f54","9e4c8c7b","8d748b58","9de5e594","c54cf6c5","28520915","9b01374f","55115cbd","1f19558b","56cd3b43","795e4a37","83c6c409","3ee399c3","a8876db3","30df3273","77261ab5","4901243f","7040c096","1af8be29","f5b8c21a","499edb7c","6c930e6e","1cf54632","f56e0afc","6c517a88","d38c2fd7","2ec694de","562cec5f","832735e1","736f9581","b1d5101d","36fa3ebe","3dfd4aa4","1c178d24","2dc29e21","4c2ec19f","38074c54","ca11f653","f32856e4","89aace00","29bdd9ba","4ef8cdd3","d2278a3b","0330ab6a","46cd75b4","6f445b57","5e812b27","4e5fc6f5","df4940d3","33505eae","f806dc10","4a4c3d21","cb6010f8","022b4259","99ea62f3","461eace6","ac92046e","0a08139c","fd20ea40","ea321fb1","e3ff61fb","c7128948","31973d56","2c4e6db0","b120f2ac","731c0cbe","d3f1e122","9d4e7b25","cb1178ad","5de79a6a","0d18d96c","a8efe47b","cdd22e43","392e14df","93edfe2e","3ccd3f02","b80e5e84","262136f4","a5be6304","e57dd7af","d3640339","565a3990","beb0a7b9","884228c8","bc8f2793","73757a5e","709b1251","c7fe2a55","3bf1cf26","c58186bf","ad148f58","cfbd47c8","6f4adc4b","e694a35b","222660ff","0086365d","3d8c61b0","37db1c2f","f50fc6c1","8f094001","2a512369","7ad3efc6","e4f1efe6","8af75982","08fd73f3","a1e4395d","3323d7e9","47efca07","5f5b2617","6043a2b4","7d093bf9","907a054b","5e109ec3","cc5087a3","8d84fa81","4074bac2","4d911100","84b0e0c8","45d01abe","0db6d71d","29f54413","1325467d","99abe2bb","f3cd5473","91561152","dcb1663e","f28c589a","0ce40006","7525289a","17113b36","29a42aea","db02c830","ad2fc29c","d185d3ea","37937459","804ee27f","28a4eb9a","49ed92e9","56817e2b","86c924c4","e5c9df6f","b7dc8128","3edf6747","8b757ab8","93b353f2","a52b92d5","9c5ef70c","dcb55a27","d06f75b5","44cb4907","c2baf0bd","27253bdc","a1bbe385","2fb91ec1","155f62a4","ec138c1c","c1cac9a2","eb2c19cd","3ddc79c3","a0faea5d","070a5291","7fd1ac25","69fdac0a","28ed704e","c277e121","13f56524","e720d930","3393b68b","f54238ee","63f13dd7","e9c52111","763fc34e","26fd2d99","7961e599","9d29771f","84538528","77ead60d","3b2048ee","923afab1","3a4be871","9ed8f6da","74e5f8a7","a1192f43","e5734469","d122731b","04df9b66","5348fd84","92687c59","532a2afb","2dcad279","77c76bc5","71e712d8","a16a373e","3afde5dd","d2659ab4","88d4a5be","6088b756","85d1b0de","08ff79ad","15ba1109","7372e1a5","9ee1c98c","71fe8f75","6aeafed4","756e5507","d45ed6a1","1996c610","de26c3a6","1340b8d7","5d042115","e7e44842","2230fab4","9554a50b","4d6737eb","bd612267","3bb91ced","28f975ea","b7530680","b88f38da","5a848010","85de926c","c0415e5c","a29c5338","d88ca108","16667cc5","87d743c1","d02b7a8e","e04fb33d","bbfe0445","ecaab346","48349b14","5dc079d8","e080a381"]
EVENT_CODE_LIST  = ['2050', '4100', '4230', '5000', '4235', '2060', '4110', '5010', '2070', '2075', '2080', '2081', '2083', '3110', '4010', '3120', '3121', '4020', '4021', 
                    '4022', '4025', '4030', '4031', '3010', '4035', '4040', '3020', '3021', '4045', '2000', '2010', '2020', '4070', '2025', '2030', '4080', '2035', 
                    '2040', '4090', '4220', '4095', '4050']
TITLE_LIST = ["All Star Sorting","Welcome to Lost Lagoon!","Scrub-A-Dub","Costume Box","Magma Peak - Level 1","Bubble Bath","Crystal Caves - Level 2","Tree Top City - Level 3","Slop Problem","Happy Camel","Dino Drink","Watering Hole (Activity)","Heavy Heavier Heaviest","Honey Cake","Crystal Caves - Level 1","Air Show","Chicken Balancer (Activity)","Flower Waterer (Activity)","Cart Balancer (Assessment)","Mushroom Sorter (Assessment)","Leaf Leader","Dino Dive","Pan Balance","Cauldron Filler (Assessment)","Treasure Map","Crystals Rule","Lifting Heavy Things","Rulers","Fireworks (Activity)","Pirate's Tale","12 Monkeys","Balancing Act","Bottle Filler (Activity)","Chest Sorter (Assessment)","Sandcastle Builder (Activity)","Crystal Caves - Level 3","Egg Dropper (Activity)","Magma Peak - Level 2","Tree Top City - Level 2","Ordering Spheres","Chow Time","Bug Measurer (Activity)","Bird Measurer (Assessment)","Tree Top City - Level 1"]
ACC_ASSESSMENT = ["acc_Mushroom Sorter (Assessment)", "acc_Chest Sorter (Assessment)", "acc_Bird Measurer (Assessment)", "acc_Cart Balancer (Assessment)", "acc_Cauldron Filler (Assessment)"]
ACC_NO = ["acc_0", "acc_1", "acc_2", "acc_3"]
DROP_FEATURES = ['mean_accuracy_group_label', 'count_correct_attempts', 'total_duration', 'count_uncorrect_attempts', 'frequency_label', 'count_accuracy_label']
parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)
parser.add_argument('test_csv', type=str)


def main():
    args = parser.parse_args()
    train_df = pd.read_csv(f"{args.train_csv}", index_col=0)
    test_df = pd.read_csv(f"{args.test_csv}", index_col=0)
    train_df.columns = train_df.columns.str.replace(',', '')
    test_df.columns = test_df.columns.str.replace(',', '')

    preprocess = PreprocessFeatures()
    train = preprocess.process(train_df)
    test = preprocess.process(test_df)

    time_preprocess = PreprocessTime()
    train = time_preprocess.process(train)
    test = time_preprocess.process(test)

    mutual_preprocess = MutualPreprocessFeatures()
    train, test = mutual_preprocess.process(train, test)

    train.drop(DROP_FEATURES, axis=1, inplace=True)
    test.drop(DROP_FEATURES, axis=1, inplace=True)

    train.to_csv('data/transformation/trans_train.csv')
    test.to_csv('data/transformation/trans_test.csv')
    return train, test


class MutualPreprocessFeatures:
    
    def process(self, train: pd.DataFrame, test: pd.DataFrame):
        # train, test, label = self.list_label_encoder(train, test, ['description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['frequency_label', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['mean_accuracy_group_label', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['count_action_label', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        # train, test, label = self.list_label_encoder(train, test, ['session_title', 'description_val'])
        # train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'count_action_label'])
        # train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'count_accuracy_label'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'mean_accuracy_group_label'])
        train, test = self.frequency_encoder(train, test, label)
        
        train, test, label = self.list_label_encoder(train, test, ['EVENT_CODE_LIST_lank4', 'ACC_ASSESSMENT_lank0'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['ACC_NO_lank0', 'hour_lank2'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['session_title', 'hour_lank1'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['dayofweek_lank0', 'non_zero_info'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['count_accuracy_label', 'info_lank0'])
        train, test = self.frequency_encoder(train, test, label)
        train, test, label = self.list_label_encoder(train, test, ['frequency_label', 'EVENT_CODE_LIST_lank4'])
        train, test = self.frequency_encoder(train, test, label)
        return train, test

    def list_label_encoder(self, train, test, columns: list):
        le = LabelEncoder()
        train_description_val_session_title = train[columns].apply(lambda x: str([i for i in x]), axis=1)
        test_description_val_session_title = test[columns].apply(lambda x: str([i for i in x]), axis=1)
        le.fit(train_description_val_session_title)
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        label_name = '_'.join(columns)
        label_name = f'label_{label_name}'
        train[label_name] = train_description_val_session_title.apply(lambda x: le_dict.get(x, -1))
        test[label_name] = test_description_val_session_title.apply(lambda x: le_dict.get(x, -1))
        return train, test, label_name

    def frequency_encoder(self, train, test, label_name):
        freq = train[label_name].value_counts()
        train[f'count_{label_name}'] = train[label_name].map(freq)
        test[f'count_{label_name}'] = test[label_name].map(freq)
        return train, test


class PreprocessFeatures:

    def process(self, df):
        df['frequency_label'] = df.frequency.apply(lambda x: self.frequency_label_encoder(np.log1p(x)))
        df.drop('frequency', axis=1, inplace=True)
        df['count_action_label'] = df.count_actions.apply(lambda x: self.bins_count_action_label_encoder(np.log1p(x)))
        df.drop('count_actions', axis=1, inplace=True)
        df['count_accuracy_label'] = df.count_accuracy.apply(lambda x: self.count_accuracy_label_encoder(x))
        df.drop('count_accuracy', axis=1, inplace=True)
        df['mean_accuracy_group_label'] = df.mean_accuracy_group.apply(lambda x: self.mean_accuracy_group_label_encoder(x))
        df.drop('mean_accuracy_group', axis=1, inplace=True)
        
        sum_type = df[['Game', 'Clip', 'Activity']].sum(axis=1)
        df['ratio_Game'] = df['Game']/sum_type
        df['ratio_Clip'] = df['Clip']/sum_type
        df['ratio_Activity'] = df['Activity']/sum_type
        # df['ratio_Assessment'] = df['Assessment']/sum_type
        df.drop(['Assessment', 'Game', 'Clip', 'Activity'], axis=1, inplace=True)

        """game_sessionごとではなく、instration_idごとのcountとか"""
        # df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        # df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        # df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        # df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        # df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
        #                                 4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
        #                                 2040, 4090, 4220, 4095]].sum(axis=1)
        df['sum_event_code_2000'] = df[['2050', '2060', '2070', '2075', '2080', '2081', '2083',
                                        '2000', '2010', '2020', '2025', '2030', '2035', '2040']].sum(axis=1)
        df['sum_event_code_3000'] = df[['3110', '3120', '3121',
                                        '3010', '3020', '3021']].sum(axis=1)
        df['sum_event_code_4000'] = df[['4100', '4230', '4235', '4110', '4010', '4020', '4021',
                                        '4022', '4025', '4030', '4031', '4035', '4040', '4045',
                                        '4070', '4080', '4090', '4220', '4095', "4050"]].sum(axis=1)

        # df['event_id_non_zero_count'] = df.loc[:, EVENT_ID_LIST].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)
        # df['event_code_non_zero_count'] = df.loc[:, EVENT_CODE_LIST].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)
        df['title_non_zero_count'] = df.loc[:, TITLE_LIST].apply(lambda x: len(x.to_numpy().nonzero()[0]), axis=1)
        # df['good_comment_ratio'] = df['good_comment'] / df['description']

        sum_event_code = df[['sum_event_code_2000', 'sum_event_code_3000', 'sum_event_code_4000']].sum(axis=1)
        # df['ratio_event_code_2000'] = df['sum_event_code_2000']/sum_event_code
        df['ratio_event_code_3000'] = df['sum_event_code_3000']/sum_event_code
        df['ratio_event_code_4000'] = df['sum_event_code_4000']/sum_event_code
        df.drop(['sum_event_code_2000','ratio_event_code_3000', 'ratio_event_code_4000'], axis=1, inplace=True)
        
        # df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        # df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        # df.drop('sum_event_code_count', axis=1, inplace=True)
        
        return df

    def count_accuracy_label_encoder(self, x):
        if x == 0:
            y = 0
        elif x > 0 and x <= 0.34:
            y = 1
        elif x > 0.34 and x <= 0.67:
            y = 2
        elif x > 0.67 and x < 1:
            y = 3
        elif x == 1:
            y = 4
        return y

    def mean_accuracy_group_label_encoder(self, x):
        if x == 0:
            y = 0
        elif x > 0 and x < 1.0:
            y = 1
        elif x >= 1.0 and x < 2.0:
            y = 2
        elif x >= 2.0 and x < 3.0:
            y = 3
        elif x == 3.0:
            y = 4
        return y

    def bins_count_action_label_encoder(self, x):
        if x < 1.0:
            y = 0
        elif x >= 1.0 and x < 2.0:
            y = 1
        elif x >= 2.0 and x < 3.0:
            y = 2
        elif x >= 3.0 and x < 4.0:
            y = 3
        elif x >= 4.0 and x < 5.0:
            y = 4
        elif x >= 5.0 and x < 6.0:
            y = 5
        elif x >= 6.0 and x < 7.0:
            y = 6
        elif x >= 7.0 and x < 8.0:
            y = 7
        elif x >= 8.0 and x < 9.0:
            y = 8
        else:
            y = 9
        return y

    def frequency_label_encoder(self, x):
        if x <= 0.1:
            y = 0
        elif x > 0.1 and x <= 0.2:
            y = 1
        elif x > 0.2 and x <= 0.3:
            y = 2
        elif x > 0.3 and x <= 0.4:
            y = 3
        elif x > 0.4 and x <= 0.5:
            y = 4
        elif x > 0.5 and x <= 1.0:
            y=5
        else:
            y=6
        return y


class PreprocessTime:

    def process(self, df):
        lank = 5
        new_columns = [f'event_id_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=EVENT_ID_LIST)
        df.drop(EVENT_ID_LIST, axis=1, inplace=True)

        new_columns = [f'EVENT_CODE_LIST_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=[str(i) for i in EVENT_CODE_LIST])
        df.drop([str(i) for i in EVENT_CODE_LIST], axis=1, inplace=True)

        new_columns = [f'TITLE_LIST_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=TITLE_LIST)
        df.drop(TITLE_LIST, axis=1, inplace=True)

        new_columns = [f'ACC_ASSESSMENT_lank{i}' for i in range(lank)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=ACC_ASSESSMENT)
        df.drop(ACC_ASSESSMENT, axis=1, inplace=True)

        new_columns = [f'ACC_NO_lank{i}' for i in range(4)]
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=ACC_NO)
        df.drop(ACC_NO, axis=1, inplace=True)

        columns = df.columns[df.columns.str.contains('dayofweek')]
        new_columns = [f'dayofweek_lank{i}' for i in range(1)]
        df[new_columns] = self.best_columns(df=df, lank=1, columns=columns)
        df.drop(columns, axis=1, inplace=True)
    
        # df = self.common_process(df, 'date', lank=3)
        df = self.common_process(df, 'hour', lank=3)
        # df = self.common_process(df, 'month', lank=1)
        # df = self.common_process(df, 'dayofweek', lank=1)
        df = self.common_process(df, 'args', lank=3)
        df = self.common_process(df, 'info', lank=3)
        
        return df

    def common_process(self, df, word_in_colums: str, lank: int):
        columns = df.columns[df.columns.str.contains(word_in_colums)]
        # df[f'mean_{word_in_colums}_count'] = df[columns].mean(axis=1)
        df[f'std_{word_in_colums}_count'] = df[columns].std(axis=1)
        df[f'non_zero_{word_in_colums}'] = (df[columns] == 0).sum(axis=1)
        new_columns = [f'{word_in_colums}_lank{i}' for i in range(lank)]
        print(new_columns)
        df[new_columns] = self.best_columns(df=df, lank=lank, columns=columns)
        df.drop(columns, axis=1, inplace=True)
        return df

    def best_columns(self, df, lank, columns):
        df = df[columns]
        rename_columns = {date: i for i, date in enumerate(columns)}
        df = df.rename(columns=rename_columns)
        best_df = df.rank(axis=1, ascending=False).apply(
            lambda x: pd.Series(list(x.sort_values()[0:lank].index)), axis=1
            )
        return best_df


if __name__ == "__main__":
    main()
