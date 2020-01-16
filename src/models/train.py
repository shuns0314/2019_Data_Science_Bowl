import argparse
import os
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import KFold

from loss_function import qwk, lgb_qwk
from post_process import post_processing, threshold
from closs_validation import stratified_group_k_fold


parser = argparse.ArgumentParser()
parser.add_argument('train_csv', type=str)
parser.add_argument('--test_csv', type=str, default=None)
parser.add_argument('--name', type=str)


MODEL_1_DROP_LIST = ["2050","4100","2060","4110","2070","2075","2080","2081","2083","3110","3120","3121","4220","4230","5000","4235","5010","4010","4020","4021","4022","4025","4030","4031","3010","4035","4040","3020","3021","4045","2000","4050","2010","2020","4070","2025","2030","4080","2035","2040","4090","4095","c51d8688","d51b1749","cf82af56","30614231","f93fc684","1b54d27f","a8a78786","37ee8496","51102b85","2a444e03","01ca3a3c","acf5c23f","86ba578b","a8cc6fec","160654fd","65a38bf7","c952eb01","06372577","d88e8f25","0413e89d","8d7e386c","58a0de5c","4bb2f698","5be391b5","ab4ec3a4","b2dba42b","2b9272f4","7f0836bf","5154fc30","3afb49e6","9ce586dd","16dffff1","2b058fe3","a7640a16","3bfd1a65","bfc77bd6","003cd2ee","fbaf3456","19967db1","df4fe8b6","5290eab1","3bb91dda","6f8106d9","5f0eb72c","5c2f29ca","7d5c30a2","90d848e0","3babcb9b","53c6e11a","d9c005dd","15eb4a7d","15a43e5b","f71c4741","895865f3","b5053438","7da34a02","a5e9da97","a44b10dc","17ca3959","ab3136ba","76babcde","1375ccb7","67439901","363d3849","bcceccc6","90efca10","a6d66e51","90ea0bac","7ab78247","25fa8af4","dcaede90","65abac75","15f99afc","47f43a44","8fee50e2","51311d7a","c6971acf","56bcd38d","c74f40cd","d3268efa","c189aaf2","7ec0c298","9e6b7fb5","5c3d2b2f","e4d32835","0d1da71f","abc5811c","e37a2b78","6d90d394","3d63345e","b012cd7f","792530f8","26a5a3dd","d2e9262e","598f4598","00c73085","7cf1bc53","46b50ba8","ecc36b7f","363c86c9","5859dfb6","8ac7cce4","ecc6157f","daac11b0","4a09ace1","47026d5f","9b4001e4","67aa2ada","f7e47413","cf7638f3","a2df0760","e64e2cfd","119b5b02","1bb5fbdb","1beb320a","7423acbc","14de4c5d","587b5989","828e68f9","3d0b9317","4b5efe37","fcfdffb6","a76029ee","1575e76c","bdf49a58","b2e5b0f1","bd701df8","250513af","6bf9e3e1","7dfe6d8a","9e34ea74","6cf7d25c","1cc7cfca","e79f3763","5b49460a","37c53127","ea296733","b74258a0","02a42007","9b23e8ee","bb3e370b","6f4bd64e","05ad839b","611485c5","a592d54e","c7f7f0e1","857f21c0","6077cc36","3dcdda7f","5e3ea25a","e7561dd2","f6947f54","9e4c8c7b","8d748b58","9de5e594","c54cf6c5","28520915","9b01374f","55115cbd","1f19558b","56cd3b43","795e4a37","83c6c409","3ee399c3","a8876db3","30df3273","77261ab5","4901243f","7040c096","1af8be29","f5b8c21a","499edb7c","6c930e6e","1cf54632","f56e0afc","6c517a88","d38c2fd7","2ec694de","562cec5f","832735e1","736f9581","b1d5101d","36fa3ebe","3dfd4aa4","1c178d24","2dc29e21","4c2ec19f","38074c54","ca11f653","f32856e4","89aace00","29bdd9ba","4ef8cdd3","d2278a3b","0330ab6a","46cd75b4","6f445b57","5e812b27","4e5fc6f5","df4940d3","33505eae","f806dc10","4a4c3d21","cb6010f8","022b4259","99ea62f3","461eace6","ac92046e","0a08139c","fd20ea40","ea321fb1","e3ff61fb","c7128948","31973d56","2c4e6db0","b120f2ac","731c0cbe","d3f1e122","9d4e7b25","cb1178ad","5de79a6a","0d18d96c","a8efe47b","cdd22e43","392e14df","93edfe2e","3ccd3f02","b80e5e84","262136f4","a5be6304","e57dd7af","d3640339","565a3990","beb0a7b9","884228c8","bc8f2793","73757a5e","709b1251","c7fe2a55","3bf1cf26","c58186bf","ad148f58","cfbd47c8","6f4adc4b","e694a35b","222660ff","0086365d","3d8c61b0","37db1c2f","f50fc6c1","8f094001","2a512369","7ad3efc6","e4f1efe6","8af75982","08fd73f3","a1e4395d","3323d7e9","47efca07","5f5b2617","6043a2b4","7d093bf9","907a054b","5e109ec3","cc5087a3","8d84fa81","4074bac2","4d911100","84b0e0c8","45d01abe","0db6d71d","29f54413","1325467d","99abe2bb","f3cd5473","91561152","dcb1663e","f28c589a","0ce40006","7525289a","17113b36","29a42aea","db02c830","ad2fc29c","d185d3ea","37937459","804ee27f","28a4eb9a","49ed92e9","56817e2b","86c924c4","e5c9df6f","b7dc8128","3edf6747","8b757ab8","93b353f2","a52b92d5","9c5ef70c","dcb55a27","d06f75b5","44cb4907","c2baf0bd","27253bdc","a1bbe385","2fb91ec1","155f62a4","ec138c1c","c1cac9a2","eb2c19cd","3ddc79c3","a0faea5d","070a5291","7fd1ac25","69fdac0a","28ed704e","c277e121","13f56524","e720d930","3393b68b","f54238ee","63f13dd7","e9c52111","763fc34e","26fd2d99","7961e599","9d29771f","84538528","77ead60d","3b2048ee","923afab1","3a4be871","9ed8f6da","74e5f8a7","a1192f43","e5734469","d122731b","04df9b66","5348fd84","92687c59","532a2afb","2dcad279","77c76bc5","71e712d8","a16a373e","3afde5dd","d2659ab4","88d4a5be","6088b756","85d1b0de","08ff79ad","15ba1109","7372e1a5","9ee1c98c","71fe8f75","6aeafed4","756e5507","d45ed6a1","1996c610","de26c3a6","1340b8d7","5d042115","e7e44842","2230fab4","9554a50b","4d6737eb","bd612267","3bb91ced","28f975ea","b7530680","b88f38da","5a848010","85de926c","c0415e5c","a29c5338","d88ca108","16667cc5","87d743c1","d02b7a8e","e04fb33d","bbfe0445","ecaab346","48349b14","5dc079d8","e080a381"]
MODEL_DROP_LIST = ["c51d8688","d51b1749","cf82af56","30614231","f93fc684","1b54d27f","a8a78786","37ee8496","51102b85","2a444e03","01ca3a3c","acf5c23f","86ba578b","a8cc6fec","160654fd","65a38bf7","c952eb01","06372577","d88e8f25","0413e89d","8d7e386c","58a0de5c","4bb2f698","5be391b5","ab4ec3a4","b2dba42b","2b9272f4","7f0836bf","5154fc30","3afb49e6","9ce586dd","16dffff1","2b058fe3","a7640a16","3bfd1a65","bfc77bd6","003cd2ee","fbaf3456","19967db1","df4fe8b6","5290eab1","3bb91dda","6f8106d9","5f0eb72c","5c2f29ca","7d5c30a2","90d848e0","3babcb9b","53c6e11a","d9c005dd","15eb4a7d","15a43e5b","f71c4741","895865f3","b5053438","7da34a02","a5e9da97","a44b10dc","17ca3959","ab3136ba","76babcde","1375ccb7","67439901","363d3849","bcceccc6","90efca10","a6d66e51","90ea0bac","7ab78247","25fa8af4","dcaede90","65abac75","15f99afc","47f43a44","8fee50e2","51311d7a","c6971acf","56bcd38d","c74f40cd","d3268efa","c189aaf2","7ec0c298","9e6b7fb5","5c3d2b2f","e4d32835","0d1da71f","abc5811c","e37a2b78","6d90d394","3d63345e","b012cd7f","792530f8","26a5a3dd","d2e9262e","598f4598","00c73085","7cf1bc53","46b50ba8","ecc36b7f","363c86c9","5859dfb6","8ac7cce4","ecc6157f","daac11b0","4a09ace1","47026d5f","9b4001e4","67aa2ada","f7e47413","cf7638f3","a2df0760","e64e2cfd","119b5b02","1bb5fbdb","1beb320a","7423acbc","14de4c5d","587b5989","828e68f9","3d0b9317","4b5efe37","fcfdffb6","a76029ee","1575e76c","bdf49a58","b2e5b0f1","bd701df8","250513af","6bf9e3e1","7dfe6d8a","9e34ea74","6cf7d25c","1cc7cfca","e79f3763","5b49460a","37c53127","ea296733","b74258a0","02a42007","9b23e8ee","bb3e370b","6f4bd64e","05ad839b","611485c5","a592d54e","c7f7f0e1","857f21c0","6077cc36","3dcdda7f","5e3ea25a","e7561dd2","f6947f54","9e4c8c7b","8d748b58","9de5e594","c54cf6c5","28520915","9b01374f","55115cbd","1f19558b","56cd3b43","795e4a37","83c6c409","3ee399c3","a8876db3","30df3273","77261ab5","4901243f","7040c096","1af8be29","f5b8c21a","499edb7c","6c930e6e","1cf54632","f56e0afc","6c517a88","d38c2fd7","2ec694de","562cec5f","832735e1","736f9581","b1d5101d","36fa3ebe","3dfd4aa4","1c178d24","2dc29e21","4c2ec19f","38074c54","ca11f653","f32856e4","89aace00","29bdd9ba","4ef8cdd3","d2278a3b","0330ab6a","46cd75b4","6f445b57","5e812b27","4e5fc6f5","df4940d3","33505eae","f806dc10","4a4c3d21","cb6010f8","022b4259","99ea62f3","461eace6","ac92046e","0a08139c","fd20ea40","ea321fb1","e3ff61fb","c7128948","31973d56","2c4e6db0","b120f2ac","731c0cbe","d3f1e122","9d4e7b25","cb1178ad","5de79a6a","0d18d96c","a8efe47b","cdd22e43","392e14df","93edfe2e","3ccd3f02","b80e5e84","262136f4","a5be6304","e57dd7af","d3640339","565a3990","beb0a7b9","884228c8","bc8f2793","73757a5e","709b1251","c7fe2a55","3bf1cf26","c58186bf","ad148f58","cfbd47c8","6f4adc4b","e694a35b","222660ff","0086365d","3d8c61b0","37db1c2f","f50fc6c1","8f094001","2a512369","7ad3efc6","e4f1efe6","8af75982","08fd73f3","a1e4395d","3323d7e9","47efca07","5f5b2617","6043a2b4","7d093bf9","907a054b","5e109ec3","cc5087a3","8d84fa81","4074bac2","4d911100","84b0e0c8","45d01abe","0db6d71d","29f54413","1325467d","99abe2bb","f3cd5473","91561152","dcb1663e","f28c589a","0ce40006","7525289a","17113b36","29a42aea","db02c830","ad2fc29c","d185d3ea","37937459","804ee27f","28a4eb9a","49ed92e9","56817e2b","86c924c4","e5c9df6f","b7dc8128","3edf6747","8b757ab8","93b353f2","a52b92d5","9c5ef70c","dcb55a27","d06f75b5","44cb4907","c2baf0bd","27253bdc","a1bbe385","2fb91ec1","155f62a4","ec138c1c","c1cac9a2","eb2c19cd","3ddc79c3","a0faea5d","070a5291","7fd1ac25","69fdac0a","28ed704e","c277e121","13f56524","e720d930","3393b68b","f54238ee","63f13dd7","e9c52111","763fc34e","26fd2d99","7961e599","9d29771f","84538528","77ead60d","3b2048ee","923afab1","3a4be871","9ed8f6da","74e5f8a7","a1192f43","e5734469","d122731b","04df9b66","5348fd84","92687c59","532a2afb","2dcad279","77c76bc5","71e712d8","a16a373e","3afde5dd","d2659ab4","88d4a5be","6088b756","85d1b0de","08ff79ad","15ba1109","7372e1a5","9ee1c98c","71fe8f75","6aeafed4","756e5507","d45ed6a1","1996c610","de26c3a6","1340b8d7","5d042115","e7e44842","2230fab4","9554a50b","4d6737eb","bd612267","3bb91ced","28f975ea","b7530680","b88f38da","5a848010","85de926c","c0415e5c","a29c5338","d88ca108","16667cc5","87d743c1","d02b7a8e","e04fb33d","bbfe0445","ecaab346","48349b14","5dc079d8","e080a381"]

# PARAMS = {
#     'threshold_0': 1.12232214,
#     'threshold_1': 1.73925866,
#     'threshold_2': 2.22506454,
#     }

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

    # folderの作成
    if args.name is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'lgb_{args.train_csv}_{now}'
    if not os.path.exists(f'models/{args.name}'):
        os.makedirs(f'models/{args.name}')

    coefficient = train_df['accuracy_group'].value_counts(sort=False)/len(train_df['accuracy_group'])
    test_df = pd.read_csv(f"data/processed/{args.test_csv}.csv", index_col=0)
    loss_1, total_test_pred_1, _, all_importance_1 = lgb_regression(train_df, test_df, seed=77)
    loss_2, total_test_pred_2, _, all_importance_2 = lgb_regression(train_df, test_df, seed=5)
    loss_3, total_test_pred_3, _, all_importance_3 = lgb_regression(train_df, test_df, seed=18)

    loss = np.mean([loss_1, loss_2, loss_3])
    
    print(f'loss_1: {loss_1}')
    print(f'loss_2: {loss_2}')
    print(f'loss_3: {loss_3}')
    print(f'total: {loss}')

    func = np.frompyfunc(threshold, 2, 1)


    total_test_pred = pd.concat([total_test_pred_1, total_test_pred_2, total_test_pred_3], axis=1)
    total_test_pred = total_test_pred.mean(axis=1)
    pred_df = func(total_test_pred, PARAMS)

    all_importance = pd.concat([all_importance_1, all_importance_2, all_importance_3], axis=1)
    all_importance.to_csv(f'models/{args.name}/all_importance.csv')
    pred_df.to_csv(f'models/{args.name}/check_cv.csv')
    # pred_df = pred_df.apply(lambda x: x.mode()[0] if len(x.mode()) == 1 else coefficient[x.mode()].idxmax(), axis=1)
    pred_df.to_csv(f'models/{args.name}/submission.csv', header=False)

    with open(f'models/{args.name}/loss.txt', mode='w') as f:
        f.write(f"val_loss: {loss_1}\n")
        f.write(f"val_loss: {loss_2}\n")
        f.write(f"val_loss: {loss_3}\n")
        f.write(f"val_loss: {loss}\n")
        f.write(f"train_csv: {args.name}\n")

    # modelのsave
    # joblib.dump(model, f'models/{args.name}/model_{args.name}.pkl')

    # paramsのsave
    # with open(f'models/{args.name}/params_{args.name}.pkl', 'wb') as handle:
    #     pickle.dump(params, handle)


def lgb_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> pd.DataFrame:

    num_fold = 8
    columns_for_target_encoding = [
        'frequency_label',
        'count_action_label',
        'count_accuracy_label',
        'mean_accuracy_group_label',
        'label_description_val',
        'label_session_title_count_accuracy_label',
        'label_mean_accuracy_group_label_description_val',
        'label_frequency_label_description_val'
        ]

    # train_df = train_df.drop(MODEL_DROP_LIST, axis=1)
    # test_df = test_df.drop(MODEL_DROP_LIST, axis=1)

    train_df = train_df.drop(train_df.loc[:, train_df.columns.str.contains('args')].columns, axis=1)
    train_df = train_df.drop(train_df.loc[:, train_df.columns.str.contains('info')].columns, axis=1)
    test_df = test_df.drop(test_df.loc[:, test_df.columns.str.contains('args')].columns, axis=1)
    test_df = test_df.drop(test_df.loc[:, test_df.columns.str.contains('info')].columns, axis=1)


    y = train_df['accuracy_group']
    x = train_df.drop('accuracy_group', axis=1)

    groups = np.array(x['installation_id'])

    lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            }

    x = x.drop('installation_id', axis=1)
    # total_pred = np.zeros(y.shape)
    total_val_pred = np.zeros(y.shape)

    test_x = test_df.drop('accuracy_group', axis=1)
    test_x = test_x.drop('installation_id', axis=1)
    total_test_pred = np.zeros([test_df.shape[0], num_fold])

    all_importance = []
    test_index_all = []

    for fold_ind, (train_ind, test_ind) in enumerate(
            stratified_group_k_fold(X=x, y=y, groups=groups, k=num_fold, seed=seed)):
        # print(dev_ind)
        x_train = x.iloc[train_ind]
        y_train = y.iloc[train_ind]
        x_test = x.iloc[test_ind]
        y_test = y.iloc[test_ind]

        for column in columns_for_target_encoding:
            data_tmp = pd.DataFrame({column: x_train[column], 'target': y_train})
            target_mean = data_tmp.groupby(column)['target'].mean()
            x_test[f'target_{column}'] = x_test[column].map(target_mean)
            test_x[f'target_{column}'] = test_x[column].map(target_mean)

            tmp = np.repeat(np.nan, x_train.shape[0])

            kf = KFold(n_splits=4, shuffle=True, random_state=seed)
            for idx_1, idx_2 in kf.split(x_train):
                target_mean = data_tmp.iloc[idx_1].groupby(column)['target'].mean()
                tmp[idx_2] = x_train[column].iloc[idx_2].map(target_mean)

            x_train[f'target_{column}'] = tmp

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)

        model = lgb.train(params=lgb_params,
                          train_set=lgb_train,
                          valid_sets=lgb_val,
                          feval=lgb_qwk,
                          num_boost_round=1000,
                          early_stopping_rounds=50)

        # y_val_pred = model.predict(x_test, num_iteration=model.best_iteration)
        # params = post_processing(y_val, y_val_pred)
        # all_params.append(params)

        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        all_importance.append(pd.DataFrame(model.feature_importance('gain'), index=x_train.columns))

        total_val_pred[test_ind] = y_pred
        test_index_all += test_ind
        # total_pred[test_ind] = y_pred

        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        total_test_pred[:, fold_ind] = test_pred

    all_importance = pd.concat(all_importance, axis=1)
    

    # print(total_val_pred)
    func = np.frompyfunc(threshold, 2, 1)
    total_val_pred[test_index_all] = func(total_val_pred[test_index_all], PARAMS)
    loss = qwk(total_val_pred[test_index_all], y[test_index_all].values)
    print(loss)
    return loss, pd.DataFrame(total_test_pred), model, all_importance


if __name__ == "__main__":
    main()
