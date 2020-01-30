import argparse
import pickle
import glob

import pandas as pd
import numpy as np
import joblib

from post_process import threshold


parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('test_csv', type=str)

PARAMS = {
    'threshold_0': 1.12,
    'threshold_1': 1.62,
    'threshold_2': 2.20
    }


def main():
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv, index_col=0)
    test_df = test_df.drop('accuracy_group', axis=1)
    test_df = test_df.drop('installation_id', axis=1)

    post_pred = predict_main(test_df, args.model_path)

    submission = pd.read_csv('data/raw/sample_submission.csv')
    submission['accuracy_group'] = post_pred.astype(int)
    print(submission.dtypes)
    submission.to_csv(f'submission.csv', index=False)


def predict_main(test_df: pd.DataFrame, path) -> pd.DataFrame:
    model_name_list = glob.glob(f'{path}/*')

    pred_list = []
    for i, model_name in enumerate(model_name_list):
        with open(model_name, mode='rb') as fp:
            restored_model = pickle.load(fp)

        # 復元したモデルを使って Hold-out したデータを推論する
        y_pred = restored_model.predict(
            test_df, restored_model.best_iteration)
        print(f'{i}/{len(model_name_list)}')
        pred_list.append(y_pred.reshape(y_pred.shape[0], 1))
        # print(y_pred.reshape(y_pred.shape[0], 1))

    pred = np.hstack(pred_list)
    print(pred)
    pred = np.mean(pred, axis=1)

    func = np.frompyfunc(threshold, 2, 1)
    pred = func(pred, PARAMS)
    return pred


if __name__ == "__main__":
    main()
