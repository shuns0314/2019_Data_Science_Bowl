import argparse
import pickle

import pandas as pd
import numpy as np
import joblib

from post_process import threshold


parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('test_csv', type=str)


def main():
    args = parser.parse_args()
    model = joblib.load(f'models/{args.model}/model_{args.model}.pkl')
    with open(f'models/{args.model}/params_{args.model}.pkl', 'rb') as handle:
        params = pickle.load(handle)

    test_df = pd.read_csv(f'data/processed/{args.test_csv}.csv', index_col=0)
    test_df = test_df.drop('accuracy_group', axis=1)
    pred_df = model.predict(test_df)
    func = np.frompyfunc(threshold, 2, 1)
    post_pred = func(pred_df, params)

    submission = pd.read_csv('data/raw/sample_submission.csv')
    submission['accuracy_group'] = post_pred.astype(int)
    print(submission.dtypes)
    submission.to_csv(f'models/{args.model}/submission.csv', index=False)


if __name__ == "__main__":
    main()
