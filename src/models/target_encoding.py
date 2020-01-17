import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class TargetEncoding:

    def process(self, x_train, y_train, x_val,
                test_x, columns: list, seed: int = 77):
        """
        columns: for target_encoding features
        """
        for column in columns:
            data_tmp = pd.DataFrame(
                {column: x_train[column], 'target': y_train}
                )
            target_mean = data_tmp.groupby(column)['target'].mean()
            x_val[f'target_{column}'] = x_val[column].map(target_mean)
            test_x[f'target_{column}'] = test_x[column].map(target_mean)
            tmp = np.repeat(np.nan, x_train.shape[0])
            kf = KFold(n_splits=4, shuffle=True, random_state=seed)
            for idx_1, idx_2 in kf.split(x_train):
                target_mean = data_tmp.iloc[idx_1].groupby(column)['target'].mean()
                tmp[idx_2] = x_train[column].iloc[idx_2].map(target_mean)
            x_train[f'target_{column}'] = tmp

        return x_train, x_val, test_x
