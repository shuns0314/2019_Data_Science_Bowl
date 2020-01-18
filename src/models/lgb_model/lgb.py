import pandas as pd
import lightgbm as lgb

from models.loss_function import lgb_qwk
from models.target_encoding import TargetEncoding


FEATURES_FOR_TARGET_ENCODING = [
    # 'frequency_label',
    # 'count_action_label',
    # 'count_accuracy_label',
    'mean_accuracy_group_label',
    # 'label_description_val',
    # 'label_session_title_count_accuracy_label',
    # 'label_mean_accuracy_group_label_description_val',
    # 'label_frequency_label_description_val'
    ]

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    }


def lgb_model(x, y, train_ind, test_ind, test_x, seed, val_x=None):
    x_train = x.iloc[train_ind]
    y_train = y.iloc[train_ind]
    x_test = x.iloc[test_ind]
    y_test = y.iloc[test_ind]

    target_encoding = TargetEncoding()
    if val_x is not None:
        x_train, x_test, test_x, val_x = target_encoding.process(
            x_train, y_train, x_test, test_x,
            columns=FEATURES_FOR_TARGET_ENCODING,
            seed=seed, val_x=val_x)
    else:
        x_train, x_test, test_x = target_encoding.process(
            x_train, y_train, x_test, test_x,
            columns=FEATURES_FOR_TARGET_ENCODING,
            seed=seed)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)

    model = lgb.train(params=LGB_PARAMS,
                      train_set=lgb_train,
                      valid_sets=lgb_val,
                      feval=lgb_qwk,
                      num_boost_round=1000,
                      early_stopping_rounds=50)

    y_pred = model.predict(x_test, num_iteration=model.best_iteration)

    importance = pd.DataFrame(model.feature_importance('gain'),
                              index=x_train.columns)
    return model, y_pred, importance, test_x, val_x
