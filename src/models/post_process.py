import numpy as np
import optuna

from loss_function import qwk


def post_processing(y_test, y_pred):

    def objectives(trial):
        params = {
            'threshold_0': trial.suggest_uniform('threshold_0', 0.0, 1.5),
            'threshold_1': trial.suggest_uniform('threshold_1', 1.2, 2.0),
            'threshold_2': trial.suggest_uniform('threshold_2', 2.0, 3.0),
        }
        func = np.frompyfunc(threshold, 2, 1)
        post_pred = func(y_pred, params)
        loss = qwk(y_test, post_pred)

        return loss

    study = optuna.create_study(direction='maximize')
    study.optimize(objectives, n_trials=150)

    print(f'Number of finished trials: {len(study.trials)}')

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')

    print(f'  Params: ')
    print(trial.params)
    for param in trial.params:
        print(f'    {param}: {trial.params[param]}')

    return trial.params


def threshold(x, params):
    if x < params['threshold_0']:
        y = 0
    elif x < params['threshold_1']:
        y = 1
    elif x < params['threshold_2']:
        y = 2
    else:
        y = 3
    return y
