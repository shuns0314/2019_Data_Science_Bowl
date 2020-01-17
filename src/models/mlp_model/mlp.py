import numpy as np
import pandas as pd

# Importa os pacotes de algoritmos de redes neurais (Keras)
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.utils import class_weight

from loss_function import qwk, threshold


COLUMNS_NUM = 512
EPOCH_NUM = 30
PARAMS = {
    'threshold_0': 1.12,
    'threshold_1': 1.62,
    'threshold_2': 2.20
    }


class MLPModel:

    def __init__(self, y: pd.Series):
        self.y = y

    def calc_class_weight(self):
        class_weight_y = class_weight.compute_class_weight(
            'balanced', np.unique(self.y), self.y)
        return class_weight_y

    def get_nn(self, x_tr, y_tr, x_val, y_val, shape, class_weight_y):
        K.clear_session()

        inp = Input(shape=(x_tr.shape[1],))

        x = Dense(1024, input_dim=x_tr.shape[1], activation='relu')(inp)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)

        out = Dense(1)(x)

        model = Model(inp, out)
        model.compile(optimizer='Adam',
                      loss='mse')

        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           restore_best_weights=True,
                           verbose=1,
                           patience=20)

        mc = ModelCheckpoint('best_model.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1,
                             save_weights_only=True)

        class_weight_y = self.calc_class_weight()
        model.fit(x_tr, y_tr,
                  validation_data=[x_val, y_val],
                  callbacks=[es, mc],
                  epochs=100,
                  batch_size=128,
                  verbose=1,
                  class_weight=class_weight_y,
                  shuffle=True)

        model.load_weights("best_model.h5")

        y_pred = model.predict(x_val)

        func = np.frompyfunc(threshold, 2, 1)
        test_pred = func(y_pred, PARAMS)
        loss = qwk(test_pred, y_val)
        print(loss[0])
        return model, loss
