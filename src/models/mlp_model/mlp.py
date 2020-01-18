import numpy as np
import pandas as pd

# Importa os pacotes de algoritmos de redes neurais (Keras)
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.utils import class_weight
from keras.callbacks import LearningRateScheduler
import keras


COLUMNS_NUM = 512
EPOCH_NUM = 30


class MLPModel:

    def __init__(self, y: pd.Series):
        self.y = y

    def process(self, x, y, train_ind, val_ind):
        x_train = x.iloc[train_ind]
        y_train = y.iloc[train_ind]
        x_val = x.iloc[val_ind]
        y_val = y.iloc[val_ind]
        model, loss = self.get_nn(x_train, y_train, x_val, y_val)
        return model, loss

    def calc_class_weight(self):
        class_weight_y = class_weight.compute_class_weight(
            'balanced', np.unique(self.y), self.y)
        return class_weight_y

    def define_model(self, x_tr):
        inp = Input(shape=(x_tr.shape[1],))

        x = Dense(256, input_dim=x_tr.shape[1], activation='relu')(inp)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        # x = Dense(256, activation='relu')(x)
        # x = Dropout(0.3)(x)
        # x = BatchNormalization()(x)

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        out = Dense(1)(x)

        model = Model(inp, out)
        return model

    def get_nn(self, x_tr, y_tr, x_val, y_val):
        K.clear_session()
        model = self.define_model(x_tr)
        sgd = keras.optimizers.SGD(lr=0.005, clipnorm=1., momentum=0.9)
        model.compile(optimizer=sgd,
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
        lr_decay = LearningRateScheduler(self.step_decay)
        model.fit(x_tr, y_tr,
                  validation_data=[x_val, y_val],
                  callbacks=[es, mc, lr_decay],
                  epochs=400,
                  batch_size=256,
                  verbose=1,
                  class_weight=class_weight_y,
                  shuffle=True)

        model.load_weights("best_model.h5")

        y_pred = model.predict(x_val)

        return model, y_pred

    def step_decay(self, epoch):
        x = 0.005
        if epoch >= 20:
            x = 0.001
        return x
