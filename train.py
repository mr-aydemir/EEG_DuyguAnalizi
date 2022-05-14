from keras.layers import Dense
from keras.layers import RNN
from keras.regularizers import l2
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import numpy as np


def train(x, y, rnn: RNN) -> Sequential:
    model = Sequential()
    model.add(rnn(4, input_shape=x.shape[1:], kernel_regularizer=l2(
        0.01), dropout=0.4, recurrent_dropout=0.4, recurrent_regularizer=l2(0.4), bias_regularizer=l2(0.4)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x, y, epochs=100, batch_size=4096, verbose=2)
    return model


def predict(x_test, y_test, model: Sequential):
    # make prediction
    testPredict = model.predict(x_test)
    print(testPredict)
    print(y_test)
    print(label_tranform(testPredict))
    accuracy = accuracy_score(y_test, label_tranform(testPredict))
    return accuracy


def label_tranform(labels):
    values = []
    for i in range(0, len(labels)):
        values.append(round(labels[i][0]))
    return np.asarray(values, np.int32)
