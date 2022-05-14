from numpy import ndarray
from dataset_read import *
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN
from sklearn.preprocessing import StandardScaler

from train import predict, train

if (__name__ == "__main__"):
    dataset = read_dataset('Data')
    X_train = dataset.train.images
    X_test = dataset.test.images
    y_train = dataset.train.labels
    y_test = dataset.test.labels

    scaler:StandardScaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    X_train = np.reshape(X_train, (X_train_transformed.shape[0], X_train_transformed.shape[1], 1))
    X_test = np.reshape(X_test, (X_test_transformed.shape[0], X_test_transformed.shape[1], 1))
  
   # Yinelemeli Sinir ağlarından hangisinin daha iyi olduğunu bulmak için accuracy değerlerini karşılaştıralım.
    model = train(X_train, y_train, GRU)
    accuracy = predict(X_test, y_test, model)
    print ("Accuuracy GRU", accuracy)

    model = train(X_train, y_train, LSTM)
    accuracy = predict(X_test, y_test, model)
    print ("Accuuracy LSTM", accuracy)


    model = train(X_train, y_train, SimpleRNN)
    accuracy = predict(X_test, y_test, model)
    print ("Accuuracy RNN", accuracy)