from mimetypes import init
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes
import tensorflow as tf

trainDataUrl="train_data.csv"
trainLabelUrl="train_label.csv"
testDataUrl = "test_data.csv"
testLabelUrl = "test_label.csv"

def extract_labels(filename) -> np.ndarray:
    """
    label dosyalarının okunması bunların bir array içerisine alınması işlemi
    """
    dataset = pd.read_csv(
        filename, header=None)
    datasets = dataset.astype(np.int32).values
    return datasets


def extract_images(filename) -> np.ndarray:
    """
    özniteliklerden dizi oluşturma
    """
    dataset = pd.read_csv(filename, header=None)
    # 32 Sütunluk verinin çekilmesi
    dataset = dataset.iloc[:, 0:32].values.astype(np.float32)
    # 0-255 aralığında olan veri 0-1 aralığına çekilir.
    dataset =  np.multiply(dataset, 1.0 / 255.0)
    return dataset

class DataSet(object):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """
        Veri setinin oluşturulması.
        """
        self.images = images
        self.labels = labels
class Datasets():
    def __init__(self, train:DataSet, validation:DataSet, test:DataSet) -> None:
        """
        Tüm verisetleri bir arada tutuldu, train, validation, test
        """
        self.train = train
        self.validation = validation
        self.test = test

def read_dataset(train_dir, 
    VALIDATION_SIZE = 5000)->Datasets:
    """
    veri setinin oluşturulmasında kullanılacak metottur.
    train_dir -> veri csvlerinin bulunduğu klasör
    
    """
    # eğitim için kullanılacak özniteliklerin çekilmesi
    train_images = extract_images(train_dir + "/" + trainDataUrl)
    # eğitim için kullanılacak labellerin çekilmesi
    train_labels = extract_labels(train_dir + "/" + trainLabelUrl)
    # test için kullanılacak özniteliklerin çekilmesi
    test_images = extract_images(train_dir + "/" + testDataUrl)
    # test için kullanılacak labellerin çekilmesi
    test_labels = extract_labels(train_dir + "/" + testLabelUrl)
    # 5000 ya da farklı adet belirtilmişse o kadar adet veriyi validation için ayırdık.
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_images[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    
    # train, validation ve test veri setlerini oluşturduk
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)
    return Datasets(train=train, validation=validation, test=test)