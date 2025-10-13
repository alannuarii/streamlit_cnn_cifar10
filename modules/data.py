from tensorflow.keras.datasets import cifar10
import numpy as np

def load_cifar10():
    return cifar10.load_data()

def get_class_names():
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

def class_distribution(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    return unique, counts
