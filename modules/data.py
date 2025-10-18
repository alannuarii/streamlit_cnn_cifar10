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

def get_team_members():
    return [
        {"NIM": "2802643186", "Name": "Alan Nuari"},
        {"NIM": "2802642214", "Name": "Alvin Saputra Zaelani"},
        {"NIM": "2802641312", "Name": "Bayu Bagus Bagaswara"},
        {"NIM": "2802640266", "Name": "Raditya Firman Syaputra"},
        {"NIM": "2802646004", "Name": "Samuel Parlinggoman"},
    ]
