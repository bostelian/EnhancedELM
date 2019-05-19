import cv2
from smallnorb.dataset import SmallNORBDataset
from utils.read_idx import read_idx, unpickle
import numpy as np
from sklearn.model_selection import train_test_split

def dataset_wrapper(dataset = None, data_resize = False, data_size = None, data_flatten = False, test_size = 0.3, single_channel=False):
    if dataset == 'norb':
        X, y = norb_wrapper(single_channel)

    if dataset == 'mnist':
        X, y = mnist_wrapper()

    if dataset == 'cifar10':
        X, y = cifar10_wrapper()
    else:
        Exception("Invalid dataset")

    if data_resize == True and data_size != None:
        X = resize(X, data_size)
    
    if data_flatten == True:
        X = flatten(X)

    return train_test_split(X, y, test_size=test_size, random_state=42)       


def norb_wrapper(single_channel=False):
    dataset = SmallNORBDataset(dataset_root='databases/small_norb_root')

    X = []
    y = []

    for data in dataset.data['train']:
        left = data.image_lt.flatten()
        right = data.image_rt.flatten()
        if single_channel == True:
            result = data.image_lt
        else:
            result = np.append(left, right).reshape([2,96,96]).transpose([1,2,0])
        X.append(result)
        y.append(data.category)

    for data in dataset.data['test']:
        left = data.image_lt.flatten()
        right = data.image_rt.flatten()
        if single_channel == True:
            result = data.image_lt
        else:
            result = np.append(left, right).reshape([2,96,96]).transpose([1,2,0])
        X.append(result)
        y.append(data.category)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def mnist_wrapper():
    X_train = read_idx("databases/mnist/train-images-idx3-ubyte.gz")
    y_train = read_idx("databases/mnist/train-labels-idx1-ubyte.gz")
    X_test= read_idx("databases/mnist/t10k-images-idx3-ubyte.gz")
    y_test = read_idx("databases/mnist/t10k-labels-idx1-ubyte.gz")

    X = []
    y = []

    for data in X_train:
        X.append(data)
    for data in X_test:
        X.append(data)
    for data in y_train:
        y.append(data)
    for data in y_test:
        y.append(data)
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def cifar10_wrapper():
    batch1 = unpickle("databases/cifar10/data_batch_1")
    batch2 = unpickle("databases/cifar10/data_batch_2")
    batch3 = unpickle("databases/cifar10/data_batch_3")
    batch4 = unpickle("databases/cifar10/data_batch_4")
    batch5 = unpickle("databases/cifar10/data_batch_5")
    batch6 = unpickle("databases/cifar10/test_batch")

    X = []
    y = []

    for data in batch1[b'data']:
        X.append(data)
    for data in batch1[b'labels']:
        y.append(data)
    for data in batch2[b'data']:
        X.append(data)
    for data in batch2[b'labels']:
        y.append(data)
    for data in batch3[b'data']:
        X.append(data)
    for data in batch3[b'labels']:
        y.append(data)
    for data in batch4[b'data']:
        X.append(data)
    for data in batch4[b'labels']:
        y.append(data)
    for data in batch5[b'data']:
        X.append(data)
    for data in batch5[b'labels']:
        y.append(data)
    for data in batch6[b'data']:
        X.append(data)
    for data in batch6[b'labels']:
        y.append(data)

    X = np.asarray(X)
    y = np.asarray(y)

    X = convert_cifar(X)
    return X, y

def convert_cifar(raw):  
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def resize(X, size):
    X_resized = []
    for data in X:
        X_resized.append(cv2.resize(data, dsize=size, interpolation=cv2.INTER_CUBIC))
    
    return np.asarray(X_resized)

def flatten(X):
    samples = X.shape[0]
    width = X.shape[1]
    height = X.shape[2]
    depth = X.shape[3] if len(X.shape) > 3 else 1
    return X.flatten().reshape(samples, width * height * depth)
