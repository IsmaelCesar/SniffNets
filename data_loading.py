"""
    This module contains all the procedures necessary for
    loading the data.
"""
import pickle as pkl
import numpy as np
import logging
import sklearn
from sklearn.model_selection import train_test_split

# DATA_FOLDER = "data/"

EXTENSION = ".pkl"
DATA_SETS = {
    "fonollosa": {0: "B1-system",
                  1: "B2-system",
                  2: "B3-system",
                  3: "B4-system",
                  4: "B5-system",
                  "n_classes": 4
                  },
    "turbulent_gas_mixtures": {0: "preloaded_dataset", "n_classes": 4},
    "windtunnel": {
                0: "preloaded_dataset-L1",
                1: "preloaded_dataset-L2",
                2: "preloaded_dataset-L3",
                3: "preloaded_dataset-L4",
                4: "preloaded_dataset-L5",
                5: "preloaded_dataset-L6",
                "n_classes": 11,
                }
    # Uncomment the following line if you have been authorized to use the dataset
    # ,"coffee_dataset": {0: "preloaded_dataset", "n_classes": 3}
}
DS_WINE = {"QWines-CsystemTR": 3,
           "QWinesEa-CsystemTR": 4}


def load(ds_choice, ds_idx=0):
    """
    choices : 0 -> fonollosa, 1 -> turbulent_gas_mixtures, 2 -> windtunnel
    :param ds_choice: the index naming the dataset chosen
    :param ds_idx: the index if the folder containing the dataset has one or more datasets
    :return: the dataset read,the labels and  the number of classes
    """
    global DATA_FOLDER, DATA_SETS, EXTENSION
    assert ds_choice in list(DATA_SETS.keys())
    # ds_name =[ds_choice]
    ds_name = ds_choice
    dataset_name = ds_name+"/"

    logging.info(ds_name + " Is being loaded")

    n_classes = DATA_SETS[ds_name]['n_classes']
    print("\n\n ds_name:"+ds_name+"\n\n")
    print("\n\n ds_idx"+str(ds_idx)+"\n\n")
    subds_name = DATA_SETS[ds_name][ds_idx]
    sub_dataset_name = subds_name + "/"

    data, labels = None, None
    with open(DATA_FOLDER + ds_name + "/" + subds_name + EXTENSION, 'rb') as d:
        data, labels, _ = pkl.load(d)
        d.close()

    return data, labels, n_classes, dataset_name, sub_dataset_name


def load_wine(ds_choice):
    """
    choices : 0 -> QWines-CsystemTR, 1 -> QWinesEa-CsystemTR
    :param ds_choice: the index naming the dataset chosen
    :return: the dataset read,the labels and  the number of classes
    """
    global DATA_FOLDER, DS_WINE, EXTENSION

    assert ds_choice in list(DS_WINE.keys())
    ds_name = ds_choice
    dataset_name = ds_name+'/'

    logging.info(ds_name + " Is being loaded")

    n_classes = DS_WINE[ds_name]

    data, labels = None, None
    with open(DATA_FOLDER + "wines/" + ds_name + EXTENSION, "rb") as d:
        data, labels, _, _ = pkl.load(d)
        d.close()

    return data, labels, n_classes, dataset_name, None # sub_dataset_name


def data_set_reshaped(data_set):
    new_data = []
    for d in data_set:
        new_data.append(d.reshape(d.shape[0], d.shape[1], 1).tolist())
    return np.array(new_data)


def load_and_split(ds_choice, ds_idx=0, read_wine_datasets=False):
    # Loading dataset
    data = None
    labels = None
    if not read_wine_datasets:
        data, labels, n_classes, dataset_name, sub_dataset_name = load(ds_choice, ds_idx)
    else:
        data, labels, n_classes, dataset_name, sub_dataset_name = load_wine(ds_choice)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2)
    train_data = data_set_reshaped(train_data)
    test_data = data_set_reshaped(test_data)

    # input_shape = train_data[0].shape

    train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)
    test_data, test_labels = sklearn.utils.shuffle(test_data, test_labels)

    return train_data, train_labels, test_data, test_labels


def standardize_data(train_data, test_data, input_shape):

    flat_train_data = train_data.reshape(train_data.shape[0], input_shape[0] * input_shape[1])
    flat_test_data = test_data.reshape(test_data.shape[0], input_shape[0] * input_shape[1])

    scaler = sklearn.preprocessing.StandardScaler().fit(flat_train_data)
    flat_train_data = scaler.transform(flat_train_data)

    scaler = sklearn.preprocessing.StandardScaler().fit(flat_test_data)
    flat_test_data = scaler.transform(flat_test_data)

    new_train = flat_train_data.reshape(train_data.shape[0], input_shape[0], input_shape[1], 1)
    new_test = flat_test_data.reshape(test_data.shape[0], input_shape[0], input_shape[1], 1)
    return new_train, new_test


def split_datasamples_by_sensors(data):
    """
        This is an auxiliary procedure for executing the
        SniffMultinose model split turn each column of
        the data matrix into an individual vector.
    :param data: matrix of signals encoded in an numpy array of doubles
    :return: a list with each column of the data matrix saved in a
            list item different
    """
    shape = data.shape
    new_split = []
    # Iterate over data columns
    for i in range(shape[2]):
        new_split.append(data[:, :, i])
        new_split[i] = new_split[i].reshape(new_split[i].shape[0], new_split[i].shape[1])
    return new_split


def load_dataset(ds_choice, ds_idx, read_wine_datasets=False):
    """
    Loads the dataset from the experiment
    :param ds_choice: Name of the dataset_chosen
    :param ds_idx: index indicating wich subset should be loaded
    :param read_wine_datasets:  True, if it is desired to read the wine dataset
    :return: data_samples,
            data labels,
            name of the dataset and name of the data subset,
            name of the input_shape
    """
    data = None
    labels = None
    dataset_name = None
    sub_dataset_name = None
    if not read_wine_datasets:
        data, labels, n_classes, dataset_name, sub_dataset_name = load(ds_choice, ds_idx)
    else:
        data, labels, n_classes, dataset_name, sub_dataset_name = load_wine(ds_choice)

    data = np.array(data)

    input_shape = data[0].shape

    return data, labels, n_classes, dataset_name, sub_dataset_name, input_shape


def dataset_classes_number(dataset_name):
    global DATA_SETS
    return DATA_SETS[dataset_name]["n_classes"]


def dataset_wine_classes_number(dataset_name):
    global DS_WINE
    return DS_WINE[dataset_name]


if __name__ == "data_loading":
    global DATA_FOLDER
    DATA_FOLDER = "data/"
