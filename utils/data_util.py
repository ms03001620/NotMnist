import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorflow.python.framework import graph_util


def dataset_load_from_pickle(pickle_path):
    """
    从pickle文件中读取预存数据到指定变量中
    :param pickle_path: pickle文件所在路径
    :return: 训练集、验证集、测试集数据和标签
    """
    with open(pickle_path, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def dataset_reformat(dataset, labels, image_size, num_labels):
    """
    重构数据格式
    :param dataset: dataset: index，image_size（28）*image_size（28） -> index, 782, num_channels
    :param labels: labels: index, 1-
    :param image_size:
    :param num_labels:
    :return: 数据集、数据的标签
    """
    num_channels = 1  # grayscale
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

def dataset_normalize(image_size, num_labels, pickle_path):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dataset_load_from_pickle(pickle_path)
    train_dataset, train_labels = dataset_reformat(train_dataset, train_labels, image_size, num_labels)
    valid_dataset, valid_labels = dataset_reformat(valid_dataset, valid_labels, image_size, num_labels)
    test_dataset, test_labels = dataset_reformat(test_dataset, test_labels, image_size, num_labels)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def save_graph_to_pb(path, session, node):
    """
    保存计算图和数据
    :param path: 本地
    :param session:
    :param node:
    :return:
    """
    constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, [node])
    with tf.gfile.FastGFile(path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

def __print():
    """
    打印数据维度
    :return:
    """
    image_size = 28
    num_labels = 10
    pickle_path = "../output/notMNIST.pickle"

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels \
        = dataset_normalize(image_size, num_labels, pickle_path)

    print('reformat Training set', train_dataset.shape, train_labels.shape)
    print('reformat Validation set', valid_dataset.shape, valid_labels.shape)
    print('reformat Test set', test_dataset.shape, test_labels.shape)

#__print()
