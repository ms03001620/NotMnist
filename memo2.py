import numpy as np
from memory_profiler import profile
from six.moves import cPickle as pickle


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

@profile
def test1():
    image_size = 28
    num_labels = 10
    pickle_path = "output/notMNIST40k.pickle"

    with open(pickle_path, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        train_dataset, train_labels = dataset_reformat(train_dataset, train_labels, image_size, num_labels)
        valid_dataset, valid_labels = dataset_reformat(valid_dataset, valid_labels, image_size, num_labels)
        test_dataset, test_labels = dataset_reformat(test_dataset, test_labels, image_size, num_labels)



if __name__ == "__main__":
    test1()
