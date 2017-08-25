from __future__ import print_function

import os
import sys
import tarfile
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

last_percent_reported = None
def maybe_download(fileurl, filename, expected_bytes, data_root, force=False):
    """
    尝试下载文件，如果文件不存在，或大小和expected_bytes描述的不一致
    :param fileurl: url
    :param filename: 文件名称
    :param expected_bytes: 正确文件的大小
    :param data_root: 保存到本地的路径
    :param force: 是否强制下载（不考虑是否已经下载过）
    :return:
    """

    def download_progress_hook(count, blockSize, totalSize):
        """
        控制台输出下载进度
        :param count:
        :param blockSize:
        :param totalSize:
        :return:已下载的文件
        """
        global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)

        if last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
            last_percent_reported = percent


    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(fileurl + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


def maybe_extract(data_root, filename, force=False):
    """
    尝试解压缩，如果文件已解压完毕则不比再解，也可以强制解压缩
    :param data_root: 压缩文件所在路径
    :param filename: 压缩文件名
    :param force: 是否强制解压缩
    :return: 解压后的文件夹
    """
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))if os.path.isdir(os.path.join(root, d))]
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images, image_size, pixel_depth):
    """
    将指定文件夹内图片读取到内存，并修改起形状
    :param folder: 目标文件夹
    :param min_num_images:
    :param image_size: 图片尺寸。只针对该尺寸处理
    :param pixel_depth: 图片像素。只针对该像素处理
    :return: 图片内存引用
    """
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            # pip3 install --upgrade pillow
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def load_letter2(image_file, min_num_images, image_size, pixel_depth):
    image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
    return image_data


def maybe_pickle(data_folders, min_num_images_per_class, image_size, pixel_depth, force=False):
    """
    尝试将指定文件先读取到内存，然后将内存数据保存到本地。以便下次直接还原成内存文件。
    :param data_folders: 目标文件夹
    :param min_num_images_per_class:
    :param image_size: 图片尺寸，默认为正方形
    :param pixel_depth: 图片总像素
    :param force: 是否强制
    :return: 内存文件名（*.pickle)
    """
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class, image_size, pixel_depth)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def random_show_an_image_from_dataset(train_datasets):
    """
    弹出dialog，随机显示一张图片（可用来显示无法转化为内存文件的图片。以便目测其是否有问题）
    :param train_datasets: 图片集合的内存引用
    :return:
    """
    pickle_file = train_datasets[0]  # index 0 should be all As, 1 = all Bs, etc.
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)  # unpickle
        sample_idx = np.random.randint(len(letter_set))  # pick a random image index
        sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
        plt.figure()
        plt.imshow(sample_image)
        plt.pause(10)


def merge_datasets(image_size, pickle_files, train_size, valid_size=0):
    """
    从pickle文件中读取到内存，并生成其label文件（label是pickle的文件名）
    :param image_size:
    :param pickle_files:
    :param train_size:
    :param valid_size:
    :return:
    """
    def make_arrays(nb_rows, img_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
            labels = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, labels = None, None
        return dataset, labels

    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def check_overlaps(images1, images2):
    """
    去重复的数据
    :param images1:
    :param images2:
    :return:
    """
    start = time.clock()
    hash1 = set([hash(image1.tobytes()) for image1 in images1])
    hash2 = set([hash(image2.tobytes()) for image2 in images2])
    all_overlaps = set.intersection(hash1, hash2)
    return all_overlaps, time.clock() - start

def randomize(dataset, labels):
    """
    讲数据集（dataset）内顺序打乱。数据集内原先是依次存放A-J所有数据
    :param dataset:
    :param labels:
    :return:
    """
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def save_all(data_root, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    """
    保存数据到一个pickle文件（该文件可以作为正则完毕后的数据给其他程序使用）
    :param data_root: pickle文件所在目录
    :param train_dataset: 要写入的数据
    :param train_labels: 要写入的数据
    :param valid_dataset: 要写入的数据
    :param valid_labels: 要写入的数据
    :param test_dataset: 要写入的数据
    :param test_labels: 要写入的数据
    :return:
    """

    pickle_file = os.path.join(data_root, 'notMNIST.pickle')
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


def main():
    """
    该程序将从网络中下载训练图。并导出一个pickle文件作为训练使用
    下载-》解压缩-》校验图片格式（必须是28*28）打包成小的pickle -》读取到内存 -》去除重复-》随机打乱-》打包为最终约690mb左右的pickle文件
    这个文件最终可以随时读入内存并取得训练、验证、测试集数据来使用
    :return:
    """
    url = 'http://cn-static.udacity.com/mlnd/'
    data_root = 'output/'

    train_filename = maybe_download(url, "notMNIST_large.tar.gz", 247336696, data_root)
    test_filename = maybe_download(url, "notMNIST_small.tar.gz", 8458043, data_root)

    train_folders = maybe_extract(data_root, train_filename)
    test_folders = maybe_extract(data_root, test_filename)

    image_size = 28  # Pixel width and height.
    pixel_depth = 255.0  # Number of levels per pixel.

    train_datasets = maybe_pickle(train_folders, 45000, image_size, pixel_depth)
    test_datasets = maybe_pickle(test_folders, 1800, image_size, pixel_depth)

    #random_show_an_image_from_dataset(test_datasets)

    np.random.seed(133)
    train_size = 200000
    valid_size = 10000
    test_size = 10000

    valid_dataset, valid_labels, train_dataset, train_labels \
        = merge_datasets(image_size, train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels \
        = merge_datasets(image_size, test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    r, execTime = check_overlaps(train_dataset, test_dataset)
    print('Number of overlaps between training and test sets: {}. Execution time: {}.'.format(len(r), execTime))
    r, execTime = check_overlaps(train_dataset, valid_dataset)
    print('Number of overlaps between training and validation sets: {}. Execution time: {}.'.format(len(r), execTime))
    r, execTime = check_overlaps(valid_dataset, test_dataset)
    print('Number of overlaps between validation and test sets: {}. Execution time: {}.'.format(len(r), execTime))


    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    save_all(data_root, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)


#main()
