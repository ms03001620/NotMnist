# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range
from termcolor import colored

from utils import data_util


def get_weight(weight):
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(weight))
    return weight


def build_network(patch_size, image_size, num_channels, depth, num_labels, num_hidden):
    keep_prob = tf.placeholder(tf.float32, name="keep")
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name='input')
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)

        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [tf.shape(hidden)[0], shape[1] * shape[2] * shape[3]])

        fc1 = (tf.matmul(reshape, layer3_weights) + layer3_biases)

        hidden = tf.nn.relu(tf.nn.dropout(fc1, keep_prob))

        fc2 = tf.matmul(hidden, layer4_weights) + layer4_biases

        return fc2

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits, name="out_softmax")

    return dict(
        keep_prob = keep_prob,
        x_placeholder=tf_train_dataset,
        y_placeholder=tf_train_labels,
        loss=loss,
        optimizer=optimizer,
        train_prediction=train_prediction,
    )


def train_network(graph,
                  batch_size,
                  num_steps,
                  train_dataset,
                  train_labels,
                  test_dataset,
                  test_labels,
                  valid_dataset,
                  valid_labels,
                  pb_file_path):
    threshold_train = 94
    threshold_test = 96.4

    def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    def recording(accuracy,
                  graph,
                  pb_file_path,
                  session,
                  test_dataset,
                  test_labels,
                  valid_dataset,
                  valid_labels,
                  threshold_test):

        feed_test = {graph['x_placeholder']: test_dataset,
                     graph['y_placeholder']: test_labels,
                     graph['keep_prob']: 1}
        acc = accuracy(session.run(graph['train_prediction'], feed_dict=feed_test), test_labels)

        print(colored('Recording Test accuracy: %f%%'% acc, 'red'))
        if acc > threshold_test:
            data_util.save_graph_to_pb(pb_file_path, session, "out_softmax")
            return acc
        return threshold_test

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        size = len(train_labels)

        for step in range(num_steps):
            start = (step * batch_size) % size
            end = min(start + batch_size, size)

            batch_labels = train_labels[start:end]

            feed_dict = {graph['x_placeholder']: train_dataset[start:end],
                         graph['y_placeholder']: batch_labels,
                         graph['keep_prob']: .75}

            _, loss, predictions = session.run([graph['optimizer'], graph['loss'], graph['train_prediction']], feed_dict=feed_dict)

            if step % 100 == 0:
                acc = accuracy(predictions, batch_labels)
                #print('%d\t step, accuracy %0.1f%% lost %f' % (step, acc, loss))
                print('%d\t step, accuracy %0.1f%% ' % (step, acc))

                if acc >= threshold_train:
                    threshold_test = recording(accuracy,
                                               graph,
                                               pb_file_path,
                                               session,
                                               test_dataset,
                                               test_labels,
                                               valid_dataset,
                                               valid_labels,
                                               threshold_test)
                    #threshold_train = acc






def main():
    patch_size = 5
    batch_size = 100
    image_size = 28
    num_channels = 1
    depth = 16
    num_labels = 10
    num_hidden = 64
    pickle_path = "output/notMNIST.pickle"

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels \
        = data_util.dataset_normalize(image_size, num_labels, pickle_path)

    graph = build_network(patch_size, image_size, num_channels, depth, num_labels, num_hidden)
    num_steps = 10001999999

    pb_file_path = "output/not-mnist-a-j-tf1.2.pb"

    print(train_dataset.shape)
    train_network(graph,
                  batch_size,
                  num_steps,
                  train_dataset,
                  train_labels,
                  test_dataset,
                  test_labels,
                  valid_dataset,
                  valid_labels,
                  pb_file_path)

main()
