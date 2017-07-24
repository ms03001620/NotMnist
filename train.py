import numpy as np
import tensorflow as tf
from six.moves import range
from tensorflow.python.framework import graph_util

from utils import data_util


def build_network(patch_size, batch_size, image_size, num_channels, depth, num_labels, num_hidden, valid_dataset,test_dataset):
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    one_image_dataset = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels), name='input')

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(
        tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        #hidden = tf.nn.relu(conv + layer1_biases)

        hidden = tf.nn.relu(tf.nn.dropout((conv + layer1_biases), 0.5))

        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits, name="out_softmax")

    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    one_image_prediction = tf.argmax(tf.nn.softmax(model(one_image_dataset), 1, name='argmax_output'))

    return dict(
        x_placeholder=tf_train_dataset,
        y_placeholder=tf_train_labels,
        loss=loss,
        optimizer=optimizer,
        train_prediction=train_prediction,
        valid_prediction=valid_prediction,
        test_prediction=test_prediction
    )


def train_network(graph, batch_size, num_steps, train_dataset, train_labels, valid_labels, test_labels, pb_file_path):
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {graph['x_placeholder']: batch_data, graph['y_placeholder']: batch_labels}

            _, l, predictions = session.run([graph['optimizer'], graph['loss'], graph['train_prediction']],
                                            feed_dict=feed_dict)

            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(graph['valid_prediction'].eval(), valid_labels))

        print('Test accuracy: %.1f%%' % accuracy(graph['test_prediction'].eval(), test_labels))

        # 训练完后就保存为pb文件
        constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, ["argmax_output"])
        with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def main():
    patch_size = 5
    batch_size = 67
    image_size = 28
    num_channels = 1
    depth = 16
    num_labels = 10
    num_hidden = 64
    pickle_path = "notMNIST.pickle"

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels \
        = data_util.dataset_normalize(image_size, num_labels, pickle_path)

    graph = build_network(patch_size, batch_size, image_size, num_channels, depth, num_labels, num_hidden,valid_dataset, test_dataset)
    num_steps = 1001

    #pb_file_path = "output/mnist-tf1.0.1.pb"
    pb_file_path = "output/not-mnist-a-j-tf1.2.pb"
    train_network(graph, batch_size, num_steps, train_dataset, train_labels, valid_labels, test_labels, pb_file_path)

main()
