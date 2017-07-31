import numpy as np
import tensorflow as tf
from six.moves import range

from utils import data_util


def build_network(patch_size, image_size, num_channels, depth, num_labels, num_hidden):
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

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(tf.nn.dropout((conv + layer1_biases), 0.5))
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [tf.shape(hidden)[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits, name="out_softmax")

    return dict(
        x_placeholder=tf_train_dataset,
        y_placeholder=tf_train_labels,
        loss=loss,
        optimizer=optimizer,
        train_prediction=train_prediction,
    )


def train_network(graph, batch_size, num_steps, train_dataset, train_labels, test_dataset, test_labels, pb_file_path):
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    save_threshold = 80.0

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        size = len(train_labels)

        for step in range(num_steps):
            start = (step * batch_size) % size
            end = min(start + batch_size, size)

            batch_labels = train_labels[start:end]

            feed_dict = {graph['x_placeholder']: train_dataset[start:end], graph['y_placeholder']: batch_labels}

            _, loss, predictions = session.run([graph['optimizer'], graph['loss'], graph['train_prediction']], feed_dict=feed_dict)

            if step % 100 == 0:
                acc = accuracy(predictions, batch_labels)
                print('%d\t step, accuracy %0.1f%% lost %f' % (step, acc, loss))

                if acc >= save_threshold:
                    recording(accuracy, graph, pb_file_path, session, test_dataset, test_labels)
                    save_threshold = acc

        recording(accuracy, graph, pb_file_path, session, test_dataset, test_labels)


def recording(accuracy, graph, pb_file_path, session, test_dataset, test_labels):
    feed_test = {graph['x_placeholder']: test_dataset, graph['y_placeholder']: test_labels}
    print('Recording Test accuracy: %.1f%%' % accuracy(session.run(graph['train_prediction'], feed_dict=feed_test), test_labels))
    data_util.save_graph_to_pb(pb_file_path, session, "out_softmax")


def main():
    patch_size = 5
    batch_size = 200
    image_size = 28
    num_channels = 1
    depth = 16
    num_labels = 10
    num_hidden = 64
    pickle_path = "output/notMNIST.pickle"

    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels \
        = data_util.dataset_normalize(image_size, num_labels, pickle_path)

    graph = build_network(patch_size, image_size, num_channels, depth, num_labels, num_hidden)
    num_steps = 1000

    pb_file_path = "output/not-mnist-a-j-tf1.2.pb"

    print(train_dataset.shape)
    train_network(graph, batch_size, num_steps, train_dataset, train_labels, test_dataset, test_labels, pb_file_path)

main()
