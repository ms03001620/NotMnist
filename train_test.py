import numpy as np
import tensorflow as tf

import data


def recognize(dataset, pb_file_path):
    """
    使用深度神经网络模型进行预测。
    :param png_path: 要预测的图片的路径。
    :param pb_file_path: 网络模型文件
    :return:
    """
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())  # rb
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            input_x = sess.graph.get_tensor_by_name("input:0")

            output = sess.graph.get_tensor_by_name("out_softmax:0")

            img_output = sess.run(output, feed_dict={input_x: dataset})
            print(np.argmax(img_output, axis=1))



dataset = data.load_letter("images", 2, 28, 255).reshape((-1, 28, 28, 1)).astype(np.float32)

# pickle_path = "output/notMNIST.pickle"
# train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels \
#     = data_util.dataset_normalize(28, 10, pickle_path)

recognize(dataset, "output/not-mnist-a-j-tf1.2.pb")



