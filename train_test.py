import numpy as np
import tensorflow as tf

import data


def recognize(png_path, pb_file_path):
    """
    使用深度神经网络模型进行预测。
    :param png_path: 要预测的图片的路径。
    :param pb_file_path: 网络模型文件
    :return:
    """
    dataset = data.load_letter(png_path, 2, 28, 255).reshape((-1, 28, 28, 1)).astype(np.float32)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())  # rb
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            keep = sess.graph.get_tensor_by_name("keep:0")

            input_x = sess.graph.get_tensor_by_name("input:0")

            output = sess.graph.get_tensor_by_name("out_softmax:0")

            img_output = sess.run(output, feed_dict={input_x: dataset, keep:1})
            print(np.argmax(img_output, axis=1))


recognize("images/", "output/not-mnist-a-j-tf1.2.pb")

