import os
import shutil

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

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())  # rb
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            input_x = sess.graph.get_tensor_by_name("input:0")

            output = sess.graph.get_tensor_by_name("out_softmax:0")

            # feed_test = {graph['x_placeholder']: test_dataset, graph['y_placeholder']: test_labels, graph['keep_prob']: 1}

            image_files = os.listdir(png_path)

            target = "output/notMNIST_large/A12"
            targetR = "output/notMNIST_large/A12_right"

            for image in image_files:
                image_file = os.path.join(png_path, image)

                try:
                    dataset = data.load_letter2(image_file, 1, 28, 255).reshape((-1, 28, 28, 1)).astype(np.float32)

                    img_output = sess.run(output, feed_dict={input_x: dataset, })
                    value = np.argmax(img_output, axis=1)
                    if value != 0:
                        shutil.copy(image_file, target)
                    else:
                        shutil.copy(image_file, targetR)

                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
                    shutil.copy(image_file, target)






recognize("output/notMNIST_large/A11", "output/not-mnist-a-j-tf1.2.pb")
