import tensorflow as tf
import numpy as np
import PIL.Image as Image


def recognize(list, pb_file_path):
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
            print(input_x)

            output = sess.graph.get_tensor_by_name("argmax_output:0")
            print(output)

            for png_path in list:
                img_datas = np.array(Image.open(png_path).convert('L')).reshape(1, 28, 28, 1)
                img_output = sess.run(output, feed_dict={
                    input_x: img_datas,
                })
                print(img_output)


list = ["images/a0.png",
        "images/a1.png",
        "images/b0.png",
        "images/j0.png"]

recognize(list, "output/not-mnist-a-j-tf1.2.pb")



