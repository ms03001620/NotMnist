import numpy as np
import tensorflow as tf

np.random.seed(1)
random_data = np.random.rand(2,2)#/np.sqrt(2)
truncated_normal = tf.truncated_normal([2,2], seed=1, mean=0, stddev=1)
random_normal = tf.random_normal([2,2], seed=1, mean=0, stddev=1)


print(random_data)

with tf.Session() as sess:
    print("truncated_normal", sess.run(truncated_normal))
    print("random_normal", sess.run(random_normal))