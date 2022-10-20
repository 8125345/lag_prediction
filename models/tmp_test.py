import os
import numpy as np

# test_arr = np.array([
#     [1, 111],
#     [1, 222],
#     [2, 1123],
#     [3, 666],
# ])
# print(test_arr[:, 0])
#
# num = 3
# res = [test_arr for k in np.unique(test_arr[:, 0])]

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# import tensorflow as tf
# import numpy as np
#
# for i in range(5):
#     r = tf.random.uniform(shape=[], minval=-160, maxval=160, dtype=tf.int64)
#     print(r)

# --------------------

# x = np.array([
#     [1., 2.],
#     [3., 4.],
#     [5., 6.]
# ], dtype=np.float32).reshape((1, 3, 2, 1))
# kernel = np.array([
#     [1., 2.],
#     [3., 4]
# ], dtype=np.float32).reshape((2, 1, 1, 2))
# rst = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
#                              padding='VALID').numpy()
#
# print(x.shape)
# print(kernel.shape)
# print(rst.shape)

# --------------------

# x_in = np.array([[
#     [[2], [1], [2], [0], [1]],
#     [[1], [3], [2], [2], [3]],
#     [[1], [1], [3], [3], [0]],
#     [[2], [2], [0], [1], [1]],
#     [[0], [0], [3], [1], [2]], ]])
# print(x_in.shape)
# kernel_in = np.array([
#     [[[2, 0.1]], [[3, 0.2]]],
#     [[[0, 0.3]], [[1, 0.4]]], ])
# print(kernel_in.shape)
# x = tf.constant(x_in, dtype=tf.float32)
# kernel = tf.constant(kernel_in, dtype=tf.float32)
#
# rst = tf.nn.conv2d(x, kernel, strides=[1, 0, 1, 1], padding='VALID')
#
# print(rst.shape)
# --------------------
# x_in = np.array([
#     [
#         [0],
#         [1],
#         [2],
#         [3],
#         [0],
#         [0],
#         [1],
#         [2],
#         [3],
#         [0],
#     ],
# ])
# # print(x_in[:, :, 1])
# kernel_in = np.array([
#     [[1, ]],
#     [[2, ]],
#     [[3, ]],
#     [[0, ]],
#     # [[0, ]],
#     # [[1, ]],
#     # [[2, ]],
#     # [[3, ]],
#     # [[0, ]],
#     # [[0, ]],
# ])
# # kernel_in = np.array([
# #     [[1, 1, 0]],
# #     [[2, 1, 0]],
# #     [[3, 1, 0]],
# #
# # ])
# x = tf.constant(x_in, dtype=tf.float32)
# kernel = tf.constant(kernel_in, dtype=tf.float32)
#
# print(x.shape)
# print(kernel_in.shape)
#
# output = tf.nn.conv1d(x, kernel, stride=1, padding='SAME')
# print(output)
# ------------------------
# kernel = tf.constant(kernel_in, dtype=tf.float32)
# output = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
# print(output)
# print(output.shape)

# x_in = np.array([[
#     [[2], [1], [2], [0], [1]],
#     [[1], [3], [2], [2], [3]],
#     [[1], [1], [3], [3], [0]],
#     # [[2], [2], [0], [1], [1]],
#     # [[0], [0], [3], [1], [2]],
# ]])
# kernel_in = np.array([
#     [[[2, 0.1]], [[3, 0.2]]],
#     # [[[0, 0.3]], [[1, 0.4]]],
#
# ])
# x = tf.constant(x_in, dtype=tf.float32)
#
# print(x.shape)
# print(kernel_in.shape)
# kernel = tf.constant(kernel_in, dtype=tf.float32)
# output = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
# print(output)
# print(output.shape)
