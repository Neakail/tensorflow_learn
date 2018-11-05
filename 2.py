# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np

# 定义两个矩阵
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

# 定义矩阵乘法
product = tf.matmul(matrix1, matrix2)

# 运行矩阵乘法，session用法一
sess = tf.Session()
result = sess.run(product)
print 'Session用法一'
print result
sess.close()

## session用法二，不用考虑close，会自动关闭

with tf.Session() as sess:
    result = sess.run(product)
    print 'Session用法二'
    print result
