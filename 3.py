# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np

# 定义变量
state = tf.Variable(0, name = 'counter')
print(state.name)

# 定义常量
one = tf.constant(1)
# 定义新的变量，值等于state + one
new_value = tf.add(state, one)
# 定义update操作，将new_value赋值给state
update = tf.assign(state, new_value)

# 定义的变量激活，如果定义了Variable，必须有
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        # 执行update操作，update操作包括add,assign两部分
        sess.run(update)
        # 输出state结果，必须放到sess.run()中
        print sess.run(state)

