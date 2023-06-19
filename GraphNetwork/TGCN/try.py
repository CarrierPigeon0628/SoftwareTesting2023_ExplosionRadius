import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tf.compat.v1.disable_eager_execution()
#st = tf.SparseTensor(values=[1, 2], indices=[[0, 0], [1, 1]], dense_shape=[2, 2])
#dt = tf.ones(shape=[2, 2], dtype=tf.int32)
#sess = tf.compat.v1.Session()
#with sess.as_default():
#	for m in st:
#		result = tf.compat.v1.sparse_tensor_dense_matmul(m, dt)
#		print(result.eval())
v = tf.compat.v1.get_variable('trial', [3,3], initializer=tf.compat.v1.random_uniform_initializer(maxval=0., minval=1., seed=0))
u = tf.compat.v1.ones([3,1])
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
with sess.as_default():
	print(v.eval())
	v = v + tf.compat.v1.eye(3)
	a = v
	print(v.eval())
	v = tf.matmul(v, u)
	print(v.eval())
	v = tf.reshape(v, [v.shape[0]])
	v = tf.compat.v1.diag(v)
	print(v.eval())
	v = tf.linalg.inv(tf.sqrt(v))
	print(v.eval())
	v = tf.matmul(tf.matmul(v, a), v)
	print(v.eval())
