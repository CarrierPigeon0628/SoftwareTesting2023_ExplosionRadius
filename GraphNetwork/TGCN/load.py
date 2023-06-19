import tensorflow as tf
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tf.compat.v1.disable_eager_execution()
#checkpoint_path='./out/tgcn/tgcn_los_lr0.001_batch32_unit64_seq30_pre5_epoch500/model_100/TGCN_pre_0.3854-19'
#with tf.compat.v1.Session() as sess:
#	saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
#	saver.restore(sess,checkpoint_path)
#	weights = {
#		'out': tf.convert_to_tensor(sess.run('weight_o:0'), dtype=tf.float32)
#	}
#	biases = {
#		'out': tf.convert_to_tensor(sess.run('bias_o:0'), dtype=tf.float32)
#	}
#	print(weights['out'])
#	print(biases['out'])
#reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#    print("tensor_name:", key)
#adj = tf.compat.v1.get_variable('adj_o', [10, 10], initializer=tf.compat.v1.random_uniform_initializer(minval=0., maxval=1.))
#
#with tf.compat.v1.Session() as sess:
#	saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
#	saver.restore(sess,checkpoint_path)
#	adj = (adj + tf.compat.v1.transpose(adj)) / 2
#	adj = tf.compat.v1.maximum(tf.compat.v1.minimum(adj, 1), 0)
#	adj = adj.eval()
#	adj = pd.DataFrame(adj, index=None, columns=None)
#	adj.to_csv('./adj.csv')

data = pd.read_csv(r'/home/External/ldd/xiajiaxing/data/HomeC.csv')
num_nodes = data.shape[1]

adj = tf.compat.v1.get_variable('adj_o', [num_nodes, num_nodes], initializer=tf.compat.v1.random_uniform_initializer(minval=0., maxval=1.))

checkpoint_path = './out/tgcn/SmartHouse_tgcn_los_lr0.001_batch32_unit64_seq12_pre3_epoch5000/model_100/TGCN_pre_0.4081-110'
saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
saver.restore(sess,checkpoint_path)

adj = (adj + tf.compat.v1.transpose(adj)) / 2
adj = tf.compat.v1.maximum(adj, 0)
adj = adj + tf.compat.v1.eye(num_nodes)
adj = tf.compat.v1.minimum(adj, 1)
adj = adj - tf.compat.v1.eye(num_nodes)

adj = sess.run(adj)
adj = pd.DataFrame(adj, index=data.columns.tolist(), columns=data.columns.tolist())
adj.to_csv('./trained_adj.csv')

