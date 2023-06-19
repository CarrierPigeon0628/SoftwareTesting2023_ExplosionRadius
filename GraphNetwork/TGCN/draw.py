# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from TGCN.input_data_null_adj import preprocess_data,load_sz_data,load_los_data
from TGCN.tgcn_null_adj import tgcnCell
#from gru import GRUCell 

from TGCN.visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt
import time

import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
time_start = time.time()
###### Settings ######
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 5000, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 12, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 3, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('dataset', 'los', 'sz or los.')
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
if data_name == 'sz':
    data = load_sz_data('sz')
if data_name == 'los':
    data = load_los_data('los')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)
data1 = np.asarray(data1, dtype=np.float32)

#### normalization
#max_value = np.max(data1)
#data1  = data1/max_value
scaler = MinMaxScaler(feature_range=(0, 1))
data1 = scaler.fit_transform(data1)
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(gru_units, num_nodes=num_nodes)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.compat.v1.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states
        
###### placeholders ######
tf.compat.v1.disable_eager_execution()
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.compat.v1.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.compat.v1.random_normal([pre_len]),name='bias_o')}

if model_name == 'tgcn':
    pred,ttts,ttto = TGCN(inputs, weights, biases)

y_pred = pred
      

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
checkpoint_path = './TGCN/TGCN_pre_0.932222381234169-126'
saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
saver.restore(sess,checkpoint_path)

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(out,path1)
#if not os.path.exists(path):
#    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var
 
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
max_value = 1.0

epoch = 0

#for m in range(totalbatch):
#   mini_batch = trainX[m * batch_size : (m+1) * batch_size]
#   mini_label = trainY[m * batch_size : (m+1) * batch_size]
#   loss1, rmse1, train_output = sess.run([loss, error, y_pred],
#                                            feed_dict = {inputs:mini_batch, labels:mini_label})
#   batch_loss.append(loss1)
#   batch_rmse.append(rmse1 * max_value)

#train_label = np.reshape(trainY, [-1,num_nodes])
#train_output = sess.run(y_pred, feed_dict={inputs:trainX})
#train_output = np.maximum(train_output, 0)
#errors = train_label - train_output
#errors_output = pd.DataFrame(errors)
#errors_output.to_csv('./errors.csv', index=False, header=False)

# testX = trainX
# testY = trainY

loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
#test_output = np.maximum(test_output, 0)
test_label = np.reshape(testY,[-1,num_nodes])
test_label = scaler.inverse_transform(test_label)
#test_label /= 1000

#feedforward = load_model('./feed_forward.h5')
#errors = feedforward.predict(testX)
#errors = np.reshape(errors, [-1, num_nodes])
#test_output = test_output + errors

test_output = scaler.inverse_transform(test_output)
test_output = np.maximum(test_output, 0)
#test_output /= 1000
rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
test_label1 = test_label * max_value
test_output1 = test_output * max_value
test_loss.append(loss2)
test_rmse.append(rmse * max_value)
test_mae.append(mae * max_value)
test_acc.append(acc)
test_r2.append(r2_score)
test_var.append(var_score)
test_pred.append(test_output1)

#test_output = pd.DataFrame(test_output)
#test_output.to_csv('../ones_output.csv', index=False, header=False)


#print('Iter:{}'.format(0),
#     'rmse:{:.4}'.format(rmse),
#     'mae:{:.4}'.format(mae),
#     'acc:{:.4}'.format(acc),
#     'r2:{:.4}'.format(r2_score),
#     'var:{:.4}'.format(var_score))

def test():
    test_label = []
    test_output = []
    for i in range(0, test_output1.shape[0], pre_len):
        test_label.append(test_label1[i].tolist())
        test_output.append(test_output1[i].tolist())
    test_label = np.array(test_label).T
    test_output = np.array(test_output).T
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    print('TS:{}'.format(1),
        'rmse:{:.4}'.format(rmse),
        'mae:{:.4}'.format(mae),
        'acc:{:.4}'.format(acc),
        'r2:{:.4}'.format(r2_score),
        'var:{:.4}'.format(var_score))

    # test_label = []
    # test_output = []
    # for i in range(1, test_output1.shape[0], pre_len):
    #     test_label.append(test_label1[i].tolist())
    #     test_output.append(test_output1[i].tolist())
    # test_label = np.array(test_label)
    # test_output = np.array(test_output)
    # rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    # print('TS:{}'.format(2),
    #     'rmse:{:.4}'.format(rmse),
    #     'mae:{:.4}'.format(mae),
    #     'acc:{:.4}'.format(acc),
    #     'r2:{:.4}'.format(r2_score),
    #     'var:{:.4}'.format(var_score))
    #
    # test_label = []
    # test_output = []
    # for i in range(2, test_output1.shape[0], pre_len):
    #     test_label.append(test_label1[i].tolist())
    #     test_output.append(test_output1[i].tolist())
    # test_label = np.array(test_label)
    # test_output = np.array(test_output)
    # rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    # print('TS:{}'.format(3),
    #     'rmse:{:.4}'.format(rmse),
    #     'mae:{:.4}'.format(mae),
    #     'acc:{:.4}'.format(acc),
    #     'r2:{:.4}'.format(r2_score),
    #     'var:{:.4}'.format(var_score))

    fig = plt.figure(figsize=(20, 50))
    for num in range(num_nodes):
        plt.subplot(int((num_nodes + 1) / 2), 2, num + 1)
        plt.plot(test_label[num], linestyle='-', color='b', label='True')
        plt.plot(test_output[num], linestyle='-', color='r', label='Pred')
        plt.xlabel('Time', fontsize=14, fontweight='bold')
        plt.ylabel('Kpi ' + data.columns.tolist()[num], fontsize=14, fontweight='bold')
        plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./test_result.png')

    # fig, axes = plt.subplots(int((num_nodes + 1) / 2), 2, figsize=(20, 30))
    # for num in range(num_nodes):
    #     subplot = axes[num]
    #     subplot.plot(test_label[num], color='b', label='True')
    #     subplot.plot(test_output[num], color='r', label='Pred')
    #     subplot.set_xlabel('Time', fontsize=14, fontweight='bold')
    #     subplot.set_ylabel('Kpi ' + data.columns.tolist()[num], fontsize=14, fontweight='bold')
    #     subplot.legend()
    # plt.tight_layout()

    return fig, test_label[2], test_output[2], test_label[32], test_output[32]
