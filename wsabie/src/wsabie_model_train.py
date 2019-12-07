#-*- coding:utf-8 -*-
import sys
import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

class DataLoader():
	'''DataLoader类用于从文件加载数据。
	文件的第一行格式为m \t n \t b：
	--m：表示左bow字典大小；
	--n：和右bow字典大小；
	--b：表示一组训练数据的行数。
	后续每一行为一个样本，每b（例如11）个样本为一组训练数据，称为one batch。
	每个one_batch的第一行为正样本，其余b-1行为负样本。
	每行的格式为：label \t left_bow_vec \t right_bow_vec，其中bow_vec用word的index表示。
	--left_bow_vec格式为：w_0 w_2 ... w_i，i in [0, m]；
	--right_bow_vec格式为：w_0 w_2 ... w_j，j in [0, m]。
	注意，左右bow字典是独立的，即左右特征的原始特征在不同的特征空间。
	'''
	def __init__(self, filename):
		self.filename = filename
		self.left_bow_size = 0
		self.right_bow_size = 0
		self.one_batch = 0
		self.left_vec_size = 0
		self.right_vec_size = 0
		self.lX = []
		self.rX = []
		self.X = None
		self.y = []

	def load_data(self):
		'''从filename中加载数据。'''
		cnt = 0
		fin = open(self.filename, 'r')
		for line in fin:
			content = line.strip().split('\t')
			if cnt == 0:
				if len(content) < 3:
					print >> sys.stderr, "第一行的组织方式必须为：「left_bow_size \\t right_bow_size \\t one_batch_size」"
					return
				try:
					self.left_bow_size, self.right_bow_size, self.one_batch =[int(v) for v in content[:3]]
				except Exception as e:
					print >> sys.stderr, e
					return
			else:
				if len(content) != 3:
					continue
				label, left_feas, right_feas = content
				label = int(label)
				try:
					left_feas = [int(v) for v in left_feas.split(' ')]
					right_feas = [int(v) for v in right_feas.split(' ')]
				except Exception as e:
					print >> sys.stderr, e
					continue
				self.lX.append(left_feas)
				self.rX.append(right_feas)
				self.y.append(label)
			cnt += 1
	def preprocess_data(self, left_vec_size=None, right_vec_size=None):
		'''对数据进行预处理'''
		if left_vec_size == None:
			self.left_vec_size = max([len(ins) for ins in self.lX])
		else:
			self.left_vec_size = left_vec_size
		if right_vec_size == None:
			self.right_vec_size = max([len(ins) for ins in self.rX])
		else:
			self.right_vec_size = right_vec_size
		# padding 0可能存在问题，0是特征不在字典里面的默认值
		self.lX = keras.preprocessing.sequence.pad_sequences(self.lX, value=0, padding='post', maxlen=self.left_vec_size)
		self.rX = keras.preprocessing.sequence.pad_sequences(self.rX, value=0, padding='post', maxlen=self.right_vec_size)
		self.X = tf.concat([self.lX, self.rX], axis=1)

class WSABIE(keras.Model):
	'''
	left_bow_size：左bow的字典大小
	right_bow_size：右bow的字典大小
	left_vec_size：左特征向量维度
	'''
	def __init__(self, left_bow_size, right_bow_size, left_vec_size):
		super(WSABIE, self).__init__(self)
		self.lwn = left_vec_size
		self.left_embedding = layers.Embedding(left_bow_size, 32)
		self.right_embedding = layers.Embedding(right_bow_size, 32)
		self.pooling = keras.layers.GlobalAveragePooling1D()
		# self.left_dense = layers.Dense(16, activation='relu')
		# self.right_dense = layers.Dense(16, activation='relu')
	
	def call(self, inputs, training=None):
		lx = self.left_embedding(inputs[:, :self.lwn])
		rx = self.right_embedding(inputs[:, self.lwn:])
		lx = self.pooling(lx)
		rx = self.pooling(rx)
		# lx = self.left_dense(lx)
		# rx = self.right_dense(rx)
		x = lx * rx
		x = tf.reduce_sum(x, axis=1)
		return x
	
	def left_fea_map(self, inputs):
		x = self.left_embedding(inputs)
		x = self.pooling(x)
		return x

	def right_fea_map(self, inputs):
		x = self.right_embedding(inputs)
		x = self.pooling(x)
		return x

'''trick的实现，可能是训练速度的瓶颈。'''
def pairwise_hinge_loss(out, one_batch):
	loss = tf.constant([], tf.float32)
	pos = out[0]
	neg = out[1:]
	for i in range(0, len(out), one_batch):
		pos = out[i]
		neg = out[i+1:i+one_batch]
		zeros = tf.zeros_like(neg)
		_loss = tf.reduce_max(tf.stack([zeros, 1 - pos + neg], axis=0), axis=0)
		loss = tf.concat([loss, _loss], axis=0)
	return loss

def train(train_filename, val_filename=None, save_model_dir=None, epoch_num=50, alpha=0.001, batch_size=128, verbose=True):
	# 加载训练数据
	train_dl = DataLoader(train_filename)
	train_dl.load_data()
	train_dl.preprocess_data()
	batch_size *= train_dl.one_batch
	if verbose:
		print >> sys.stderr, "train_data: left_bow_size[%d], right_bow_size[%d], left_vec_size[%d], right_vec_size[%d]" % \
							(train_dl.left_bow_size, train_dl.right_bow_size, train_dl.left_vec_size, train_dl.right_vec_size)

	# 加载验证数据
	if val_filename:
		val_dl = DataLoader(train_filename)
		val_dl.load_data()
		val_dl.preprocess_data(train_dl.left_vec_size, train_dl.right_vec_size)
		if verbose:
			print >> sys.stderr, "val_data: left_bow_size[%d], right_bow_size[%d], left_vec_size[%d], right_vec_size[%d]" % \
							(val_dl.left_bow_size, val_dl.right_bow_size, val_dl.left_vec_size, val_dl.right_vec_size)
		if val_dl.left_bow_size != train_dl.left_bow_size or val_dl.right_bow_size != train_dl.right_bow_size:
			val_filename = None

	# 创建模型
	model = WSABIE(train_dl.left_bow_size, train_dl.right_bow_size, train_dl.left_vec_size)
	model.build(input_shape=(None, train_dl.left_vec_size + train_dl.right_vec_size))
	model.summary()
	optimizer = tf.keras.optimizers.Adam(alpha)
	train_loss_results = []
	train_auc_results = []
	# 训练
	for epoch in range(epoch_num):
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_auc = tf.keras.metrics.AUC()
		for step in range(0, len(train_dl.y), batch_size):
			input_data = train_dl.X[step:step+batch_size]
			with tf.GradientTape() as tape:
				out = model(input_data)
				loss = pairwise_hinge_loss(out, train_dl.one_batch)
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			epoch_loss_avg(loss)
			gt = train_dl.y[step:step+batch_size]
			# 必须将out变换到[0, 1]之后才能计算auc，否则会出错
			epoch_auc(gt, tf.nn.sigmoid(out))
		train_loss_results.append(epoch_loss_avg.result())
		train_auc_results.append(epoch_auc.result())
		if verbose:
			print >> sys.stderr, "Epoch {:03d}: Loss: {:.3f}, AUC: {:.3f}".format(epoch, epoch_loss_avg.result(), epoch_auc.result())
		# 验证
		if val_filename and epoch % 1 == 0:
			val_auc = tf.keras.metrics.AUC()
			val_out = model(val_dl.X)
			val_auc(val_dl.y, tf.nn.sigmoid(val_out))
			if verbose:
				print >> sys.stderr, "Epoch {:03d}: Validation AUC: {:.3f}".format(epoch, val_auc.result())
	# 保存模型
	if save_model_dir != None:
		model.save_weights(save_model_dir + 'wsabie_' + time.strftime("%Y%m%d", time.localtime()))
	return model

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "Usage: python %s train_filename [val_filename]" % sys.argv[0]
		sys.exit(0)
	train_filename = sys.argv[1]
	val_filename = None
	if len(sys.argv) >= 3:
		val_filename = sys.argv[2]
	model = train(train_filename, val_filename, epoch_num=10)
