import tensorflow as tf
import numpy as np

class ObjectiveNets(object):
	""" MLP classifier for cifar-10
		:param input_dim: number of data attribute
		:param hidden_dim: dimensions of hidden layers
		:param num_class: the number of classes 

		:instance variable self.summary: a list of summary nodes of all weights
		:instance variable self.weights: a dict contain all the weight matrix 

		:return construct Objective net
	"""
	def __init__(self,input_dim=3072,hidden_dims=[100,50],num_classes=10):
		
		weights = dict()
		self.num_layers = len(hidden_dims) + 1
		dims = [input_dim] + hidden_dims
        
		self.summary = []
		self.updates = []
		# Define weights node
		with tf.name_scope('ObjectiveNets'):
			with tf.variable_scope('ObjNets',reuse=tf.AUTO_REUSE):
				for i in range(len(hidden_dims)):
					weight_name = 'weight_{}'.format(i)
					W = tf.get_variable(name=weight_name,shape=[dims[i]+1,dims[i+1]],dtype=tf.float32,
										initializer=tf.random_normal_initializer(stddev=0.02))
					weights[weight_name] = W

					self.summary.append(tf.summary.histogram(weight_name,W))

				weight_name = 'weight_{}'.format(len(hidden_dims))
				W = tf.get_variable(name=weight_name,shape=[dims[-1]+1,num_classes],dtype=tf.float32,
										initializer=tf.random_normal_initializer(stddev=0.02))
				weights[weight_name] = W
				self.summary.append(tf.summary.histogram(weight_name,W))

		self.weights = weights



	def forward(self,X):
		''' 
			:return out:forward result of Objective net
		'''
		num_layers = self.num_layers
		weights = self.weights
		with tf.name_scope('ObjNets_forward'):
			x = tf.concat([X,tf.ones([tf.shape(X)[0],1])],axis=1)

			for i in range(num_layers - 1):
				w = weights['weight_{}'.format(i)]
				temp = tf.matmul(x,w)
				temp = tf.nn.sigmoid(temp)
				x = tf.concat([temp,tf.ones([tf.shape(temp)[0],1])],axis=1)

			w = weights['weight_{}'.format(num_layers-1)]
			out = tf.matmul(x,w)

		return out


	def loss(self,X,y,l2norm=None):
		'''
			:return loss: cross entropy loss (with or without regularization)
		'''
		weights = self.weights
		with tf.name_scope('ObjNets_loss'):
			labels = tf.one_hot(y,10)
			out = self.forward(X)

			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=out))

		grads = dict()
		with tf.name_scope('ObjNets_grads'):
			for i in range(len(weights)):
				grad_name = 'grad_{}'.format(i)
				G = tf.gradients(loss,weights['weight_{}'.format(i)])
				grads[grad_name] = G[0]

				self.summary.append(tf.summary.histogram(grad_name,G[0]))

		self.grads = grads
        
		if l2norm !=None:
			l2_loss = tf.reduce_sum([tf.norm(w) for w in weights.values()])
			loss = tf.add(loss,l2_loss * l2norm) 
    
		self.summary.append(tf.summary.scalar('ObjNets_Loss',loss))

		return loss

	def update(self,learning_rate):
		''' gradient descent method '''
		weights = self.weights
		grads = self.grads
		with tf.name_scope('ObjNets_update'):

			with tf.variable_scope('ObjNets',reuse=tf.AUTO_REUSE):

				for i in range(len(grads)):
					weight_name = 'weight_{}'.format(i)
					grad_name = 'grad_{}'.format(i)
					w_new = weights[weight_name] - learning_rate*grads[grad_name]

					w = tf.get_variable(name=weight_name)
					update_w = tf.assign(w,w_new)
					self.updates.append(update_w)

	def evaluate(self,X_test,y_test):
		out = self.forward(X_test)
		pred = tf.argmax(out,axis=1)
		error_num = tf.count_nonzero(pred-y_test)/tf.cast(tf.shape(y_test)[0],tf.int64)
		val_acc = 100 - error_num*100
		return val_acc,error_num