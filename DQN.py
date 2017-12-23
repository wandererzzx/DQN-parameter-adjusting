import tensorflow as tf
import numpy as np

class DQN(object):
	""" Deep Q-Net 
		:param input_dim: number of state variables
		:param hidden_dim: dimensions of hidden layers
		:param num_class: the number of actions we can choose

		:instance variable self.summary: a list of summary nodes of all weights
		:instance variable self.weights: a dict contain all the weight matrix 

		:return construct DQN
	"""
	def __init__(self,input_dim=6,hidden_dims=[32,16],num_classes=2):
		
		weights = dict()
		self.num_layers = len(hidden_dims) + 1
		dims = [input_dim] + hidden_dims
 
		self.summary = []
		self.updates = []
		# Define weights node
		with tf.name_scope('Deep_Q_Net'):
			with tf.variable_scope('DQN',reuse=tf.AUTO_REUSE):
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
			:param X: experience from exp memory
			
			:return out: DQN forward result (q values for each action)

			Use bias trick for convenient control of weights and biases
		'''
		num_layers = self.num_layers
		weights = self.weights

		with tf.name_scope('DQN_forward'):
			x = tf.concat([X,tf.ones([tf.shape(X)[0],1])],axis=1)

			for i in range(num_layers - 1):
				w = weights['weight_{}'.format(i)]
				temp = tf.matmul(x,w)
				temp = tf.nn.sigmoid(temp)
				x = tf.concat([temp,tf.ones([tf.shape(temp)[0],1])],axis=1)

			w = weights['weight_{}'.format(num_layers-1)]
			out = tf.matmul(x,w)

		return out



	def loss(self,X,y):
		'''
			:param X,y : experience from exp memory
			
			:instance variable self.grads: gradients for weights

			:return loss: l2 loss 
		'''
		weights = self.weights
		with tf.name_scope('DQN_loss'):
			labels = y
			current_states = X

			predict = self.forward(current_states)
			loss = tf.nn.l2_loss(predict-labels)

		grads = dict()
		with tf.name_scope('DQN_grads'):
			for i in range(len(weights)):
				grad_name = 'grad_{}'.format(i)
				G = tf.gradients(loss,weights['weight_{}'.format(i)])
				grads[grad_name] = G[0]

				self.summary.append(tf.summary.histogram(grad_name,G[0]))

		self.grads = grads

		self.summary.append(tf.summary.scalar('DQN_loss',loss))
		return loss

	def update(self,learning_rate=0.025):
		'''
			:gradient descent method
		'''
		weights = self.weights
		grads = self.grads
		with tf.name_scope('DQN_update'):

			with tf.variable_scope('DQN',reuse=tf.AUTO_REUSE):

				for i in range(len(grads)):
					weight_name = 'weight_{}'.format(i)
					grad_name = 'grad_{}'.format(i)
					w_new = weights[weight_name] - learning_rate*grads[grad_name]

					w = tf.get_variable(name=weight_name)
					update_w = tf.assign(w,w_new)
					self.updates.append(update_w)
