import numpy as np 
import queue
class FeatureGenerator(object):
	"""docstring for FeatureGenerator"""
	def __init__(self,M):
		self.loss_memory = []
		self.M = M

	# update the loss memory
	def loss_memory_update(self,loss):
		loss_memory = self.loss_memory
		M = self.M

		loss_memory.sort(reverse=True)

		if len(loss_memory)<M:
			loss_memory.append(loss)
		else:
			max_loss = loss_memory[0]
			if loss < max_loss:
				loss_memory[0] = loss


	# update the ali list, usually called after get current ali
	def ali_update(self,gradients):
		grads = []
		for value in gradients.values():
			grads.extend(value.tolist())
		self.ali = grads


	def encoding(self,loss):
		loss_memory = self.loss_memory
		M = self.M

		if len(loss_memory) < M:
			return None

		max_loss = max(loss_memory)
		min_loss = min(loss_memory)

		if loss <= min_loss:
			return 1
		elif loss > min_loss and loss <= max_loss:
			return 0
		return -1

	def gradient_norm(self,gradients):
		grad_norm = 0.0
		for value in gradients.values():
			grad_norm += np.sum(-value**2)
		
		return grad_norm

	def alignment(self,gradients):
		former = self.ali
		current = []

		if former is None:
			return None

		alignment = []

		for value in gradients.values():
			current.extend(value.tolist())

		for i in range(len(current)):
			temp = np.sign(former[i])*np.sign(current[i])
			alignment.append(np.mean(temp))

		alignment = np.mean(alignment)
		return alignment

	def generate_feature(self,iteration,learning_rate,current_loss,current_gradients):
		feature = np.zeros((6,))
		feature[0] = learning_rate
		feature[1] = current_loss
		feature[2] = self.gradient_norm(current_gradients)
		feature[3] = self.encoding(current_loss)
		feature[4] = iteration
		feature[5] = self.alignment(current_gradients)
		return np.reshape(feature,(1,6))


class Experience_Memory(object):
	def __init__(self,length):
		self.length = length
		self.memory = queue.Queue(length)

	def add_experience(self,current_state_feature,chosen_action,reward,next_state_feature,labels):
		experience = (current_state_feature,chosen_action,reward,next_state_feature,labels)
		if self.memory.qsize() < self.length:
			self.memory.put(experience)
		else:
			self.memory.get()
			self.memory.put(experience)

	def get_experience(self,batch_size):
		experience_array = np.array(list(self.memory.queue))
		choices = np.random.choice(len(experience_array),batch_size)
		ex_choices = experience_array[choices]
		result = np.reshape(ex_choices,[batch_size,-1])
		return result

def get_csf_from_experience(experience_batch):
	features = experience_batch[:,0]
	batch_size = len(features)
	result = features[0]
	for i in range(1,batch_size):
		result = np.vstack([result,features[i]])

	return result

# def get_nsf_from_experience(experience_batch):
# 	features = experience_batch[:,3]
# 	batch_size = len(features)
# 	result = features[0]
# 	for i in range(1,batch_size):
# 		result = np.vstack([result,features[i]])

# 	return result

def get_labels_from_experience(experience_batch):
	features = experience_batch[:,4]
	batch_size = len(features)
	result = features[0]
	for i in range(1,batch_size):
		result = np.vstack([result,features[i]])

	return result


def reward_function(loss,lower_bound=1e-10,c=0.1):
	reward = c*1.0/(loss-lower_bound)
	return reward

# e value should decrease to 0.1
def e_greedy(qvalues,e=1):
	rand = np.random.random()
	if rand < e:
		action = np.random.randint(len(qvalues[0]))
	else:
		action = np.argmax(qvalues[0])
	return action

# current_qvalues size of (1 * num_actions)
def DQN_labels(current_qvalues,action_chosen,reward,next_qvalues=None,gamma=0.99):
	labels = current_qvalues

	if next_qvalues is not None: # judge if terminal state
		labels[0,action_chosen] = reward + gamma*np.max(next_qvalues,axis=1)
	else:
		labels[0,action_chosen] = reward

	return labels