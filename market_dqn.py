import numpy as np
import random
from market_env import MarketEnv
from market_model_builder import MarketModelBuilder
from collections import deque
import matplotlib.pyplot as plt
import  matplotlib.dates as date
from matplotlib.dates import DateFormatter
import sys
import pandas as pd
sys.setdefaultencoding("utf-8")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import model_test_util as test_util

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

class ExperienceReplay(object):
	def __init__(self, max_memory=100, discount=.9):
		self.max_memory = max_memory
		self.memory = deque()
		self.discount = discount

	def remember(self, states, game_over):
		if len(self.memory) > self.max_memory:
			self.memory.popleft()
		self.memory.append([states, game_over])

	def get_batch(self, model, batch_size=10):
		len_memory = len(self.memory)
		num_actions = model.output_shape[-1]
		inputs = []
		#2
		dim = len(self.memory[0][0][0])
		
		for i in xrange(dim):
			inputs.append([])

		targets = np.zeros((min(len_memory, batch_size), num_actions))
		for i,ele in enumerate(random.sample(self.memory, batch_size)):
			state_t, action_t, reward_t, state_tp1 = ele[0]
			game_over = ele[1]
			for j in xrange(dim):
				inputs[j].append(state_t[j][0])

			#inputs.append(state_t)
			# There should be no target values for actions not taken.
			# Thou shalt not correct actions not taken #deep
			targets[i] = model.predict(state_t)[0]
			Q_sa = np.max(model.predict(state_tp1)[0])
			if game_over:  # if game_over is True
				targets[i, action_t] = reward_t
			else:
				# reward_t + gamma * max_a' Q(s', a')
				targets[i, action_t] = reward_t + self.discount * Q_sa
		
		#inputs = np.array(inputs)
		inputs = [np.array(inputs[i]) for i in xrange(dim)]
		
		return inputs, targets

def copy_model_weight(q_estimator, target_estimator):
	q_weights = q_estimator.get_weights()
	target_weights = target_estimator.get_weights()
	for i in xrange(len(q_weights)):
		target_weights[i] = q_weights[i]
	target_estimator.set_weights(target_weights)
	return target_estimator


def train():
	import sys
	import codecs

	codeListFilename = sys.argv[1]
	modelFilename = sys.argv[2] if len(sys.argv) > 2 else None

	codeMap = {}
	f = codecs.open(codeListFilename, "r", "utf-8")

	for line in f:
		if line.strip() != "":
			tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
			codeMap[tokens[0]] = tokens[1]

	f.close()

	env = MarketEnv(dir_path = "./If_index/", target_codes = \
		codeMap.keys(),  start_date = "2013-05-26", \
		end_date = "2015-08-25", sudden_death = -1.0)
	target_close = env.get_close()
	# parameters
	epsilon = .45  # exploration
	min_epsilon = 0.1
	epoch = 1000
	max_memory = 5000
	batch_size = 256
	discount = 0.9
	update_target_weight_step = 10
	from keras.optimizers import SGD
	model = MarketModelBuilder(modelFilename).getModel()
	target_model = MarketModelBuilder(modelFilename).getModel()
	

	sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(loss='mse', optimizer='rmsprop')
	target_model.compile(loss='mse', optimizer='rmsprop')



	# Initialize experience replay object
	exp_replay = ExperienceReplay(max_memory = max_memory, discount = discount)

	# Train
	win_cnt = 0
	for e in range(epoch):
		loss = 0.
		env.reset()
		game_over = False
		# get initial input
		input_t = env.reset()
		cumReward = 0
		cum_profit = {}
		pre_action = {}

		iter_cnt = 0
		while not game_over:
			input_tm1 = input_t
			isRandom = False

			# get next action
			if np.random.rand() <= epsilon:
				action = np.random.randint(0, env.action_space.n, size=1)[0]
				isRandom = True
			else:
				q = model.predict(input_tm1)
				action = np.argmax(q[0])
				if np.nan in q:
					print "OCCUR NaN!!!"
					exit()

			# apply action, get rewards and new state
			input_t, reward, game_over, info = env.step(action)
			cumReward += reward
			cum_profit[info["dt"]] = cumReward
			if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
				color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
				pre_action[info['dt']] = env.actions[action]
				if isRandom:
					color = bcolors.WARNING if env.actions[action] == "LONG" else bcolors.OKGREEN
				print "%s:\t%s\t%d\t%.5f\t%.2f\t" % (info["dt"], color + env.actions[action] + \
					bcolors.ENDC, info['correct_action'], cumReward, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i)\
					 for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else "")

			# store experience
			exp_replay.remember([input_tm1, action, reward, input_t], game_over)

			# adapt model
			if(len(exp_replay.memory) >= batch_size):
				inputs, targets = exp_replay.get_batch(target_model, batch_size=batch_size)
				loss += model.train_on_batch(inputs, targets)
			if(iter_cnt % update_target_weight_step == 0): # update target estimator every 5 step
				print "update target model weights"
				target_model = copy_model_weight(model, target_model)
			iter_cnt += 1

		if cumReward > 0 and game_over:
			win_cnt += 1
		test_util.plot_profit(cum_profit, target_close, pre_action, "dqn_train_"+str(e))
		test_util.get_test_performance(e,'model_dqn.h5', model)

		print("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Epsilon {:.4f}".format(e, epoch, loss, win_cnt, epsilon))
		# Save trained model weights and architecture, this will be used by the visualization code
		if(e / 5 == 0 and e != 0):
			model.save_weights("model.h5" if modelFilename == None else modelFilename, overwrite=True)
		epsilon = max(min_epsilon, epsilon * 0.99)

if __name__ == "__main__":
	#test_util.get_test_performance(0,'model.h5')
	train()

	
