from random import random
import numpy as np
import math
import gym
from gym import spaces
import localfile_data_util as file_utils

class MarketEnv(gym.Env):

	PENALTY = 1 #0.999756079

	def __init__(self, dir_path, target_codes, start_date, end_date,\
	 scope = 60, sudden_death = -1., cumulative_reward = False):
		self.startDate = start_date
		self.endDate = end_date
		self.scope = scope
		self.sudden_death = sudden_death
		self.cumulative_reward = cumulative_reward
		self.correct_action = []
		self.inputCodes = []
		self.targetCodes = []
		self.dataMap = {}
		self.target_close = {}
		for code in (target_codes):
			file_name = dir_path +  code + ".csv"
			print file_name
			data = {}
			lastClose = 0
			lastVolume = 0
			try:
				quotes = file_utils.read_local_file(file_name)
				dates_list = quotes['date']
				open_list = np.array(quotes['open'])
				close_list = np.array(quotes['close'])
				high_list = np.array(quotes['high'])
				low_list = np.array(quotes['low'])
				volume_list = np.array(quotes['volume'])
				totallen = len(open_list)
				for index in range(totallen):
					dt = dates_list[index]
					highprice = high_list[index]
					lowprice = low_list[index]
					closeprice = close_list[index]
					volume = volume_list[index]
					try:
						if dt >= start_date and dt <= end_date:						
							highprice = float(highprice) if highprice != "" else float(closeprice)
							lowprice = float(lowprice) if lowprice != "" else float(closeprice)
							closeprice = float(closeprice)
							volume = int(volume)

							self.target_close[dt] = closeprice
							if lastClose > 0 and closeprice > 0 and lastVolume > 0:
								close_ = (closeprice - lastClose) / lastClose
								if close_ < 0:
									self.correct_action.append(-1)
								else:
									self.correct_action.append(1)
								high_ = (highprice - closeprice) / closeprice
								low_ = (lowprice - closeprice) / closeprice
								volume_ = (volume - lastVolume) / lastVolume
								data[dt] = (high_, low_, close_, volume_)

							lastClose = closeprice
							lastVolume = volume
					except Exception, e:
						print e
				
			except Exception, e:
				print e

			if len(data.keys()) > scope:
				self.dataMap[code] = data
				if code in target_codes:
					self.targetCodes.append(code)

		
		self.actions = [
			"LONG",
			"SHORT",
		]

		self.action_space = spaces.Discrete(len(self.actions))
		#scope = 60 
		self.observation_space = spaces.Box(np.ones(scope ) * -1, np.ones(scope *  1))

		self.reset()
		self._seed()

	def get_close(self):
		return self.target_close
 

	def _step(self, action):
		if self.done:
			return self.state, self.reward, self.done, {}

		self.reward = 0
		if self.actions[action] == "LONG":
			if sum(self.boughts) < 0:#if previously took short action
				
				for b in self.boughts:
					self.reward += -(b + 1)
				if self.cumulative_reward:#False
					self.reward = self.reward / max(1, len(self.boughts))

				if self.sudden_death * len(self.boughts) > self.reward:
					print("done true",self.reward)
					self.done = True

				self.boughts = []

			self.boughts.append(1.0)
		elif self.actions[action] == "SHORT":
			if sum(self.boughts) > 0:#if previously took long action
				
				for b in self.boughts:
					self.reward += b - 1
				if self.cumulative_reward:#False
					self.reward = self.reward / max(1, len(self.boughts))

				if self.sudden_death * len(self.boughts) > self.reward:
					print("done true",self.reward)
					self.done = True

				self.boughts = []

			self.boughts.append(-1.0)
		else:
			pass

		#(close - lastclose) / lastclose = close / lastclose - 1
		vari = self.target[self.targetDates[self.currentTargetIndex]][2]
		self.cum = self.cum * (1 + vari)

		for i in xrange(len(self.boughts)):
			self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * \
			(1 + vari * (-1 if sum(self.boughts) < 0 else 1))

		self.defineState()
		#important to plus one 
		self.currentTargetIndex += 1
		if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[self.currentTargetIndex]:
			self.done = True

		if self.done:
			for b in self.boughts:
				self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
			if self.cumulative_reward:
				self.reward = self.reward / max(1, len(self.boughts))

			self.boughts = []

		return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], \
		"cum": self.cum, "correct_action": self.correct_action[self.currentTargetIndex]}

	def _reset(self):
		self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
		self.target = self.dataMap[self.targetCode]
		self.targetDates = sorted(self.target.keys())
		self.currentTargetIndex = self.scope
		self.boughts = []
		self.cum = 1.

		self.done = False
		self.reward = 0

		self.defineState()

		return self.state

	def _render(self, mode='human', close=False):
		if close:
			return
		return self.state


	def _seed(self):
		return int(random() * 100)

	
	def defineState(self):
		tmpState = []
		#what does budget mean
		budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
		size = math.log(max(1., len(self.boughts)), 100)
		position = 1. if sum(self.boughts) > 0 else 0.
		tmpState.append([[budget, size, position]])

		subjectclose = []
		subjectVolume = []
		for i in xrange(self.scope):
			try:
				subjectclose.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
				subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
			except Exception, e:
				print self.targetCode, self.currentTargetIndex, i, len(self.targetDates)
				self.done = True
		tmpState.append([[subjectclose, subjectVolume]])

		tmpState = [np.array(i) for i in tmpState]
		self.state = tmpState

