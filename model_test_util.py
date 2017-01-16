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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def plot_profit(cum_profit, target_close, pre_action, figlabel):
	dates = sorted(cum_profit.keys())
	reward = []
	long_action_close = []
	long_action_date = []
	short_action_close = []
	short_action_date = []
	close_p = []
	for dt in dates:

		reward.append(cum_profit[dt])
		if pre_action[dt] == "LONG":
			long_action_close.append(target_close[dt])
			long_action_date.append(pd.to_datetime(dt))
		else:
			short_action_close.append(target_close[dt])
			short_action_date.append(pd.to_datetime(dt))
		close_p.append(target_close[dt])
	dates = pd.to_datetime(dates)
	daysFmt  = DateFormatter("%Y-%m-%d")
	fig = plt.figure()  
	ax1 = fig.add_subplot(211)  
	ax1.plot_date(dates,reward, '-')  
	# format the ticks  
	ax1.xaxis.set_major_formatter(daysFmt)  
	ax1.autoscale_view()  
	# format the coords message box  
	def price(x): 
	    return '$%1.5f'%x 
	ax1.fmt_xdata = DateFormatter('%Y-%m-%d')  
	ax1.fmt_ydata = price  
	ax1.grid(True)

	ax2 = fig.add_subplot(212)
	ax2.plot_date(long_action_date,long_action_close,'r.',lw=1)
	ax2.plot_date(short_action_date,short_action_close,'g.',lw=1)
	ax2.plot_date(dates,close_p,'-')
	ax2.xaxis.set_major_formatter(daysFmt)  
	ax2.autoscale_view()  
	ax2.fmt_xdata = DateFormatter('%Y-%m-%d')  
	ax2.fmt_ydata = price  
	ax2.grid(True)


	fig.autofmt_xdate()  
	plt.title(figlabel + '_cum_profit')
	plt.savefig(figlabel + "_cum_profit.png")
	del fig

def get_test_performance(epoch, modelFilename = 'model.h5', model = None):
	
	import codecs
	codeListFilename = 'input_code.csv'
	
	codeMap = {}
	f = codecs.open(codeListFilename, "r", "utf-8")
	for line in f:
		if line.strip() != "":
			tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
			codeMap[tokens[0]] = tokens[1]

	f.close()
	env = MarketEnv(dir_path = "./If_index/", target_codes = \
		codeMap.keys(),  start_date = "2015-05-29", \
		end_date = "2016-08-25", sudden_death = -1.0)
	target_close = env.get_close()

	from keras.optimizers import SGD
	if(model == None and modelFilename == 'model_dqn.h5'):
		model = MarketModelBuilder(modelFilename).getModel()
	elif (model == None and modelFilename == 'model_pg.h5'):
		model = MarketPolicyGradientModelBuilder(modelFilename).getModel()

	loss = 0.
	game_over = False
	# get initial input
	input_t = env.reset()
	cumReward = 0
	cum_profit = {}
	pre_action = {}
	while not game_over:
		input_tm1 = input_t
		q = model.predict(input_tm1)
		action = np.argmax(q[0])
		input_t, reward, game_over, info = env.step(action)
		cumReward += reward
		cum_profit[info["dt"]] = cumReward
		if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
			pre_action[info['dt']] = env.actions[action]
			color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE	
			print "%s:\t%s\t%d\t%.2f\t%.2f\t" % (info["dt"], color + env.actions[action] + \
				bcolors.ENDC, info['correct_action'], cumReward, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i)\
					for l, i in zip(env.actions, q[0].tolist())]) )

	print len(cum_profit.keys()),len(target_close)
	plot_profit(cum_profit, target_close, pre_action, "test_" + str(epoch))
	return cum_profit,target_close
