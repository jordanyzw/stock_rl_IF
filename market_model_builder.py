from model_builder import AbstractModelBuilder

class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
		from keras.layers.advanced_activations import LeakyReLU
		from keras.utils.visualize_util import plot
		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]

		for i in xrange(1):
			S = Input(shape=[2, 60, 1])
			inputs.append(S)
			h = Convolution2D(1024, 5, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(256)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

			h = Convolution2D(1024, 10, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(256)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

		m = merge(merges, mode = 'concat', concat_axis = 1)
		m = Dense(64)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(2, activation = 'softmax')(m)
		model = Model(input = inputs, output = V)
		plot(model, to_file='model_pg.png')
		return model

class MarketModelBuilder(AbstractModelBuilder):
	
	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
		from keras.layers.advanced_activations import LeakyReLU
		from keras.utils.visualize_util import plot
		dr_rate = 0.0

		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]
		#channel,row,col
		S = Input(shape=[2, 60, 1])
		inputs.append(S)
		#kernels,kernel width,kernel height
		h = Convolution2D(64, 3, 1, border_mode = 'valid')(S)
		h = LeakyReLU(0.001)(h)
		h = Flatten()(h)
		h = Dense(32)(h)
		h = LeakyReLU(0.001)(h)
		h = Dropout(dr_rate)(h)
		merges.append(h)


		m = merge(merges, mode = 'concat', concat_axis = 1)


		m = Dense(4)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(2, activation = 'linear', init = 'zero')(m)
		model = Model(input = inputs, output = V)
		plot(model, to_file='model_dqn.png')
		return model
