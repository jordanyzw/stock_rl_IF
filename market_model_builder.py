from model_builder import AbstractModelBuilder
from keras.models import Model
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.visualize_util import plot


class MarketModelBuilder(AbstractModelBuilder):
	def buildModel(self):

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

class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

	def buildModel(self):

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

class CriticNetwork(AbstractModelBuilder):
    def __init__(self, sess,  action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)     

        #Now create the model
        self.model, self.action, self.state = self.buildModel()  
        self.target_model, self.target_action, self.target_state = self.buildModel()  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def buildModel(self):
		print("build the critic model")
		# the B and S is the RL state
		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)
		#state_inputs = [B]
		merges = [b]
		#channel,row,col
		S = Input(shape=[2, 60, 1])
		#state_inputs.append(S)
		h = Convolution2D(64, 3, 1, border_mode = 'valid')(S)
		h = LeakyReLU(0.001)(h)
		h = Flatten()(h)
		h = Dense(32)(h)
		h = LeakyReLU(0.001)(h)
		h = Dropout(dr_rate)(h)
		merges.append(h)
		#the action of Q(s,a)
		A = Input(shape=[self.action_dim],name='Q_action')
		a1 = Dense(5, activation='linear')(A)
		merges.append(a1)
		m = merge(merges, mode = 'concat', concat_axis = 1)
		m = Dense(4)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(self.action_dim, activation = 'linear', init = 'zero')(m)
		model = Model(input=[B,S,A],output=V)
		adam = Adam(lr=self.LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)
		return model, A, [B,S] 

class ActorNetwork(AbstractModelBuilder):
    def __init__(self, sess, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
       
        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.buildModel()   
        self.target_model = self.buildModel()
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + \
            (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Convolution2D, MaxPooling2D, \
		Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
		from keras.layers.advanced_activations import LeakyReLU
		from keras.utils.visualize_util import plot
		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		#inputs = [B]
		merges = [b]

		for i in xrange(1):
			S = Input(shape=[2, 60, 1])
			#inputs.append(S)
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
		model = Model(input = [B,S], output = V)
		plot(model, to_file='model_actor.png')
		#return model,model weights, RL state
		return model, model.trainable_weights, [B,S]

