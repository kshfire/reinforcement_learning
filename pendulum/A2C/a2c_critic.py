# A2C Critic

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class Critic(object):
    """
        Critic Network for A2C: V function approximator
    """
    def __init__(self, state_dim, action_dim, learning_rate):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # create and compile critic network
        #self.model, self.states = self.build_network()
        self.model = self.build_network()  # states 쓰는 곳이 없는데 ???

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')  # 모델의 학습방법 정의

    ## critic network
    def build_network(self):
        state_input = Input(shape=(self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        v_output = Dense(1, activation='linear')(h3)
        model = Model(inputs=state_input, outputs=v_output)
        model.summary()
        #return model, state_input
        return model


    ## single gradient update on a single batch data
    def train_on_batch(self, states, td_targets):
        return self.model.train_on_batch(x=np.array(states), y=np.array(td_targets))

    ## critic prediction
    def predict(self, state):
        return self.model.predict(x=np.array([state]))[0][0]

    ## save critic weights
    def save_weights(self, path):
        self.model.save_weights(path)


    ## load critic wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_critic.h5')