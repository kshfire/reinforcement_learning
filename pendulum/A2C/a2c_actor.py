# A2C Actor (A2C 액터 신경망을 설계한 파일)
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda

import tensorflow as tf

class Actor(object):
    '''
    Actor Network for A2C (A2C 액터 신경망)
    '''
    def __init__(self, state_dim, action_dim, action_bound, learning_rate):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate

        # 표준편차의 최소값과 최대값 설정
        self.std_bound = [1e-2, 1.0] # std bound

        # 액터 신경망 생성
        self.model = self.build_network()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ## actor network
    def build_network(self):
        state_input = Input(shape=(self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out_mu = Dense(self.action_dim, activation='tanh')(h3)
        std_output = Dense(self.action_dim, activation='softplus')(h3)

        # Scale output to [-action_bound, action_bound]
        mu_output = Lambda(lambda x: x*self.action_bound)(out_mu)
        model = Model(inputs=state_input, outputs=[mu_output, std_output])
        model.summary()
        return model

    ## log policy pdf
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -(0.5 * (action - mu) ** 2 / var + 0.5 * tf.math.log(var * 2 * np.pi)) # (4.26)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)


    ## actor policy
    def get_action(self, state):
        # type of action in env is numpy array
        # np.reshape(state, [1, self.state_dim]) : shape (state_dim,) -> shape (1, state_dim)
        # why [0]?  shape (1, action_dim) -> (action_dim,)
        #mu_a, std_a = self.model.predict(np.reshape(state, [1, self.state_dim]))
        mu_a, std_a = self.model.predict(np.array([state]))
        mu_a = mu_a[0]
        std_a = std_a[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return action

    ## actor prediction
    def predict(self, state):
        mu_a, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
        return mu_a[0]

    # train the actor network
    def train(self, states, actions, advantages):

        with tf.GradientTape() as tape:
            # policy pdf
            mu_a, std_a = self.model(np.array(states))
            log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

            ## A2C 알고리즘 : 2.6.
            # loss functions and its gradient
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)

        dj_dtheta = tape.gradient(loss, self.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(dj_dtheta, self.model.trainable_variables))

    ## save actor weights
    def save_weights(self, path):
        self.model.save_weights(path)

    ## load actor wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_actor.h5')