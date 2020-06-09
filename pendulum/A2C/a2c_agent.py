# A2C Agent for training and evaluation

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from a2c_actor import Actor
from a2c_critic import Critic

class A2Cagent(object):         # object: python 2 의 old class 와 호환을 위해서.. python3 만 쓴다면, (..) 생략 가능
    def __init__(self, env):    # 클래스 초기화 메서드

        # hyper parameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # 환경
        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound (행동의 최대 크기)
        self.action_bound = env.action_space.high[0]

        ## A2C 알고리즘 1. critic 과 actor 신경망의 파라메터 phi (critic 신경망) 와 theta (actor 신경망)를 초기화한다.
        # create actor and critic networks
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE);

        # save the results (에피소드에서 얻은 총 보상값을 저장하기 위한 변수)
        self.save_epi_reward = [];

    ## computing Advantages and targets: y(k) = r(k) + gamma * V(s_k + 1), A(s_k, a_k) = y_k - V(s_k)
    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = reward;
            advantage = y_k - v_value;
        else:
            y_k = reward + self.GAMMA * next_v_value;
            advantage = y_k - v_value;
        return advantage, y_k


    ## train the agent
    def train(self, max_episode_num):
        ## A2C 알고리즘 2. Repeat
        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):
            # initialize batch
            # 상태변수, 행동, 시간차 타겟, 어드밴티지를 저장할 배치를 초기화한다.
            states, actions, td_targets, advantages = [], [], [], []
            # reset episode (에피소드 초기화)
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state x0
            state = self.env.reset()    # shape of state from gym (3,)

            ## A2C 알고리즘 2. 내부 Repeat
            while not done:
                # visualize the environment
                # self.env.render()

                ## A2C 알고리즘 2.1.1. 정책 u_{i} ~ pi_{theta}(u_{i}|x_{i}) 으로 행동을 확률적으로 선택한다.
                # pick an action (shape of gym action = (action_dim,) ) (행동 추출)
                action = self.actor.get_action(state)
                # clip continuous action to be within action_bound [-2, 2]
                action = np.clip(action, -self.action_bound, self.action_bound)

                ## A2C 알고리즘 2.1.2. u_{i} 를 실행해 보상 r(x_{i}, u_{i})과 다음 상태변수 x_{i+1}을 측정한다.
                # observe reward, new_state, shape of output of gym (state_dim,)
                next_state, reward, done, _ = self.env.step(action)

                # compute next v_value
                v_value = self.critic.predict(state)
                next_v_value = self.critic.predict(next_state)

                # compute advantage and TD target
                train_reward = (reward + 8) / 8  # <-- normalization
                advantage, y_i = self.advantage_td_target(train_reward, v_value, next_v_value, done)

                # append to the batch (배치에 저장)
                states.append(state)
                actions.append(action)
                td_targets.append(y_i)
                advantages.append(advantage)

                # if batch is full, start to train networks on batch
                if len(states) == self.BATCH_SIZE:
                    ## A2C 2.5. critic 신경망 업데이트 (학습) TD target 에 대한 regression ??
                    # train critic
                    self.critic.train_on_batch(states, td_targets)
                    ## A2C 2.6. actor 신경망 업데이트 (학습)
                    # train actor
                    self.actor.train(states, actions, advantages)

                    # clear the batch
                    states, actions, td_targets, advantages = [], [], [], []

                # update current state
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## save weights every episode
            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()