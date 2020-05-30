# 수학으로 푸렁보는 강화학습 원리와 알고리즘
# A2C main
# A2C (Advantage Actor-Critic)

import gym
from a2c_agent import A2Cagent

def main():

    max_episode_num = 1000      # 최대 에피소드 설정
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = A2Cagent(env)

    agent.train(max_episode_num)

    agent.plot_result()



if __name__=="__main__":
    main()