# 수학으로 푸렁보는 강화학습 원리와 알고리즘
# A2C main
# A2C (Advantage Actor-Critic)

import gym
from a2c_agent import A2Cagent

import sys, os      # 파일 검사

def main():

    max_episode_num = 1000      # 최대 에피소드 설정
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)    # 환경으로 OpenAI Gym 의 pendulum-v0 설정
    agent = A2Cagent(env)       # A2C 에이전트 객체

    # 저장된 모델이 있는지 검사하여, 만약 저장된 모델이 있다면 불러옴
    if os.path.exists('./save_weights/pendulum_actor.h5'):
        # load model (actor & critic)
        agent.actor.load_weights('./save_weights/')
        agent.critic.load_weights('./save_weights/')

        print('load existing models')

        '''
        print('load existing models')
        #print('actor model parameters \n', agent.actor.model.trainable_variables)
        print('actor model parameters \n', agent.actor.model.trainable_weights)
        print('\n')
        #print('critic model parameters \n', agent.critic.model.trainable_variables)
        print('critic model parameters \n', agent.critic.model.trainable_weights)
        '''

    # 학습 진행
    agent.train(max_episode_num)
    # 학습 결과 도시
    agent.plot_result()

# 현재 스크립트 파일이 시작점인지, 모듈인지 판단: 시작점이면, main() 함수 호출
if __name__=="__main__":
    main()