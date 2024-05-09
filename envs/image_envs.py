import random
from collections import deque

import gym
import numpy as np
from gym import spaces
from data.image_dataset import transDataset2Ndarray, transDataset2Tensor


class ADEnv(gym.Env):
    def __init__(self, dataset=None, sampling_Du=1000, prob_au=0.5, label_normal=0, label_anomaly=1):
        super().__init__()

        # hyperparameters:
        self.num_S = sampling_Du
        self.normal = label_normal
        self.anomaly = label_anomaly
        self.prob = prob_au

        index_anomaly, index_normal, x = transDataset2Tensor(dataset)
        self.x = x
        self.index_n = index_normal
        self.index_a = index_anomaly
        self.random_range = len(dataset)

        all_element_index_list = [x for x in range(self.random_range)]
        random.shuffle(all_element_index_list)
        self.all_elements_deque = deque(all_element_index_list)

        print('_____缺陷样本数目:', len(self.index_a))
        print('_____Normal样本数目:', len(self.index_n))
        print('_____百分比:', len(self.index_a) / self.random_range)

        # observation space:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 250, 250), dtype=np.uint8)
        # self.observation_space = spaces.Discrete(self.random_range)

        # action space: 0 or 1
        self.action_space = spaces.Discrete(2)

        # initial state
        self.state = None
        self.sindex = 0
        self.done = False

    def generater_anomaly(self, *args, **kwargs):
        index = np.random.choice(self.index_a)
        return index

    def generater_normal(self, *args, **kwargs):
        index = np.random.choice(self.index_n)
        return index

    def generate_u(self, action, s_t):
        # sampling function for D_u
        S = np.random.choice(self.index_n, self.num_S)

        # calculate distance in the space of last hidden layer of DQN
        all_x = self.x[np.append(S, s_t)]

        all_dqn_s = self.DQN.get_latent(all_x)
        all_dqn_s = all_dqn_s.cpu().detach().numpy()
        dqn_s = all_dqn_s[:-1]
        dqn_st = all_dqn_s[-1]

        dist = np.linalg.norm(dqn_s-dqn_st, axis=1)

        if action == 1:
            loc = np.argmin(dist)
        elif action == 0:
            loc = np.argmax(dist)
        index = S[loc]
        return index


    def reward_h(self, action, s_t):
        # Anomaly-biased External Handcrafted Reward Function h
        if (action == 1) & (s_t in self.index_a):
            return 10
        elif (action == 0) & (s_t in self.index_n):
            return 3
        elif (action == 0) & (s_t in self.index_a):
            return -1
        elif (action == 1) & (s_t in self.index_n):
            return -1
        return 0

    def step(self, action):
        self.sindex = int(self.sindex)
        reward = self.reward_h(action, self.sindex)

        if len(self.all_elements_deque) == 0:
            a_param = [self.generater_anomaly, self.generater_normal]
            g = np.random.choice(a_param, p=[0.7, 0.3])
            self.sindex = g(action, self.sindex)
            self.state = self.x[int(self.sindex)][1]
        else:
            self.sindex = self.all_elements_deque.popleft()
            self.state = self.x[int(self.sindex)][1]

        self.done = False

        info = {"State t": self.state, "Action t": action, "State_index": self.sindex}
        return self.state, reward, self.done, self.sindex, info

    def reset(self, seed=None, options=None):
        self.done = False
        self.sindex = 0
        self.state = self.x[self.sindex][1]
        return self.state, self.done

    def render(self):
        pass

    def close(self):
        pass
