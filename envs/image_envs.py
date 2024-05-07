import gym
import numpy as np
from gym import spaces
from data.image_dataset import transDataset2Ndarray, transDataset2Tensor


class ADEnv(gym.Env):
    def __init__(self, dataset=None, sampling_Du=1000, prob_au=0.5, label_normal=0, label_anomaly=1, ENV_STPES=100):
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
        self.ENV_STPES = ENV_STPES + self.random_range

        print('_____缺陷样本数目:', len(self.index_a))
        print('_____Normal样本数目:', len(self.index_n))

        # observation space:
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 288, 288), dtype=np.uint8)
        self.observation_space = spaces.Discrete(self.random_range)

        # action space: 0 or 1
        self.action_space = spaces.Discrete(2)

        # initial state
        self.state = None
        self.state_index = 0
        self.done = False
        self.counts = 0

    def generater_anomaly(self, *args, **kwargs):
        # sampling function for D_a
        index = np.random.choice(self.index_a)
        return index

    def generater_normal(self, *args, **kwargs):
        # sampling function for D_a
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
            return 3
        elif (action == 0) & (s_t in self.index_n):
            return 3
        elif (action == 0) & (s_t in self.index_a):
            return -1
        elif (action == 1) & (s_t in self.index_n):
            return -1
        return 0

    def step(self, action):
        self.state_index = int(self.state_index)
        reward = self.reward_h(action, self.state_index)
        self.state_index = self.state_index + 1
        self.counts = self.counts + 1

        if self.state_index > self.random_range - 1:
            a_param = [self.generater_anomaly, self.generater_normal]
            g = np.random.choice(a_param, p=[0.6, 0.4])
            self.state_index = g(action, self.state_index)

        self.state = self.x[int(self.state_index)][1]  # 新的state是一个int值，相当于是新的特征向量Tensor的Index
        if self.counts > self.ENV_STPES:
            self.done = True

        # info
        info = {"State t": self.state, "Action t": action, "State_index": self.state_index}

        return self.state, reward, self.done, self.state_index, info

    def reset(self, seed=None, options=None):
        self.done = False
        self.counts = 0

        if self.state_index > 0:
            # the first observation is uniformly sampled from the D_u
            self.state_index = np.random.choice(self.random_range)
            self.state_index = int(self.state_index)

        self.state = self.x[self.state_index][1]
        return self.state, self.done

    def render(self):
        pass

    def close(self):
        pass
