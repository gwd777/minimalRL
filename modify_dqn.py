import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cmodels import Discriminator, Qnet
from data.image_dataset import get_dataloaders

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 100000
batch_size = 16
env_steps = 1000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('______device type_______>', device)

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def batch_sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device=device), torch.tensor(a_lst).to(device=device), \
               torch.tensor(r_lst).to(device=device), torch.tensor(s_prime_lst, dtype=torch.float).to(device=device), \
               torch.tensor(done_mask_lst).to(device=device)
    
    def size(self):
        return len(self.buffer)


def train(qnet, qnet_target, memory, optimizer):
    for i in range(10):
        state, action, reward, next_state, done_mask = memory.batch_sample(batch_size)
        q_out = qnet(state)
        q_a = q_out.gather(1, action)
        max_q_prime = qnet_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

from envs.image_envs import ADEnv
def main():
    # 构造Env环境
    # env = gym.make('CartPole-v1')
    train_dataset, test_dataset = get_dataloaders()
    env = ADEnv(dataset=train_dataset, ENV_STPES=env_steps)

    # 构造model
    q = Discriminator(in_planes=82944, device=device).to(device=device)
    q_target = Discriminator(in_planes=82944, device=device).to(device=device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(500):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        state, done = env.reset()
        while not done:
            action = q.sample_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward / 100.0, next_state, done_mask))
            state = next_state

            score += reward
            if done:
                break
            
        if memory.size() > 20:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
