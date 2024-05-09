import gym
import collections
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from cmodels import Discriminator, Qnet, PolicyImgNet
from common_utils import save_model, test_model, show_results
from data.image_dataset import get_dataloaders

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 5000
batch_size = 64
steps_per_episode = 2000
NUM_EPISODES = 50

target_update = 100
theta_update = 100
validation_frequency = 50

pr_auc_history = []
roc_auc_history = []
acc_history = []

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

        state = torch.stack(s_lst).to(device=device)
        state = torch.squeeze(state, dim=1)

        next_state = torch.stack(s_prime_lst).to(device=device)
        next_state = torch.squeeze(next_state, dim=1)

        return (state, torch.tensor(a_lst).to(device=device), torch.tensor(r_lst).to(device=device),
                next_state, torch.tensor(done_mask_lst).to(device=device))
    
    def size(self):
        return len(self.buffer)


def train(qnet, qnet_target, memory, optimizer):
    if memory.size() <= batch_size:
        return

    for i in range(3):
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
    env = ADEnv(dataset=train_dataset)

    # 构造model
    # q_target = Discriminator(in_planes=248832, device=device).to(device=device)
    q = PolicyImgNet().to(device=device)
    q_target = PolicyImgNet().to(device=device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for i_episode in range(NUM_EPISODES):
        epsilon = max(0.01, 0.08 - 0.01*(i_episode/200))  # Linear annealing from 8% to 1%
        state, done = env.reset()

        t_reward = []
        for t in range(steps_per_episode):
            action = q.sample_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward / 100.0, next_state, done_mask))
            state = next_state

            t_reward.append(reward)

            # update the target network
            if t % target_update == 0:
                train(q, q_target, memory, optimizer)
                q_target.load_state_dict(q.state_dict())
            # validation step
            if t % validation_frequency == 0:
                roc, pr, acc = test_model(test_set=test_dataset, policy_model=q_target, device=device)
                pr_auc_history.append(pr)
                roc_auc_history.append(roc)
                acc_history.append(acc)
                # print( "n_episode:{}-{} | epoch_score:{:.3f} | n_buffer:{} | eps:{:.1f}% | acc:{:.3f} | roc:{:.3f} | pr:{:.3f}".format(
                        # i_episode, t, np.mean(t_reward), memory.size(), epsilon * 100, acc, roc, pr))
            if t % theta_update == 0:
                # self.intrinsic_rewards = DQN_iforest(self.x_tensor, self.policy_net)
                pass

        avg_reward = np.mean(t_reward)
        avg_pr = np.mean(pr_auc_history)
        avg_roc = np.mean(roc_auc_history)
        avg_acc = np.mean(acc_history)
        print("n_episode:{} | epoch_score:{:.3f} | n_buffer:{} | eps:{:.1f}% | acc:{:.3f} | roc:{:.3f} | pr:{:.3f}".format(
            i_episode, avg_reward, memory.size(), epsilon * 100, avg_acc, avg_roc, avg_pr))

    env.close()

    # save model
    model_name = 'model_{0}_{1}_.pth'.format(NUM_EPISODES, steps_per_episode)
    save_model(model=q_target, model_name=model_name)

    # show pic
    show_results(t_reward, pr_auc_history, roc_auc_history, acc_history)


if __name__ == '__main__':
    main()
