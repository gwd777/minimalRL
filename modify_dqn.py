import gym
import collections
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from cmodels import Discriminator, Qnet, PolicyImgNet
from common_utils import save_model, test_model, show_results
from data.image_dataset import get_dataloaders

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
epoch_1_steps = 500
num_epochs = 300
print_interval = 15

episodes_total_reward = []
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
    total_loss = 0
    for i in range(10):
        state, action, reward, next_state, done_mask = memory.batch_sample(batch_size)
        q_out = qnet(state)
        q_a = q_out.gather(1, action)

        max_q_prime = qnet_target(next_state).max(1)[0].unsqueeze(1)
        # rt = qnet_target(next_state)
        # rt_max = rt.max(1)
        # rt_0_max = rt_max[0]
        # max_q_prime = rt_0_max.unsqueeze(1)

        target = reward + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
    return total_loss


from envs.image_envs import ADEnv
def main():
    # 构造Env环境
    # env = gym.make('CartPole-v1')
    train_dataset, test_dataset = get_dataloaders()
    env = ADEnv(dataset=train_dataset, ENV_STPES=epoch_1_steps)

    # 构造model
    # q_target = Discriminator(in_planes=248832, device=device).to(device=device)
    q = PolicyImgNet().to(device=device)
    q_target = PolicyImgNet().to(device=device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    score = 0.0
    total_reward = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_epochs):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))  # Linear annealing from 8% to 1%
        state, done = env.reset()
        while not done:
            action = q.sample_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward / 100.0, next_state, done_mask))
            state = next_state

            score += reward
            total_reward += reward
            if done:
                break
            
        if memory.size() > 50:
            epoch_loss = train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            roc, pr, acc = test_model(test_set=test_dataset, policy_model=q_target, device=device)
            print("n_episode:{} | epoch_score:{:.3f} | n_buffer:{} | eps:{:.1f}% | total_reward:{:.1f} | epoch_loss:{:.3f} | acc:{:.3f} | roc:{:.3f}".format(
                    n_epi, score/print_interval, memory.size(), epsilon*100, total_reward, epoch_loss, acc, roc))

            episodes_total_reward.append(total_reward)
            pr_auc_history.append(pr)
            roc_auc_history.append(roc)
            acc_history.append(acc)
            score = 0.0

    env.close()

    # save model
    model_name = 'model_{0}_{1}_.pth'.format(num_epochs, epoch_1_steps)
    save_model(model=q_target, model_name=model_name)

    # show pic
    show_results(episodes_total_reward, pr_auc_history, roc_auc_history, acc_history)


if __name__ == '__main__':
    main()
