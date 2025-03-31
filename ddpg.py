import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))  # 输出范围限制在[-1, 1]
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # 先转换为numpy数组，再转换为张量，提高效率
        return (torch.FloatTensor(np.array(state)), 
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(np.array(reward)).reshape(-1, 1),
                torch.FloatTensor(np.array(next_state)),
                torch.FloatTensor(np.array(done)).reshape(-1, 1))
    
    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.cpu().detach().numpy()[0]
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验回放中采样并移动到GPU
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # 计算目标Q值
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (1 - done) * self.gamma * target_Q.detach()
        
        # 更新Critic
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class DDPGAgent:
    def __init__(self, env):
        self.env = env
        
        # 获取状态维度
        state = env.reset()
        state_dim = sum(len(v) if isinstance(v, list) else 1 for v in state.values())
        
        # 动作维度（7个离散动作）
        action_dim = len(env.action_space)
        
        self.ddpg = DDPG(state_dim, action_dim)
        self.noise = OUNoise(action_dim)
        
    def flatten_state(self, state_dict):
        """将状态字典展平为一维数组"""
        flattened = []
        for v in state_dict.values():
            if isinstance(v, list):
                flattened.extend(v)
            else:
                flattened.append(v)
        return np.array(flattened)
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.flatten_state(state)
            episode_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.ddpg.select_action(state)
                action = np.clip(action + self.noise.sample(), -1, 1)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step([np.argmax(action)])
                next_state = self.flatten_state(next_state)
                episode_reward += reward
                
                # 存储经验
                self.ddpg.replay_buffer.push(state, action, reward, next_state, float(done))
                
                # 训练
                self.ddpg.train()
                
                state = next_state
            
            print(f'Episode {episode}, Reward: {episode_reward}')

class OUNoise:
    """Ornstein-Uhlenbeck过程噪声"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state