import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.actor(state), self.critic(state)

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self._get_state_dim()
        self.action_dim = env.action_space.nvec[0]
        self.n_uavs = len(env.uavs)
        self.learning_rate = 1e-4  # 增大学习率
        self.clip_param = 0.2    # 增大裁剪参数
        self.entropy_coef = 0.03  # 增大熵系数，增加探索
        self.gamma = 0.99        # 保持不变
        self.gae_lambda = 0.95   # GAE参数
        self.ppo_epochs = 10     # 添加PPO更新轮数
        self.batch_size = 64     # 添加批次大小
        self.max_grad_norm = 0.5
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95
        self.network = PPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)  # 修改这里，使用self.learning_rate
    
    def _preprocess_state(self, state):
        # 获取当前环境的规模
        current_n_uavs = len(state['uav_states']['positions']) // 3
        current_n_locations = len(state['location_states']['served'])
        
        # 如果规模不同，需要填充或裁剪状态
        if current_n_uavs != self.original_n_uavs or current_n_locations != self.original_n_locations:
            # 创建填充后的状态
            padded_state = {
                'uav_states': {
                    'positions': np.zeros((self.original_n_uavs * 3,)),
                    'resources': np.zeros((self.original_n_uavs * 4,)),  # 4种资源
                    'energy': np.zeros((self.original_n_uavs,))
                },
                'location_states': {
                    'served': np.zeros((self.original_n_locations,)),
                    'demands': np.zeros((self.original_n_locations * 4,))  # 4种资源
                }
            }
            
            # 复制可用的数据
            n_uavs_copy = min(current_n_uavs, self.original_n_uavs)
            n_locs_copy = min(current_n_locations, self.original_n_locations)
            
            padded_state['uav_states']['positions'][:n_uavs_copy*3] = state['uav_states']['positions'][:n_uavs_copy*3]
            padded_state['uav_states']['resources'][:n_uavs_copy*4] = state['uav_states']['resources'][:n_uavs_copy*4]
            padded_state['uav_states']['energy'][:n_uavs_copy] = state['uav_states']['energy'][:n_uavs_copy]
            padded_state['location_states']['served'][:n_locs_copy] = state['location_states']['served'][:n_locs_copy]
            padded_state['location_states']['demands'][:n_locs_copy*4] = state['location_states']['demands'][:n_locs_copy*4]
            
            state = padded_state
        
        # 归一化状态值
        state_parts = [
            state['uav_states']['positions'].flatten() / 200.0,
            state['uav_states']['resources'].flatten() / 10.0,
            state['uav_states']['energy'].flatten() / 800.0,
            state['location_states']['served'].flatten(),
            state['location_states']['demands'].flatten() / 4.0
        ]
        return torch.FloatTensor(np.concatenate(state_parts))
    def _get_state_dim(self):
        """计算状态空间维度"""
        sample_state = self.env.reset()
        state_parts = [
            sample_state['uav_states']['positions'].flatten(),
            sample_state['uav_states']['resources'].flatten(),
            sample_state['uav_states']['energy'].flatten(),
            sample_state['location_states']['served'].flatten(),
            sample_state['location_states']['demands'].flatten()
        ]
        return len(np.concatenate(state_parts))
    
    def _preprocess_state(self, state):
        # 归一化状态值
        state_parts = [
            state['uav_states']['positions'].flatten() / 200.0,  # 位置归一化
            state['uav_states']['resources'].flatten() / 10.0,   # 资源归一化
            state['uav_states']['energy'].flatten() / 800.0,     # 能量归一化
            state['location_states']['served'].flatten(),
            state['location_states']['demands'].flatten() / 4.0  # 需求归一化
        ]
        return torch.FloatTensor(np.concatenate(state_parts))
    
    def select_action(self, state):
        state_tensor = self._preprocess_state(state)
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
            
        actions = []
        for _ in range(self.n_uavs):
            # 确保概率和为1且非负
            probs = torch.clamp(action_probs, min=1e-6)
            probs = probs / probs.sum()
            
            dist = Categorical(probs)
            action = dist.sample()
            actions.append(action.item())
        
        return np.array(actions)
    def train_episode(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 2500  # 增加最大步数以匹配时间窗口
        
        # 存储轨迹
        states, actions_list, rewards, log_probs_list, values = [], [], [], [], []
        
        while not done and step_count < max_steps:
            state_tensor = self._preprocess_state(state)
            action_probs, value = self.network(state_tensor)
            
            # 选择动作
            actions = []
            log_probs = []
            
            for _ in range(self.n_uavs):
                probs = torch.clamp(action_probs, min=1e-6)
                probs = probs / probs.sum()
                
                dist = Categorical(probs)
                action = dist.sample()
                actions.append(action.item())
                log_probs.append(dist.log_prob(action))
            
            actions = np.array(actions)
            next_state, reward, done, _ = self.env.step(actions)
            
            # 存储经验
            states.append(state_tensor)
            actions_list.append(actions)
            rewards.append(reward)
            log_probs_list.append(log_probs)
            values.append(value)
            
            total_reward += reward
            state = next_state
            step_count += 1
        
        # 多轮更新
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = torch.stack([states[i] for i in batch_indices])
                batch_actions = torch.tensor(np.array([actions_list[i] for i in batch_indices]))
                batch_log_probs = torch.stack([torch.stack(log_probs_list[i]) for i in batch_indices])
                batch_rewards = torch.tensor([rewards[i] for i in batch_indices])
                batch_values = torch.stack([values[i] for i in batch_indices])
                
                # 计算优势
                advantages = []
                for idx in batch_indices:
                    if idx < len(states) - 1:
                        next_value = values[idx + 1]
                        advantage = rewards[idx] + self.gamma * next_value * (1 - int(done)) - values[idx].item()
                    else:
                        advantage = rewards[idx] - values[idx].item()
                    advantages.append(advantage)
                advantages = torch.tensor(advantages).unsqueeze(1)  # 添加维度以匹配形状
                
                # 更新网络
                self.optimizer.zero_grad()
                new_action_probs, new_values = self.network(batch_states)
                
                # 确保维度匹配，不使用detach()
                new_log_probs = []
                for i in range(len(batch_indices)):
                    probs = new_action_probs[i]  # 移除detach()
                    probs = torch.clamp(probs, min=1e-6)
                    probs = probs / probs.sum()
                    
                    dist = Categorical(probs)
                    action_log_prob = dist.log_prob(batch_actions[i])
                    new_log_probs.append(action_log_prob)
                new_log_probs = torch.stack(new_log_probs)
                
                # 计算损失时只对old_log_probs使用detach
                ratio = torch.exp(new_log_probs - batch_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失
                critic_loss = 0.5 * advantages.pow(2).mean()
                
                # 总损失
                loss = actor_loss + critic_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return total_reward
        
    def save_model(self, path):
        """保存模型参数"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])