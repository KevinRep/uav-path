import numpy as np
from rl_uav_environment import RLUAVEnvironment
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, learning_rate=0.05, discount_factor=0.99, epsilon=0.2):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_state_key(self, state):
        """将状态转换为可哈希的键，简化状态空间表示"""
        current_uav = state['current_uav_id']
        uav_info = state[f'uav{current_uav}']
        
        # 简化无人机状态：只保留位置和总资源量
        uav_pos = tuple(uav_info[:2])  # 位置
        total_resources = sum(uav_info[2:])  # 总资源量
        
        # 简化地点状态：只关注未服务的地点
        unserved_locations = tuple(
            i for i in range(1, 6)
            if not state[f'L{i}'][-1]  # 最后一个元素是服务状态
        )
        
        return (current_uav, uav_pos, total_resources, unserved_locations)
    
    def get_action(self, state):
        """使用epsilon-greedy策略选择动作"""
        state_key = self.get_state_key(state)
        
        # 探索：随机选择动作
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.env.action_space))
        
        # 利用：选择Q值最大的动作
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.env.action_space))
        return np.argmax(self.q_table[state_key])
    
    def train(self, episodes=1000):
        """训练Q-learning算法"""
        rewards_history = []
        steps_history = []
        success_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.get_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step([action])
                
                # 更新Q值
                current_state_key = self.get_state_key(state)
                next_state_key = self.get_state_key(next_state)
                
                # 确保状态存在于Q表中
                if current_state_key not in self.q_table:
                    self.q_table[current_state_key] = np.zeros(len(self.env.action_space))
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = np.zeros(len(self.env.action_space))
                
                # Q-learning更新公式
                old_value = self.q_table[current_state_key][action]
                next_max = np.max(self.q_table[next_state_key])
                new_value = (1 - self.learning_rate) * old_value + \
                           self.learning_rate * (reward + self.discount_factor * next_max)
                self.q_table[current_state_key][action] = new_value
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # 记录训练历史
            rewards_history.append(total_reward)
            steps_history.append(steps)
            success_history.append(1 if done and total_reward > 0 else 0)
            
            # 每100个回合打印一次进度
            if (episode + 1) % 100 == 0:
                success_rate = np.mean(success_history[-100:])
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                print(f"Episode {episode + 1}/{episodes}")
                print(f"Success Rate: {success_rate:.2f}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Steps: {avg_steps:.2f}\n")
        
        return rewards_history, steps_history, success_history
    
    def plot_training_results(self, rewards_history, steps_history, success_history):
        """绘制训练结果"""
        episodes = len(rewards_history)
        
        # 计算移动平均
        window = 100
        rewards_smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        steps_smooth = np.convolve(steps_history, np.ones(window)/window, mode='valid')
        success_smooth = np.convolve(success_history, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(15, 5))
        
        # 绘制奖励曲线
        plt.subplot(131)
        plt.plot(rewards_smooth)
        plt.title('平均奖励')
        plt.xlabel('训练回合')
        plt.ylabel('奖励')
        
        # 绘制步数曲线
        plt.subplot(132)
        plt.plot(steps_smooth)
        plt.title('平均步数')
        plt.xlabel('训练回合')
        plt.ylabel('步数')
        
        # 绘制成功率曲线
        plt.subplot(133)
        plt.plot(success_smooth)
        plt.title('成功率')
        plt.xlabel('训练回合')
        plt.ylabel('成功率')
        
        plt.tight_layout()
        plt.savefig('q_learning_training.png')
        plt.close()
    
    def test(self, episodes=5):
        """测试训练好的Q-learning模型"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 测试时完全按照Q表选择动作
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            print(f"\n开始测试回合 {episode + 1}")
            
            while not done:
                # 选择动作
                action = self.get_action(state)
                
                # 执行动作并获取结果
                next_state, reward, done, info = self.env.step([action])
                
                # 打印详细信息
                current_uav = state['current_uav_id']
                action_name = self.env.action_space[action]
                print(f"步骤 {steps + 1}:")
                print(f"  无人机 {current_uav} 执行动作: {action_name}")
                if 'messages' in info:
                    for msg in info['messages']:
                        print(f"  {msg}")
                
                state = next_state
                total_reward += reward
                steps += 1
            
            print(f"\n回合 {episode + 1} 结束")
            print(f"总步数: {steps}")
            print(f"总奖励: {total_reward:.2f}")
            print("=" * 50)
        
        self.epsilon = original_epsilon  # 恢复原始探索率

def main():
    # 创建环境和Q-learning代理
    env = RLUAVEnvironment()
    agent = QLearning(env, learning_rate=0.05, discount_factor=0.99, epsilon=0.2)
    
    # 训练代理
    print("开始训练Q-learning代理...")
    rewards_history, steps_history, success_history = agent.train(episodes=100000)
    
    # 绘制训练结果
    agent.plot_training_results(rewards_history, steps_history, success_history)
    print("训练完成！结果已保存到'q_learning_training.png'")
    
    # 测试训练好的模型
    print("\n开始测试训练好的模型...")
    agent.test(episodes=5)

if __name__ == "__main__":
    main()