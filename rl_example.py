# 无人机路径规划强化学习示例

import numpy as np
import matplotlib.pyplot as plt
import time
from rl_uav_environment import RLUAVEnvironment

class QLearningAgent:
    """
    Q-learning算法实现
    """
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """
        初始化Q-learning代理
        :param action_space_size: 动作空间大小
        :param learning_rate: 学习率
        :param discount_factor: 折扣因子
        :param exploration_rate: 探索率
        :param exploration_decay: 探索率衰减
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # 使用字典实现Q表，因为状态空间可能很大
    
    def _get_state_key(self, state):
        """
        将状态转换为可哈希的键
        :param state: 状态字典
        :return: 状态键
        """
        # 简化状态表示，只使用关键信息
        key_parts = []
        
        # 当前控制的无人机ID
        key_parts.append(f"uav{state['current_uav_id']}")
        
        # 当前无人机的位置和资源
        uav_state = state[f"uav{state['current_uav_id']}"]
        x, y = uav_state[0], uav_state[1]
        key_parts.append(f"pos({int(x)},{int(y)})")
        
        # 无人机剩余资源（简化为总量）
        total_resources = sum(uav_state[2:6])
        key_parts.append(f"res{total_resources}")
        
        # 地点服务状态（哪些地点已服务）
        served_locations = []
        for i in range(1, 6):
            if f"L{i}" in state and state[f"L{i}"][-1] == 1:
                served_locations.append(i)
        key_parts.append(f"served{tuple(served_locations)}")
        
        # 剩余未服务地点数量
        key_parts.append(f"remaining{state['global'][1]}")
        
        return "_".join(map(str, key_parts))
    
    def get_action(self, state):
        """
        根据当前状态选择动作
        :param state: 当前状态
        :return: 选择的动作索引
        """
        state_key = self._get_state_key(state)
        
        # 探索：随机选择动作
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        
        # 利用：选择Q值最大的动作
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        
        return np.argmax(self.q_table[state_key])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        更新Q值
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # 如果状态不在Q表中，初始化
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        # 更新Q值
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_exploration(self):
        """
        衰减探索率
        """
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)  # 设置最小探索率

def train_q_learning(episodes=1000, max_steps=100):
    """
    训练Q-learning代理
    :param episodes: 训练回合数
    :param max_steps: 每回合最大步数
    :return: 训练历史记录
    """
    # 创建环境和代理
    env = RLUAVEnvironment()
    env.setup_default_scenario()
    agent = QLearningAgent(action_space_size=len(env.action_space))
    
    # 训练历史记录
    history = {
        'episode_rewards': [],
        'episode_steps': [],
        'success_rate': []
    }
    
    # 成功完成的回合数（所有地点都被服务）
    success_count = 0
    
    print("开始Q-learning训练...")
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 选择动作
            action = agent.get_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新Q值
            agent.update_q_value(state, action, reward, next_state, done)
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            step += 1
            
            # 如果是演示模式，打印详细信息
            if episode % 100 == 0 and episode > episodes - 100:
                print(f"步骤 {step}: {env.action_space[action]} - 奖励: {reward:.2f} - {info['message']}")
        
        # 检查是否成功完成任务（所有地点都被服务）
        success = env.planner.check_all_locations_served()
        if success:
            success_count += 1
        
        # 衰减探索率
        agent.decay_exploration()
        
        # 记录历史
        history['episode_rewards'].append(total_reward)
        history['episode_steps'].append(step)
        history['success_rate'].append(success_count / episode)
        
        # 打印训练进度
        if episode % 100 == 0:
            success_rate = success_count / episode * 100
            print(f"回合 {episode}/{episodes} - 总奖励: {total_reward:.2f} - 步数: {step} - 探索率: {agent.exploration_rate:.4f} - 成功率: {success_rate:.2f}%")
            
            # 如果是演示回合，可视化路径
            if episode % 100 == 0 and episode > episodes - 100:
                env.planner.visualize_paths(f"Q-learning回合{episode}")
    
    print(f"训练完成! 最终成功率: {success_count / episodes * 100:.2f}%")
    return history, agent

def test_q_learning(agent, render=True):
    """
    测试训练好的Q-learning代理
    :param agent: 训练好的代理
    :param render: 是否渲染环境
    :return: 测试结果
    """
    env = RLUAVEnvironment()
    env.setup_default_scenario()
    state = env.reset()
    
    total_reward = 0
    done = False
    step = 0
    
    print("\n===== 测试Q-learning策略 =====")
    while not done and step < 100:
        # 选择动作（测试时不探索）
        action = np.argmax(agent.q_table.get(agent._get_state_key(state), np.zeros(len(env.action_space))))
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 更新状态和累计奖励
        state = next_state
        total_reward += reward
        step += 1
        
        # 渲染环境
        if render:
            env.render()
            print(f"步骤 {step}: {env.action_space[action]} - 奖励: {reward:.2f}")
            print(info['message'])
            print("\n")
    
    # 检查是否成功完成任务
    success = env.planner.check_all_locations_served()
    print(f"测试结果 - 总奖励: {total_reward:.2f} - 步数: {step} - 成功: {'是' if success else '否'}")
    
    # 可视化路径
    env.planner.visualize_paths("Q-learning测试")
    
    return {
        'total_reward': total_reward,
        'steps': step,
        'success': success
    }

def plot_training_history(history):
    """
    绘制训练历史
    :param history: 训练历史记录
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['episode_rewards'])
    plt.title('回合奖励')
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    
    # 绘制步数曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['episode_steps'])
    plt.title('回合步数')
    plt.xlabel('回合')
    plt.ylabel('步数')
    
    # 绘制成功率曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['success_rate'])
    plt.title('累计成功率')
    plt.xlabel('回合')
    plt.ylabel('成功率')
    
    plt.tight_layout()
    plt.savefig('q_learning_training.png')
    plt.close()

def compare_with_other_algorithms():
    """
    比较Q-learning与其他算法的性能
    """
    from uav_path_planning import UAVPathPlanning
    from main import setup_scenario
    
    # 创建环境和代理
    env = RLUAVEnvironment()
    env.setup_default_scenario()
    
    # 训练Q-learning代理（简短训练）
    print("训练Q-learning代理...")
    history, agent = train_q_learning(episodes=500, max_steps=100)
    
    # 测试Q-learning
    print("测试Q-learning...")
    ql_result = test_q_learning(agent, render=False)
    ql_time = max([uav_info['total_time'] for uav_info in env.planner.uavs.values()])
    
    # 测试启发式算法
    print("\n测试启发式算法...")
    heuristic_planner = setup_scenario()
    start_time = time.time()
    heuristic_planner.heuristic_algorithm()
    heuristic_computation_time = time.time() - start_time
    heuristic_planner.print_statistics()
    heuristic_time = max([uav_info['total_time'] for uav_info in heuristic_planner.uavs.values()])
    
    # 测试随机算法
    print("\n测试随机算法...")
    random_planner = setup_scenario()
    start_time = time.time()
    random_planner.random_algorithm()
    random_computation_time = time.time() - start_time
    random_planner.print_statistics()
    random_time = max([uav_info['total_time'] for uav_info in random_planner.uavs.values()])
    
    # 比较结果
    print("\n===== 算法性能比较 =====")
    print(f"Q-learning: 总时间 {ql_time:.2f}秒, 成功: {'是' if ql_result['success'] else '否'}")
    print(f"启发式算法: 总时间 {heuristic_time:.2f}秒, 计算时间: {heuristic_computation_time:.4f}秒")
    print(f"随机策略: 总时间 {random_time:.2f}秒, 计算时间: {random_computation_time:.4f}秒")
    
    # 绘制比较图表
    plt.figure(figsize=(10, 6))
    algorithms = ["Q-learning", "启发式算法", "随机策略"]
    times = [ql_time, heuristic_time, random_time]
    
    # 绘制比较图表
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, times)
    plt.title('不同算法的总服务时间比较')
    plt.ylabel('总时间（秒）')
    plt.savefig('rl_policy_comparison.png')
    plt.close()

# 主函数
if __name__ == "__main__":
    # 简单测试环境
    print("测试强化学习环境...")
    env = RLUAVEnvironment()
    env.setup_default_scenario()
    state = env.reset()
    
    # 执行一些随机动作
    for i in range(5):
        action = np.random.randint(0, len(env.action_space))
        next_state, reward, done, info = env.step(action)
        print(f"动作: {env.action_space[action]}, 奖励: {reward:.2f}, 信息: {info['message']}")
    
    print("\n开始训练Q-learning算法...")
    # 训练Q-learning算法
    history, agent = train_q_learning(episodes=300, max_steps=50)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 测试训练好的代理
    test_q_learning(agent)
    
    # 比较不同算法的性能
    compare_with_other_algorithms()