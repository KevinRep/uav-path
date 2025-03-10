import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
from core.models import Location, UAV, TimeWindow, DroneEnergyParams
from env.uav_env import UAVEnvironment
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
import os
from datetime import datetime
def create_test_scenario(n_locations=10, n_uavs=5):
    # 先生成地点的需求
    total_demands = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    locations = []
    
    for i in range(n_locations):
        # 每个地点需要2-4种资源，增加难度
        n_resource_types = np.random.randint(2, 5)
        resource_types = np.random.choice(['A', 'B', 'C', 'D'], size=n_resource_types, replace=False)
        
        demands = {
            'A': np.random.randint(1, 5) if 'A' in resource_types else 0,  # 增加需求量
            'B': np.random.randint(1, 5) if 'B' in resource_types else 0,
            'C': np.random.randint(1, 5) if 'C' in resource_types else 0,
            'D': np.random.randint(1, 5) if 'D' in resource_types else 0
        }
        
        # 更新总需求
        for resource, amount in demands.items():
            total_demands[resource] += amount
        
        # 为每个地点设置随机的时间窗口
        start_time = np.random.randint(0, 1000)  # 扩大起始时间范围  # 扩大起始时间范围
        window_length = np.random.randint(600, 1500)  # 扩大时间窗口  # 扩大时间窗口
        end_time = start_time + window_length
        
        locations.append(Location(
            id=i,
            position=(np.random.uniform(-1000, 1000),  # 扩大场景范围到2km×2km
                     np.random.uniform(-1000, 1000),
                     20),
            demands=demands,
            time_window=TimeWindow(start_time, end_time),
            service_time=60  # 减少服务时间
        ))
    
    # 创建无人机，确保总资源量满足需求
    energy_params = DroneEnergyParams(
        mass=5.0,
        rotor_area=0.5,
        battery_capacity=800,    # 增加电池容量
        hover_power=180,
        max_speed=20,           # 增加最大速度到20m/s (72km/h)
        blade_drag_coef=0.1
    )
    
    uavs = []
    remaining_demands = total_demands.copy()
    
    for i in range(n_uavs):
        # 每个无人机随机携带2-3种资源
        n_resource_types = np.random.randint(2, 4)
        resource_types = np.random.choice(['A', 'B', 'C', 'D'], size=n_resource_types, replace=False)
        
        resources = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for resource in resource_types:
            if remaining_demands[resource] > 0:
                # 分配1-3个资源，但不超过剩余需求
                amount = min(np.random.randint(1, 4), remaining_demands[resource])
                resources[resource] = amount
                remaining_demands[resource] -= amount
        
        uavs.append(UAV(
            id=i,
            position=(0, 0, 0),
            resources=resources,
            energy_params=energy_params
        ))
    
    # 如果还有未分配的需求，随机分配给无人机
    for resource, amount in remaining_demands.items():
        while amount > 0:
            uav = np.random.choice(uavs)
            add_amount = min(amount, 2)  # 每次最多添加2个
            uav.resources[resource] += add_amount
            amount -= add_amount
    
    return locations, uavs
def evaluate_agent(env, agent, n_episodes=5):
    total_rewards = []
    completion_times = []
    completion_rates = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        completion_times.append(env.current_step)
        completion_rates.append(
            sum(1 for loc in env.locations if loc.is_served) / len(env.locations)
        )
    
    return {
        'rewards': np.mean(total_rewards),
        'times': np.mean(completion_times),
        'rates': np.mean(completion_rates)
    }

def test_scenario(agent, n_locations=5, n_uavs=3, render=True):
    """测试特定场景"""
    locations, uavs = create_test_scenario(n_locations, n_uavs)
    env = UAVEnvironment(locations, uavs)
    state = env.reset()
    done = False
    total_reward = 0
    
    # 增加最大步数限制
    max_steps = 2500  # 增加到2500步
    step_count = 0
    
    # 记录上一步的状态
    last_positions = [None] * n_uavs
    last_paths = [None] * n_uavs
    last_resources = [None] * n_uavs
    
    while not done and step_count < max_steps:  # 添加步数限制
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        step_count += 1
        
        if render:
            # 检查是否有状态变化
            has_changes = False
            for i, uav in enumerate(uavs):
                if (last_positions[i] != uav.position or 
                    last_paths[i] != uav.path or 
                    last_resources[i] != uav.resources):
                    has_changes = True
                    break
            
            # 只在状态发生变化时输出
            if has_changes:
                print(f"\n当前步骤 {env.current_step}:")
                for i, uav in enumerate(uavs):
                    print(f"UAV {uav.id}: 位置={uav.position}, 路径={uav.path}")
                    print(f"剩余资源: {uav.resources}")
                    print(f"剩余能量: {uav.remaining_energy:.2f} Wh")
                    
                    # 更新上一步的状态
                    last_positions[i] = uav.position
                    last_paths[i] = uav.path.copy()
                    last_resources[i] = uav.resources.copy()
    
    # 计算总能量消耗
    total_energy = sum([
        uav.energy_params.battery_capacity - uav.remaining_energy 
        for uav in uavs
    ])
    
    completion_rate = sum(1 for loc in env.locations if loc.is_served) / len(env.locations)
    
    if render:
        print(f"\n测试结果:")
        print(f"总奖励: {total_reward:.2f}")
        print(f"完成率: {completion_rate:.2%}")
        print(f"总步数: {env.current_step}")
        print(f"总能量消耗: {total_energy:.2f} Wh")
    
    return total_reward, completion_rate, env.current_step, total_energy
def plot_training_results(ppo_history, random_history, save_dir):
    metrics = ['rewards', 'times', 'rates']
    titles = ['Average Rewards', 'Completion Times', 'Completion Rates']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        plt.plot(ppo_history[metric], label='PPO')
        plt.plot(random_history[metric], label='Random')
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()
def main():
    # 创建结果保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('d:', 'uavpath', 'results', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境和智能体 - 使用更大规模的场景
    n_episodes = 1000  # 增加训练轮数
    eval_interval = 10  # 减少评估频率
    
    # 实现课程学习
    scenarios = [
        (5, 2),   # 5个地点，2个UAV
        (10, 3),  # 10个地点，3个UAV
        (15, 4),  # 15个地点，4个UAV
        (20, 5)   # 20个地点，5个UAV
    ]
    
    for n_locations, n_uavs in scenarios:
        print(f"\n训练场景: {n_locations}地点, {n_uavs}UAV")
        locations, uavs = create_test_scenario(n_locations, n_uavs)
        env = UAVEnvironment(locations, uavs)
        
        ppo_agent = PPOAgent(env)
        random_agent = RandomAgent(env)
        
        # 训练历史记录
        ppo_history = {'rewards': [], 'times': [], 'rates': []}
        random_history = {'rewards': [], 'times': [], 'rates': []}
        
        for episode in range(n_episodes):
            # PPO训练
            ppo_agent.train_episode()
            
            # 定期评估
            if episode % eval_interval == 0:
                print(f"Episode {episode}/{n_episodes}")
                
                # 评估PPO
                ppo_metrics = evaluate_agent(env, ppo_agent)
                for key in ppo_metrics:
                    ppo_history[key].append(ppo_metrics[key])
                
                # 评估随机策略
                random_metrics = evaluate_agent(env, random_agent)
                for key in random_metrics:
                    random_history[key].append(random_metrics[key])
                
                # 保存训练过程图
                plot_training_results(ppo_history, random_history, save_dir)
            
            # 每100个episode保存一次模型
            if episode % 100 == 0:
                model_path = os.path.join(save_dir, f'ppo_model_ep{episode}.pt')
                ppo_agent.save_model(model_path)
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, f'ppo_model_final_{n_locations}地点_{n_uavs}UAV.pt')
        ppo_agent.save_model(final_model_path)
def load_and_test(model_path):
    """加载已训练的模型并测试"""
    # 创建与训练时相同规模的环境
    n_locations = 20  # 修改为与训练时相同的规模
    n_uavs = 5
    
    # 创建环境和智能体
    locations, uavs = create_test_scenario(n_locations=n_locations, n_uavs=n_uavs)
    env = UAVEnvironment(locations, uavs)
    ppo_agent = PPOAgent(env)
    random_agent = RandomAgent(env)
    
    # 加载模型前先打印维度信息
    print(f"环境状态维度: {ppo_agent.state_dim}")
    print(f"环境动作维度: {ppo_agent.action_dim}")
    
    # 加载PPO模型参数
    ppo_agent.load_model(model_path)
    
    # 进行多次测试
    n_tests = 5
    print(f"\n=== 对比测试 {n_tests} 个随机场景 ===")
    print(f"场景规模: 地点数={n_locations}, UAV数={n_uavs}")
    
    # 记录测试结果
    results = {
        'ppo': {'rewards': [], 'rates': [], 'times': [], 'energy': []},
        'random': {'rewards': [], 'rates': [], 'times': [], 'energy': []}
    }
    
    for i in range(n_tests):
        print(f"\n测试场景 {i+1}:")
        
        # 测试PPO
        print("\nPPO智能体:")
        reward, completion_rate, steps, total_energy = test_scenario(ppo_agent, n_locations, n_uavs)
        results['ppo']['rewards'].append(reward)
        results['ppo']['rates'].append(completion_rate)
        results['ppo']['times'].append(steps)
        results['ppo']['energy'].append(total_energy)
        
        # 测试随机策略
        print("\n随机策略:")
        reward, completion_rate, steps, total_energy = test_scenario(random_agent, n_locations, n_uavs)
        results['random']['rewards'].append(reward)
        results['random']['rates'].append(completion_rate)
        results['random']['times'].append(steps)
        results['random']['energy'].append(total_energy)
    
    # 输出总体对比结果
    print("\n=== 总体测试结果对比 ===")
    print("\nPPO智能体:")
    print(f"平均奖励: {np.mean(results['ppo']['rewards']):.2f}")
    print(f"平均完成率: {np.mean(results['ppo']['rates']):.2%}")
    print(f"平均步数: {np.mean(results['ppo']['times']):.1f}")
    print(f"平均能量消耗: {np.mean(results['ppo']['energy']):.2f} Wh")
    
    print("\n随机策略:")
    print(f"平均奖励: {np.mean(results['random']['rewards']):.2f}")
    print(f"平均完成率: {np.mean(results['random']['rates']):.2%}")
    print(f"平均步数: {np.mean(results['random']['times']):.1f}")
    print(f"平均能量消耗: {np.mean(results['random']['energy']):.2f} Wh")

    
if __name__ == "__main__":
    # 如果要训练新模型
    main()
    
    #如果要测试已有模型
    # model_path = "D:/uavpath/uavpath/results/20250224_062944/ppo_model_final.pt"  # 替换为您的模型文件路径
    # load_and_test(model_path)