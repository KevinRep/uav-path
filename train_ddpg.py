import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from rl_uav_environment import RLUAVEnvironment

# 创建环境和代理
env = RLUAVEnvironment()
agent = DDPGAgent(env)

# 训练参数
num_episodes = 1000
rewards_history = []

# 训练代理
for episode in range(num_episodes):
    state = env.reset()
    state = agent.flatten_state(state)
    episode_reward = 0
    done = False
    step = 0
    
    while not done:
        # 选择动作
        action = agent.ddpg.select_action(state)
        action = np.clip(action + agent.noise.sample(), -1, 1)
        
        # 执行动作
        next_state, reward, done, info = env.step([np.argmax(action)])
        next_state = agent.flatten_state(next_state)
        episode_reward += reward
        
        # 存储经验
        agent.ddpg.replay_buffer.push(state, action, reward, next_state, float(done))
        
        # 训练
        agent.ddpg.train()
        
        state = next_state
        step += 1
        
        # # 打印调试信息
        # if 'messages' in info:
        #     for msg in info['messages']:
        #         print(f'Step {step}: {msg}')
    
    rewards_history.append(episode_reward)
    print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward:.2f}, Steps: {step}')
    
    # 每100个episode保存一次训练曲线
    if (episode + 1) % 100 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_history)
        plt.title('DDPG Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('ddpg_training.png')
        plt.close()

# 保存最终的训练曲线
plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.title('DDPG Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('ddpg_training_final.png')
plt.close()

print('Training completed!')

# 测试训练好的代理
print('\nTesting the trained agent...')
state = env.reset()
state = agent.flatten_state(state)
done = False
total_reward = 0
step = 0

while not done:
    action = agent.ddpg.select_action(state)
    next_state, reward, done, info = env.step([np.argmax(action)])
    next_state = agent.flatten_state(next_state)
    total_reward += reward
    state = next_state
    step += 1
    
    if 'messages' in info:
        for msg in info['messages']:
            print(f'Test Step {step}: {msg}')

print(f'\nTest completed! Total Reward: {total_reward:.2f}, Steps: {step}')

# 可视化最终路径
env.render()