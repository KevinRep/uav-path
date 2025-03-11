# 无人机路径规划强化学习环境

import numpy as np
import matplotlib.pyplot as plt
from uav_path_planning import UAVPathPlanning

class RLUAVEnvironment:
    """
    无人机路径规划强化学习环境
    实现状态表示、动作执行、奖励计算和状态转移
    """
    def __init__(self, planner=None):
        """
        初始化环境
        :param planner: UAVPathPlanning实例，如果为None则创建新实例
        """
        self.planner = planner if planner else UAVPathPlanning()
        self.action_space = [
            "go_to_L1", "go_to_L2", "go_to_L3", "go_to_L4", "go_to_L5",
            "return_to_base", "serve_current_location"
        ]
        self.current_uav_id = None  # 当前控制的无人机ID
        self.episode_step = 0
        self.max_steps = 100  # 每个episode的最大步数
        self.total_reward = 0
        
    def setup_default_scenario(self):
        """
        设置默认测试场景
        """
        # 添加地点及其资源需求
        self.planner.add_location(1, (2, 3), {'a': 1, 'b': 1})
        self.planner.add_location(2, (5, 1), {'a': 1, 'c': 1})
        self.planner.add_location(3, (8, 4), {'a': 2, 'c': 1, 'd': 2})
        self.planner.add_location(4, (6, 7), {'b': 1, 'd': 1})
        self.planner.add_location(5, (3, 6), {'c': 2, 'd': 1})
        
        # 添加无人机及其携带的资源
        self.planner.add_uav(1, {'a': 2, 'b': 1, 'c': 3}, speed=1.0)
        self.planner.add_uav(2, {'a': 2, 'b': 1, 'd': 4}, speed=1.2)
        
        # 计算所有地点之间的距离
        self.planner.calculate_distances()
        
        # 设置当前控制的无人机
        self.current_uav_id = 1
    
    def reset(self):
        """
        重置环境到初始状态
        :return: 初始状态
        """
        # 重置所有地点的服务状态和剩余需求
        for loc_id, loc_info in self.planner.locations.items():
            loc_info['served'] = False
            loc_info['remaining_demands'] = loc_info['resource_demands'].copy()
        
        # 重置所有无人机的状态
        for uav_id, uav_info in self.planner.uavs.items():
            uav_info['remaining_resources'] = uav_info['resources_carried'].copy()
            uav_info['current_position'] = self.planner.base_location
            uav_info['path'] = [self.planner.base_location]
            uav_info['service_time'] = 0
            uav_info['flight_time'] = 0
            uav_info['total_time'] = 0
            uav_info['served_locations'] = []
        
        self.episode_step = 0
        self.total_reward = 0
        
        # 随机选择一个无人机作为当前控制的无人机
        self.current_uav_id = np.random.choice(list(self.planner.uavs.keys()))
        
        # 确保距离矩阵已计算
        self.planner.calculate_distances()
        
        return self._get_state()
    
    def step(self, action_idx):
        """
        执行动作并返回下一个状态、奖励、是否结束和额外信息
        :param action_idx: 动作索引
        :return: (next_state, reward, done, info)
        """
        self.episode_step += 1
        action = self.action_space[action_idx]
        uav = self.planner.uavs[self.current_uav_id]
        reward = 0
        info = {}
        
        # 解析动作
        if action.startswith("go_to_L"):
            # 前往地点
            location_id = int(action[-1])
            
            # 检查地点是否存在
            if location_id not in self.planner.locations:
                reward = -10  # 惩罚无效动作
                info["message"] = f"地点{location_id}不存在"
            else:
                # 检查地点是否已被服务
                if self.planner.locations[location_id]['served']:
                    reward = -5  # 惩罚前往已服务地点
                    info["message"] = f"地点{location_id}已被服务"
                else:
                    # 确保距离矩阵已计算
                    if not self.planner.distances:
                        self.planner.calculate_distances()
                        
                    # 移动到目标地点
                    flight_time = self.planner.move_uav(self.current_uav_id, location_id)
                    uav['total_time'] += flight_time
                    
                    # 计算奖励：距离越短奖励越高
                    reward = -flight_time  # 飞行时间越短奖励越高
                    info["message"] = f"无人机{self.current_uav_id}前往地点{location_id}，飞行时间: {flight_time:.2f}秒"
        
        elif action == "return_to_base":
            # 返回基地
            if uav['current_position'] == self.planner.base_location:
                reward = -1  # 已经在基地，轻微惩罚
                info["message"] = "无人机已在基地"
            else:
                # 确保距离矩阵已计算
                if not self.planner.distances:
                    self.planner.calculate_distances()
                    
                flight_time = self.planner.move_uav(self.current_uav_id, -1)  # -1表示基地
                uav['total_time'] += flight_time
                reward = -flight_time / 2  # 返回基地的惩罚较轻
                info["message"] = f"无人机{self.current_uav_id}返回基地，飞行时间: {flight_time:.2f}秒"
        
        elif action == "serve_current_location":
            # 原地服务
            current_pos = uav['current_position']
            
            # 检查是否在基地
            if current_pos == self.planner.base_location:
                reward = -5  # 在基地服务，惩罚
                info["message"] = "无人机在基地，无法提供服务"
            else:
                # 找出当前位置对应的地点ID
                current_location_id = None
                for loc_id, loc_info in self.planner.locations.items():
                    if loc_info['position'] == current_pos:
                        current_location_id = loc_id
                        break
                
                if current_location_id is None:
                    reward = -5  # 不在任何地点，惩罚
                    info["message"] = "无人机不在任何地点，无法提供服务"
                else:
                    # 检查地点是否已被服务
                    if self.planner.locations[current_location_id]['served']:
                        reward = -5  # 地点已被服务，惩罚
                        info["message"] = f"地点{current_location_id}已被服务"
                    else:
                        # 服务地点
                        service_time, resources_provided = self.planner.serve_location(self.current_uav_id, current_location_id)
                        uav['total_time'] += service_time
                        
                        # 计算奖励：提供的资源越多奖励越高
                        if resources_provided:
                            total_resources = sum(resources_provided.values())
                            reward = total_resources * 10  # 每单位资源奖励10分
                            info["message"] = f"无人机{self.current_uav_id}服务地点{current_location_id}，提供资源: {resources_provided}，服务时间: {service_time}秒"
                            
                            # 如果地点被完全服务，额外奖励
                            if self.planner.locations[current_location_id]['served']:
                                reward += 50  # 完全服务一个地点的额外奖励
                                info["message"] += "，地点已完全服务！"
                        else:
                            reward = -5  # 无法提供任何资源，惩罚
                            info["message"] = f"无人机{self.current_uav_id}无法为地点{current_location_id}提供任何资源"
        
        # 更新总奖励
        self.total_reward += reward
        
        # 检查是否结束
        done = False
        if self.planner.check_all_locations_served():
            done = True
            reward += 100  # 所有地点都被服务，大奖励
            info["message"] += "\n所有地点都已被服务！"
        elif self.episode_step >= self.max_steps:
            done = True
            info["message"] += "\n达到最大步数限制"
        
        # 随机切换控制的无人机（模拟多无人机协作）
        if not done and len(self.planner.uavs) > 1 and np.random.random() < 0.3:  # 30%概率切换无人机
            available_uavs = list(self.planner.uavs.keys())
            available_uavs.remove(self.current_uav_id)
            self.current_uav_id = np.random.choice(available_uavs)
            info["message"] += f"\n控制切换到无人机{self.current_uav_id}"
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """
        获取当前状态表示
        :return: 状态字典
        """
        state = {}
        
        # 无人机状态
        for uav_id, uav_info in self.planner.uavs.items():
            x, y = uav_info['current_position']
            
            # 检查无人机是否正在服务（简化处理，根据位置判断）
            is_serving = 0
            if uav_info['current_position'] != self.planner.base_location:
                for loc_id, loc_info in self.planner.locations.items():
                    if loc_info['position'] == uav_info['current_position'] and not loc_info['served']:
                        is_serving = 1
                        break
            
            # 资源状态（转换为列表）
            resources = [0, 0, 0, 0]  # [a, b, c, d]
            for i, res in enumerate(['a', 'b', 'c', 'd']):
                if res in uav_info['remaining_resources']:
                    resources[i] = uav_info['remaining_resources'][res]
            
            state[f"uav{uav_id}"] = [x, y] + resources + [is_serving]
        
        # 地点状态
        for loc_id, loc_info in self.planner.locations.items():
            # 资源需求（转换为列表）
            demands = [0, 0, 0, 0]  # [a, b, c, d]
            for i, res in enumerate(['a', 'b', 'c', 'd']):
                if res in loc_info['remaining_demands']:
                    demands[i] = loc_info['remaining_demands'][res]
            
            state[f"L{loc_id}"] = demands + [1 if loc_info['served'] else 0]
        
        # 全局状态
        # 计算总时间和剩余未服务地点数量
        current_time = max([uav_info['total_time'] for uav_info in self.planner.uavs.values()])
        remaining_locations = sum(1 for loc_info in self.planner.locations.values() if not loc_info['served'])
        state["global"] = [current_time, remaining_locations]
        
        # 当前控制的无人机ID
        state["current_uav_id"] = self.current_uav_id
        
        return state
    
    def render(self, mode='human'):
        """
        渲染环境
        :param mode: 渲染模式
        """
        if mode == 'human':
            print(f"\n===== 当前状态 =====")
            print(f"当前控制的无人机: {self.current_uav_id}")
            print(f"当前步数: {self.episode_step}")
            print(f"当前总奖励: {self.total_reward:.2f}")
            
            # 打印无人机状态
            for uav_id, uav_info in self.planner.uavs.items():
                print(f"无人机{uav_id}:")
                print(f"  位置: {uav_info['current_position']}")
                print(f"  剩余资源: {uav_info['remaining_resources']}")
                print(f"  总时间: {uav_info['total_time']:.2f}秒")
            
            # 打印地点状态
            for loc_id, loc_info in self.planner.locations.items():
                status = "已服务" if loc_info['served'] else "未服务"
                print(f"地点{loc_id} ({status}):")
                print(f"  位置: {loc_info['position']}")
                print(f"  剩余需求: {loc_info['remaining_demands']}")
            
            # 可视化当前状态
            self.planner.visualize_paths("当前状态")

# 简单测试环境
if __name__ == "__main__":
    # 创建环境
    env = RLUAVEnvironment()
    env.setup_default_scenario()
    
    # 重置环境
    state = env.reset()
    print("初始状态:")
    print(state)
    
    # 执行一些随机动作
    for i in range(10):
        action = np.random.randint(0, len(env.action_space))
        next_state, reward, done, info = env.step(action)
        
        print(f"\n执行动作: {env.action_space[action]}")
        print(f"奖励: {reward}")
        print(f"信息: {info['message']}")
        
        if done:
            print("环境结束!")
            break
    
    # 渲染最终状态
    env.render()
    
    print("\n测试完成，环境正常工作!")