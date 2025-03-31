# 无人机路径规划强化学习环境

import numpy as np
from uav_path_planning import UAVPathPlanning

class RLUAVEnvironment:
    """
    无人机路径规划强化学习环境
    状态空间：
    - 当前控制的无人机ID
    - 每个无人机的位置和剩余资源
    - 每个地点的资源需求状态
    - 全局状态（已服务地点数、未服务地点数）
    
    动作空间：
    - 前往地点1-5
    - 返回基地
    - 服务当前地点
    
    奖励设计：
    - 基于资源匹配度的正奖励
    - 无效动作的负奖励
    - 完成所有服务的额外奖励
    """
    
    def __init__(self):
        self.planner = None
        self.current_uav_id = 1
        self.action_space = [
            "前往地点1", "前往地点2", "前往地点3", "前往地点4", "前往地点5",
            "返回基地", "服务当前地点"
        ]
        self.reset()
    
    def setup_default_scenario(self):
        """设置默认场景"""
        self.planner = UAVPathPlanning()
        
        # 添加地点及其资源需求
        self.planner.add_location(1, (2, 3), {'a': 1, 'b': 1})
        self.planner.add_location(2, (5, 1), {'a': 1, 'c': 1})
        self.planner.add_location(3, (8, 4), {'a': 2, 'c': 1, 'd': 2})
        self.planner.add_location(4, (6, 7), {'b': 1, 'd': 1})
        self.planner.add_location(5, (3, 6), {'c': 1, 'd': 1})
        
        # 添加无人机及其携带的资源
        self.planner.add_uav(1, {'a': 2, 'b': 1, 'c': 3}, speed=1.0)
        self.planner.add_uav(2, {'a': 2, 'b': 1, 'd': 4}, speed=1.2)
    
    def reset(self):
        """重置环境状态"""
        if self.planner is None:
            self.setup_default_scenario()
        else:
            # 重置所有地点的服务状态
            for loc_info in self.planner.locations.values():
                loc_info['served'] = False
                loc_info['remaining_demands'] = loc_info['resource_demands'].copy()
            
            # 重置所有无人机的状态
            for uav_info in self.planner.uavs.values():
                uav_info['remaining_resources'] = uav_info['resources_carried'].copy()
                uav_info['current_position'] = self.planner.base_location
                uav_info['path'] = [self.planner.base_location]
                uav_info['service_time'] = 0
                uav_info['flight_time'] = 0
                uav_info['total_time'] = 0
                uav_info['served_locations'] = []
        
        self.current_uav_id = 1
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        state = {
            'current_uav_id': self.current_uav_id,
            'global': [
                sum(1 for loc in self.planner.locations.values() if loc['served']),  # 已服务地点数
                sum(1 for loc in self.planner.locations.values() if not loc['served'])  # 未服务地点数
            ]
        }
        
        # 添加每个无人机的状态
        for uav_id, uav_info in self.planner.uavs.items():
            pos = uav_info['current_position']
            resources = list(uav_info['remaining_resources'].values())
            state[f'uav{uav_id}'] = [pos[0], pos[1]] + resources
        
        # 添加每个地点的状态
        for loc_id, loc_info in self.planner.locations.items():
            pos = loc_info['position']
            demands = list(loc_info['remaining_demands'].values())
            served = 1 if loc_info['served'] else 0
            state[f'L{loc_id}'] = [pos[0], pos[1]] + demands + [served]
        
        return state
    
    def _calculate_reward(self, action, resources_provided=None):
        """计算奖励"""
        reward = 0
        messages = []        
        # 获取当前无人机和其位置
        uav = self.planner.uavs[self.current_uav_id]
        current_pos = uav['current_position']
        
        # 前往地点的动作
        if action < 5:  # 前往地点1-5
            target_loc_id = action + 1
            target_loc = self.planner.locations[target_loc_id]
            
            if current_pos == target_loc['position']:
                reward -= 5  # 已经在目标位置，无效动作惩罚加大
                messages.append(f"无人机{self.current_uav_id}已经在地点{target_loc_id}")
            elif target_loc['served']:
                reward -= 6  # 前往已服务的地点，无效动作惩罚加大
                messages.append(f"地点{target_loc_id}已经被服务完成")
            else:
                # 计算资源匹配度和距离的综合评分
                matching_score = 0
                for resource, demand in target_loc['remaining_demands'].items():
                    if demand > 0 and resource in uav['remaining_resources']:
                        matching_score += min(demand, uav['remaining_resources'][resource])
                
                if matching_score > 0:
                    # 计算距离
                    distance = np.sqrt((current_pos[0] - target_loc['position'][0])**2 + 
                                     (current_pos[1] - target_loc['position'][1])**2)
                    # 综合评分 = 匹配度 / 距离
                    score = matching_score / (distance + 1) 
                    reward += score * 4  
                    messages.append(f"无人机{self.current_uav_id}前往地点{target_loc_id}，综合评分{score:.2f}")
                else:
                    reward -= 5  # 没有匹配的资源
                    messages.append(f"无人机{self.current_uav_id}无法为地点{target_loc_id}提供资源")
        
        # 返回基地
        elif action == 5:
            if current_pos == self.planner.base_location:
                reward -= 2  # 已经在基地，无效动作惩罚加大
                messages.append(f"无人机{self.current_uav_id}已经在基地")
            else:
                # 如果资源耗尽，奖励返回基地
                if all(res == 0 for res in uav['remaining_resources'].values()):
                    reward += 6  # 增加奖励
                    messages.append(f"无人机{self.current_uav_id}资源耗尽，返回基地")
                else:
                    reward -= 3  # 还有资源就返回基地，惩罚加大
                    messages.append(f"无人机{self.current_uav_id}还有资源但选择返回基地")
        
        # 服务当前地点
        elif action == 6:
            # 找到当前所在地点
            current_loc_id = None
            for loc_id, loc_info in self.planner.locations.items():
                if loc_info['position'] == current_pos:
                    current_loc_id = loc_id
                    break
            
            if current_loc_id is None:
                reward -= 6  # 不在任何地点，无法服务，惩罚加大
                messages.append(f"无人机{self.current_uav_id}不在任何地点，无法提供服务")
            elif self.planner.locations[current_loc_id]['served']:
                reward -= 6  # 地点已被服务，惩罚加大
                messages.append(f"地点{current_loc_id}已经被服务完成")
            elif resources_provided:
                # 根据提供的资源数量和时间效率给予奖励
                resource_reward = sum(resources_provided.values()) * 10  # 增加基础奖励
                # 计算时间效率奖励
                time_efficiency = 1.0
                if uav['total_time'] > 0:
                    time_efficiency = len(uav['served_locations']) / uav['total_time']
                reward += resource_reward * (1 + time_efficiency) * 3  # 增加奖励权重
                messages.append(f"无人机{self.current_uav_id}成功为地点{current_loc_id}提供服务，资源奖励{resource_reward:.2f}，时间效率{time_efficiency:.2f}")
            else:
                reward -= 5  # 无法提供任何资源，惩罚加大
                messages.append(f"无人机{self.current_uav_id}无法为地点{current_loc_id}提供任何资源")
        
        # 检查是否所有地点都已被服务
        if self.planner.check_all_locations_served():
            reward += 200  # 显著提高完成所有服务的奖励
            messages.append("所有地点服务完成！")
        
        return reward, messages
    
    def step(self, actions):
        """执行动作并返回新的状态、奖励和是否结束"""
        action = actions[0]  # 只处理当前控制的无人机的动作
        
        # 获取当前无人机
        uav = self.planner.uavs[self.current_uav_id]
        resources_provided = None
        
        # 执行动作
        if action < 5:  # 前往地点1-5
            target_loc_id = action + 1
            if uav['current_position'] != self.planner.locations[target_loc_id]['position']:
                flight_time = self.planner.move_uav(self.current_uav_id, target_loc_id)
                uav['total_time'] += flight_time
        
        elif action == 5:  # 返回基地
            if uav['current_position'] != self.planner.base_location:
                flight_time = self.planner.move_uav(self.current_uav_id, -1)
                uav['total_time'] += flight_time
        
        elif action == 6:  # 服务当前地点
            # 找到当前所在地点
            current_loc_id = None
            for loc_id, loc_info in self.planner.locations.items():
                if loc_info['position'] == uav['current_position']:
                    current_loc_id = loc_id
                    break
            
            if current_loc_id is not None and not self.planner.locations[current_loc_id]['served']:
                service_time, resources_provided = self.planner.serve_location(self.current_uav_id, current_loc_id)
                uav['total_time'] += service_time
        
        # 计算奖励
        reward, messages = self._calculate_reward(action, resources_provided)
        
        # 更新状态
        new_state = self._get_state()
        
        # 检查是否结束
        done = self.planner.check_all_locations_served()
        
        # 切换到下一个无人机
        self.current_uav_id = 2 if self.current_uav_id == 1 else 1
        
        return new_state, reward, done, {'messages': messages}
    
    def render(self):
        """可视化当前状态"""
        self.planner.visualize_paths("当前状态")