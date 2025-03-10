import gym
from gym import spaces
import numpy as np
from typing import List, Dict, Tuple
from core.models import UAV, Location, Resource
from core.energy_model import EnergyModel

class UAVEnvironment(gym.Env):
    def __init__(self, 
                 locations: List[Location], 
                 uavs: List[UAV],
                 max_steps: int = 1000):
        super().__init__()
        self.locations = locations
        self.uavs = uavs
        self.max_steps = max_steps
        self.current_step = 0
        
        # 状态空间
        self.observation_space = self._create_observation_space()
        
        # 动作空间：每个UAV选择下一个地点
        self.action_space = spaces.MultiDiscrete(
            [len(self.locations) + 1] * len(self.uavs)
        )
        
    def _create_observation_space(self):
        n_uavs = len(self.uavs)
        n_locations = len(self.locations)
        n_resource_types = len(set().union(
            *[uav.resources.keys() for uav in self.uavs]
        ))
        
        return spaces.Dict({
            'uav_states': spaces.Dict({
                'positions': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_uavs, 3)
                ),
                'resources': spaces.Box(
                    low=0, high=np.inf, shape=(n_uavs, n_resource_types)
                ),
                'energy': spaces.Box(
                    low=0, high=np.inf, shape=(n_uavs,)
                )
            }),
            'location_states': spaces.Dict({
                'served': spaces.MultiBinary(n_locations),
                'demands': spaces.Box(
                    low=0, high=np.inf, 
                    shape=(n_locations, n_resource_types)
                )
            })
        })
    
    def reset(self):
        self.current_step = 0
        
        # 重置UAV状态
        for uav in self.uavs:
            uav.position = (0, 0, 0)
            uav.resources = uav.initial_resources.copy()
            uav.remaining_energy = uav.energy_params.battery_capacity
            uav.path.clear()
        
        # 重置地点状态
        for loc in self.locations:
            loc.is_served = False
            
        return self._get_observation()
    
    def step(self, action):
        self.current_step += 1
        reward = 0
        done = False
        
        # 执行动作
        for uav_idx, target_loc_idx in enumerate(action):
            if target_loc_idx < len(self.locations):
                reward += self._process_uav_action(
                    self.uavs[uav_idx], 
                    self.locations[target_loc_idx]
                )
        
        # 检查是否完成所有任务
        all_served = all(loc.is_served for loc in self.locations)
        timeout = self.current_step >= self.max_steps
        done = all_served or timeout
        
        if all_served:
            completion_rate = sum(1 for loc in self.locations if loc.is_served) / len(self.locations)
            time_efficiency = max(0, 1 - self.current_step / self.max_steps)
            reward += 100 * completion_rate * time_efficiency  # 根据完成率和时间效率给奖励
        elif timeout:
            completion_rate = sum(1 for loc in self.locations if loc.is_served) / len(self.locations)
            reward -= 20 * (1 - completion_rate)  # 根据未完成率给惩罚
        
        return self._get_observation(), reward, done, {}
    def _get_observation(self):
        return {
            'uav_states': {
                'positions': np.array([uav.position for uav in self.uavs]),
                'resources': self._get_resource_matrix(),
                'energy': np.array([uav.remaining_energy for uav in self.uavs])
            },
            'location_states': {
                'served': np.array([loc.is_served for loc in self.locations]),
                'demands': self._get_demand_matrix()
            }
        }
    def _get_resource_matrix(self):
        """获取所有UAV的资源矩阵"""
        resource_types = sorted(set().union(
            *[uav.resources.keys() for uav in self.uavs]
        ))
        n_uavs = len(self.uavs)
        n_resources = len(resource_types)
        
        matrix = np.zeros((n_uavs, n_resources))
        for i, uav in enumerate(self.uavs):
            for j, r_type in enumerate(resource_types):
                matrix[i, j] = uav.resources.get(r_type, 0)
        return matrix
    def _get_demand_matrix(self):
        """获取所有地点的需求矩阵"""
        resource_types = sorted(set().union(
            *[uav.resources.keys() for uav in self.uavs]
        ))
        n_locations = len(self.locations)
        n_resources = len(resource_types)
        
        matrix = np.zeros((n_locations, n_resources))
        for i, loc in enumerate(self.locations):
            for j, r_type in enumerate(resource_types):
                matrix[i, j] = loc.demands.get(r_type, 0)
        return matrix
    def _process_uav_action(self, uav: UAV, location: Location) -> float:
        # 检查时间窗口约束
        if self.current_step < location.time_window.start_time:  # 修改这里
            return -1  # 太早到达的惩罚
        if self.current_step > location.time_window.end_time:    # 修改这里
            return -5  # 超时的严重惩罚
        
        # 如果地点已经被服务过，给予负奖励
        if location.is_served:
            return -1  # 减小惩罚
        
        # 计算距离和能耗
        distance = np.linalg.norm(
            np.array(location.position) - np.array(uav.position)
        )
        
        velocity = 0.6 * uav.energy_params.max_speed
        energy_model = EnergyModel(uav.energy_params)
        energy_consumption = energy_model.calculate_energy_consumption(
            distance=distance,
            velocity=velocity,
            hover_time=location.service_time,
            payload=sum(uav.resources.values())
        )
        
        # 能量不足，给予高负奖励
        if energy_consumption > uav.remaining_energy:
            return -5  # 减小惩罚
        
        # 资源不足，给予负奖励
        if not uav.can_serve(location):
            return -2  # 减小惩罚
        
        # 执行服务
        uav.serve_location(location)
        location.is_served = True
        uav.position = location.position
        uav.remaining_energy -= energy_consumption
        uav.path.append(location.id)
        
        # 计算奖励
        reward = 0
        
        # 1. 完成任务的基础奖励（降低基础奖励）
        completion_bonus = 5.0
        time_factor = max(0.5, 1 - self.current_step / self.max_steps)
        reward += completion_bonus * time_factor
        
        # 2. 能量消耗惩罚（增加权重）
        energy_penalty = -0.1 * (energy_consumption / uav.energy_params.battery_capacity)
        reward += energy_penalty
        
        # 3. 距离奖励（增加权重）
        distance_penalty = -0.05 * distance
        reward += distance_penalty
        
        # 4. 资源利用效率奖励（降低奖励）
        total_demands = sum(location.demands.values())
        if total_demands > 0:
            resource_efficiency = sum(min(uav.resources[r], location.demands[r]) 
                                for r in location.demands) / total_demands
            reward += 1.0 * resource_efficiency
        
        # 5. 时间窗口奖励（降低奖励）
        if self.current_step <= location.time_window.end_time:
            time_efficiency = 1.0 - (self.current_step - location.time_window.start_time) / (location.time_window.end_time - location.time_window.start_time)
            time_window_reward = 2.0 * time_efficiency
            reward += time_window_reward
        
        return reward