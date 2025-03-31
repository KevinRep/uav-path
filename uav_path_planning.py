# 无人机辅助移动边缘计算路径规划

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import defaultdict
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class UAVPathPlanning:
    def __init__(self):
        # 初始化地点、资源需求、无人机及其携带的资源
        self.locations = {}
        self.uavs = {}
        self.resources = set()
        self.distances = {}
        self.base_location = (0, 0)  # 无人机的起点和终点位置
        
    def add_location(self, location_id, position, resource_demands):
        """
        添加地点及其资源需求
        :param location_id: 地点ID
        :param position: 地点位置坐标 (x, y)
        :param resource_demands: 字典，键为资源类型，值为需求数量
        """
        self.locations[location_id] = {
            'position': position,
            'resource_demands': resource_demands,
            'remaining_demands': resource_demands.copy(),  # 用于跟踪剩余需求
            'served': False  # 标记该地点是否已被完全服务
        }
        # 更新资源类型集合
        for resource in resource_demands.keys():
            self.resources.add(resource)
    
    def add_uav(self, uav_id, resources_carried, speed=1.0):
        """
        添加无人机及其携带的资源
        :param uav_id: 无人机ID
        :param resources_carried: 字典，键为资源类型，值为携带数量
        :param speed: 无人机飞行速度
        """
        self.uavs[uav_id] = {
            'resources_carried': resources_carried,
            'remaining_resources': resources_carried.copy(),  # 用于跟踪剩余资源
            'speed': speed,
            'current_position': self.base_location,
            'path': [self.base_location],  # 路径记录
            'service_time': 0,  # 服务时间
            'flight_time': 0,  # 飞行时间
            'total_time': 0,  # 总时间
            'served_locations': []  # 已服务的地点
        }
    
    def calculate_distances(self):
        """
        计算所有地点之间的距离（包括基地）
        """
        # 添加基地到距离计算中
        all_positions = {-1: self.base_location}
        for loc_id, loc_info in self.locations.items():
            all_positions[loc_id] = loc_info['position']
        
        # 计算所有位置之间的欧几里得距离
        for id1 in all_positions:
            if id1 not in self.distances:
                self.distances[id1] = {}
            for id2 in all_positions:
                if id1 != id2:
                    pos1 = all_positions[id1]
                    pos2 = all_positions[id2]
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    self.distances[id1][id2] = dist
        
        # 确保所有地点ID都在距离矩阵中
        for id1 in all_positions:
            for id2 in all_positions:
                if id1 != id2 and (id1 not in self.distances or id2 not in self.distances[id1]):
                    pos1 = all_positions[id1]
                    pos2 = all_positions[id2]
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if id1 not in self.distances:
                        self.distances[id1] = {}
                    self.distances[id1][id2] = dist
    
    def can_uav_serve_location(self, uav_id, location_id):
        """
        检查无人机是否能够服务某个地点（至少部分满足其资源需求）
        """
        uav = self.uavs[uav_id]
        location = self.locations[location_id]
        
        if location['served']:
            return False
        
        for resource, demand in location['remaining_demands'].items():
            if demand > 0 and uav['remaining_resources'].get(resource, 0) > 0:
                return True
        return False
    
    def serve_location(self, uav_id, location_id):
        """
        无人机服务地点，更新资源和需求
        :return: 服务时间
        """
        uav = self.uavs[uav_id]
        location = self.locations[location_id]
        service_time = 0
        resources_provided = {}
        
        # 计算可以提供的资源
        for resource, demand in location['remaining_demands'].items():
            if demand > 0 and resource in uav['remaining_resources']:
                provided = min(demand, uav['remaining_resources'][resource])
                if provided > 0:
                    resources_provided[resource] = provided
                    location['remaining_demands'][resource] -= provided
                    uav['remaining_resources'][resource] -= provided
                    service_time += provided * 10  # 每单位资源服务时间为10秒
        
        # 检查地点是否已完全服务
        location['served'] = all(demand <= 0 for demand in location['remaining_demands'].values())
        
        # 更新无人机信息
        if service_time > 0:
            uav['service_time'] += service_time
            if location_id not in uav['served_locations']:
                uav['served_locations'].append(location_id)
        
        return service_time, resources_provided
    
    def move_uav(self, uav_id, target_location_id):
        """
        移动无人机到目标地点
        :return: 飞行时间
        """
        uav = self.uavs[uav_id]
        current_pos = uav['current_position']
        
        # 确定当前位置的ID
        current_id = -1  # 默认为基地
        for loc_id, loc_info in self.locations.items():
            if loc_info['position'] == current_pos:
                current_id = loc_id
                break
        
        # 计算飞行时间
        if target_location_id == -1:  # 返回基地
            target_pos = self.base_location
        else:
            target_pos = self.locations[target_location_id]['position']
        
        # 直接计算距离，不依赖于distances字典
        distance = np.sqrt((current_pos[0] - target_pos[0])**2 + (current_pos[1] - target_pos[1])**2)
        flight_time = distance / uav['speed']
        
        # 更新无人机位置和路径
        uav['current_position'] = target_pos
        uav['path'].append(target_pos)
        uav['flight_time'] += flight_time
        
        # 更新距离矩阵
        if current_id not in self.distances:
            self.distances[current_id] = {}
        self.distances[current_id][target_location_id] = distance
        
        if target_location_id not in self.distances:
            self.distances[target_location_id] = {}
        self.distances[target_location_id][current_id] = distance
        
        return flight_time
    
    def check_all_locations_served(self):
        """
        检查是否所有地点都已被完全服务
        :return: 布尔值，表示是否所有地点都已被服务
        """
        return all(loc['served'] for loc in self.locations.values())
    
    def reset_uavs_resources(self):
        """
        重置所有无人机的资源到初始状态
        """
        for uav_id, uav_info in self.uavs.items():
            uav_info['remaining_resources'] = uav_info['resources_carried'].copy()
            # 如果无人机不在基地，将其移回基地
            if uav_info['current_position'] != self.base_location:
                flight_time = self.move_uav(uav_id, -1)  # 返回基地
                uav_info['total_time'] += flight_time
    
    def heuristic_algorithm(self):
        """
        启发式算法实现路径规划
        使用基于资源匹配度和距离的启发式函数
        """
        # 计算所有地点之间的距离
        self.calculate_distances()
        
        # 最大迭代次数，防止无限循环
        max_iterations = 10
        current_iteration = 0
        
        while not self.check_all_locations_served() and current_iteration < max_iterations:
            current_iteration += 1
            
            # 重置所有地点的服务状态和剩余需求（如果是第一次迭代，则不需要重置）
            if current_iteration > 1:
                for loc_id, loc_info in self.locations.items():
                    if not loc_info['served']:
                        print(f"重新规划地点 {loc_id} 的服务")
            
            # 重置所有无人机的资源（如果是第一次迭代，则不需要重置）
            if current_iteration > 1:
                self.reset_uavs_resources()
            
            # 为每个无人机创建任务队列
            task_queues = {uav_id: [] for uav_id in self.uavs}
            
            # 计算每个地点的资源需求总量
            location_total_demands = {}
            for loc_id, loc_info in self.locations.items():
                if not loc_info['served']:  # 只考虑未服务的地点
                    total_demand = sum(loc_info['remaining_demands'].values())
                    location_total_demands[loc_id] = total_demand
            
            # 如果所有地点都已被服务，跳出循环
            if not location_total_demands:
                break
            
            # 计算每个无人机和地点之间的资源匹配度
            resource_matching = {}
            for uav_id, uav_info in self.uavs.items():
                resource_matching[uav_id] = {}
                for loc_id, loc_info in self.locations.items():
                    if not loc_info['served']:  # 只考虑未服务的地点
                        matching_score = 0
                        for resource, demand in loc_info['remaining_demands'].items():
                            if demand > 0 and resource in uav_info['remaining_resources']:
                                matching_score += min(demand, uav_info['remaining_resources'][resource])
                        resource_matching[uav_id][loc_id] = matching_score
            
            # 根据资源匹配度和距离分配任务
            unassigned_locations = [loc_id for loc_id, loc_info in self.locations.items() if not loc_info['served']]
            
            while unassigned_locations:
                best_assignment = None
                best_score = -float('inf')
                
                for loc_id in unassigned_locations:
                    for uav_id in self.uavs:
                        # 计算匹配分数（资源匹配度 / 距离）
                        matching = resource_matching[uav_id][loc_id]
                        if matching > 0:
                            # 计算从基地到该地点的距离
                            distance = self.distances[-1][loc_id]
                            score = matching / distance if distance > 0 else float('inf')
                            
                            if score > best_score:
                                best_score = score
                                best_assignment = (uav_id, loc_id)
                
                if best_assignment:
                    uav_id, loc_id = best_assignment
                    task_queues[uav_id].append(loc_id)
                    unassigned_locations.remove(loc_id)
                else:
                    # 如果没有找到合适的分配，跳出循环
                    break
            
            # 执行任务队列
            for uav_id, queue in task_queues.items():
                uav = self.uavs[uav_id]
                
                for loc_id in queue:
                    # 移动到目标地点
                    flight_time = self.move_uav(uav_id, loc_id)
                    uav['total_time'] += flight_time
                    
                    # 服务该地点
                    service_time, resources_provided = self.serve_location(uav_id, loc_id)
                    uav['total_time'] += service_time
                    
                    print(f"UAV {uav_id} 服务地点 {loc_id}，提供资源: {resources_provided}，服务时间: {service_time}秒")
                
                # 返回基地
                if uav['current_position'] != self.base_location:
                    flight_time = self.move_uav(uav_id, -1)  # 返回基地
                    uav['total_time'] += flight_time
        
        # 检查是否所有地点都已被服务
        if not self.check_all_locations_served():
            print("警告：启发式算法未能服务所有地点，可能需要调整无人机资源配置或算法策略。")
            # 打印未服务的地点信息
            for loc_id, loc_info in self.locations.items():
                if not loc_info['served']:
                    print(f"  地点 {loc_id}: 剩余需求 {loc_info['remaining_demands']}")
    
    def random_algorithm(self):
        """
        纯随机策略实现路径规划
        完全随机选择无人机和地点进行服务，持续迭代直到所有地点都被服务完成
        """
        # 计算所有地点之间的距离
        self.calculate_distances()
        
        # 所有地点是否都已被服务
        all_served = False
        
        # 迭代计数
        current_iteration = 0
        
        # 最大迭代次数，防止无限循环
        max_iterations = 10000
        
        # 持续迭代直到所有地点都被服务完成或达到最大迭代次数
        while not all_served and current_iteration < max_iterations:
            current_iteration += 1
            print(f"当前迭代次数: {current_iteration}")
            
            # 检查是否所有地点都已被服务
            all_served = self.check_all_locations_served()
            if all_served:
                break
            
            # 获取所有未服务的地点
            unserved_locations = [loc_id for loc_id, loc_info in self.locations.items() if not loc_info['served']]
            
            # 如果没有未服务的地点，跳出循环
            if not unserved_locations:
                break
            
            # 获取所有可用的无人机（有剩余资源的无人机）
            available_uavs = []
            for uav_id, uav_info in self.uavs.items():
                if any(res > 0 for res in uav_info['remaining_resources'].values()):
                    available_uavs.append(uav_id)
            
            # 如果没有可用的无人机，重置所有无人机的资源
            if not available_uavs:
                print("所有无人机资源耗尽，重置资源")
                self.reset_uavs_resources()
                available_uavs = list(self.uavs.keys())
            
            # 如果迭代次数超过300次，按照无人机ID顺序依次派遣无人机去服务未服务的地点
            if current_iteration > 300:
                print(f"\n===== 迭代次数超过300次，切换到顺序派遣模式 =====")
                # 按照无人机ID顺序排序
                sorted_uavs = sorted(available_uavs)
                
                for uav_id in sorted_uavs:
                    uav = self.uavs[uav_id]
                    
                    # 遍历所有未服务的地点
                    for loc_id in unserved_locations:
                        # 检查无人机是否能够服务该地点
                        if self.can_uav_serve_location(uav_id, loc_id):
                            # 如果无人机不在目标地点，先移动到那里
                            if uav['current_position'] != self.locations[loc_id]['position']:
                                flight_time = self.move_uav(uav_id, loc_id)
                                uav['total_time'] += flight_time
                            
                            # 服务该地点
                            service_time, resources_provided = self.serve_location(uav_id, loc_id)
                            uav['total_time'] += service_time
                            
                            # 打印服务信息
                            if resources_provided:
                                print(f"顺序模式: UAV {uav_id} 服务地点 {loc_id}，提供资源: {resources_provided}，服务时间: {service_time}秒")
                            else:
                                print(f"顺序模式: UAV {uav_id} 无法为地点 {loc_id} 提供任何资源")
                
                # 重新检查是否所有地点都已被服务
                all_served = self.check_all_locations_served()
                if all_served:
                    print("顺序派遣模式成功服务所有地点！")
                    break
                else:
                    # 如果仍有未服务的地点，重置资源继续尝试
                    print("顺序派遣一轮后仍有未服务的地点，重置资源继续尝试")
                    self.reset_uavs_resources()
            else:
                # 随机选择一个无人机和一个地点
                random_uav_id = available_uavs[np.random.randint(0, len(available_uavs))]
                random_loc_id = unserved_locations[np.random.randint(0, len(unserved_locations))]
                uav = self.uavs[random_uav_id]
                
                # 如果无人机不在目标地点，先移动到那里
                if uav['current_position'] != self.locations[random_loc_id]['position']:
                    flight_time = self.move_uav(random_uav_id, random_loc_id)
                    uav['total_time'] += flight_time
                
                # 服务该地点
                service_time, resources_provided = self.serve_location(random_uav_id, random_loc_id)
                uav['total_time'] += service_time
                
                # 打印服务信息
                if resources_provided:
                    print(f"UAV {random_uav_id} 服务地点 {random_loc_id}，提供资源: {resources_provided}，服务时间: {service_time}秒")
                else:
                    print(f"UAV {random_uav_id} 无法为地点 {random_loc_id} 提供任何资源")
            
            # 每隔50次迭代，打印当前未服务地点的状态
            if current_iteration % 50 == 0:
                print(f"\n===== 迭代 {current_iteration} 次后的状态 =====")
                for loc_id, loc_info in self.locations.items():
                    if not loc_info['served']:
                        print(f"  地点 {loc_id}: 剩余需求 {loc_info['remaining_demands']}")
        
        # 确保所有无人机都返回基地
        for uav_id in self.uavs:
            uav = self.uavs[uav_id]
            if uav['current_position'] != self.base_location:
                flight_time = self.move_uav(uav_id, -1)  # 返回基地
                uav['total_time'] += flight_time
        
        print(f"\n===== 随机算法完成 =====")
        print(f"总迭代次数: {current_iteration}")
        
        # 检查是否所有地点都已被服务
        if self.check_all_locations_served():
            print("成功：随机算法已成功服务所有地点！")
        else:
            print("警告：随机算法未能服务所有地点，可能是因为无人机携带的资源总量不足以满足所有地点的需求。")
            # 打印未服务的地点信息
            for loc_id, loc_info in self.locations.items():
                if not loc_info['served']:
                    print(f"  地点 {loc_id}: 剩余需求 {loc_info['remaining_demands']}")
        
    def visualize_paths(self, algorithm_name=""):
        """
        可视化无人机路径
        :param algorithm_name: 算法名称，用于保存不同算法的图片
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制地点
        for loc_id, loc_info in self.locations.items():
            x, y = loc_info['position']
            plt.scatter(x, y, c='blue', s=100)
            plt.text(x+0.1, y+0.1, f'地点{loc_id}')
        
        # 绘制基地
        base_x, base_y = self.base_location
        plt.scatter(base_x, base_y, c='black', s=150, marker='*')
        plt.text(base_x+0.1, base_y+0.1, '基地')
        
        # 绘制无人机路径
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
        for i, (uav_id, uav_info) in enumerate(self.uavs.items()):
            path = uav_info['path']
            color = colors[i % len(colors)]
            
            # 绘制路径线
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, c=color, linestyle='-', linewidth=2, label=f'UAV {uav_id}')
            
            # 绘制路径箭头
            for j in range(len(path)-1):
                dx = path[j+1][0] - path[j][0]
                dy = path[j+1][1] - path[j][1]
                plt.arrow(path[j][0], path[j][1], dx*0.8, dy*0.8, 
                          head_width=0.2, head_length=0.3, fc=color, ec=color, alpha=0.7)
        
        plt.legend()
        plt.grid(True)
        plt.title(f'无人机路径规划可视化 - {algorithm_name}' if algorithm_name else '无人机路径规划可视化')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        
        # 根据算法名称保存不同的图片文件
        if algorithm_name:
            filename = f'uav_paths_{algorithm_name}.png'
        else:
            filename = 'uav_paths.png'
        plt.savefig(filename)
        plt.close()  # 关闭图形，而不是显示，避免阻塞
    
    def print_statistics(self):
        """
        打印统计信息
        """
        print("\n===== 统计信息 =====")
        total_flight_time = 0
        total_service_time = 0
        total_time = 0
        
        for uav_id, uav_info in self.uavs.items():
            flight_time = uav_info['flight_time']
            service_time = uav_info['service_time']
            time = uav_info['total_time']
            served_locations = uav_info['served_locations']
            
            print(f"UAV {uav_id}:")
            print(f"  飞行时间: {flight_time:.2f}秒")
            print(f"  服务时间: {service_time}秒")
            print(f"  总时间: {time:.2f}秒")
            print(f"  服务地点: {served_locations}")
            
            total_flight_time += flight_time
            total_service_time += service_time
            total_time += time
        
        print("\n总计:")
        print(f"  总飞行时间: {total_flight_time:.2f}秒")
        print(f"  总服务时间: {total_service_time}秒")
        print(f"  总时间: {total_time:.2f}秒")
        
        # 检查是否所有地点都已被服务
        all_served = all(loc['served'] for loc in self.locations.values())
        print(f"\n所有地点是否已被服务: {'是' if all_served else '否'}")
        
        if not all_served:
            print("未被服务的地点:")
            for loc_id, loc_info in self.locations.items():
                if not loc_info['served']:
                    print(f"  地点 {loc_id}: 剩余需求 {loc_info['remaining_demands']}")