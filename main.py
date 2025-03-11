# 无人机辅助移动边缘计算路径规划测试

from uav_path_planning import UAVPathPlanning
import time
import matplotlib.pyplot as plt
import numpy as np
from resource_validator import validate_resources, print_resource_validation_results, suggest_resource_adjustments, print_adjustment_suggestions

def test_algorithm(algorithm_name, planner):
    """测试指定的算法并返回总时间"""
    print(f"\n===== 测试{algorithm_name} =====")
    start_time = time.time()
    
    # 根据算法名称调用相应的方法
    if algorithm_name == "启发式算法":
        planner.heuristic_algorithm()
    elif algorithm_name == "随机策略":
        planner.random_algorithm()
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    # 打印统计信息
    planner.print_statistics()
    print(f"计算时间: {computation_time:.4f}秒")
    
    # 可视化路径，传入算法名称
    planner.visualize_paths(algorithm_name)
    
    # 计算总时间
    total_time = 0
    for uav_id, uav_info in planner.uavs.items():
        total_time += uav_info['total_time']
    
    return total_time

def setup_scenario():
    """设置测试场景"""
    planner = UAVPathPlanning()
    
    # 添加地点及其资源需求
    # 地点1需要1个资源a和1个资源b
    planner.add_location(1, (2, 3), {'a': 1, 'b': 1})
    # 地点2需要1个资源a和1个资源c
    planner.add_location(2, (5, 1), {'a': 1, 'c': 1})
    # 地点3需要2个资源a和1个资源c和2个资源d
    planner.add_location(3, (8, 4), {'a': 2, 'c': 1, 'd': 2})
    # 地点4需要1个资源b和1个资源d
    planner.add_location(4, (6, 7), {'b': 1, 'd': 1})
    # 地点5需要2个资源c和1个资源d
    planner.add_location(5, (3, 6), {'c': 2, 'd': 1})
    
    # 添加无人机及其携带的资源
    # 无人机1携带2个资源a、1个资源b、3个资源c
    planner.add_uav(1, {'a': 2, 'b': 1, 'c': 3}, speed=1.0)
    # 无人机2携带2个资源a、1个资源b、4个资源d
    planner.add_uav(2, {'a': 2, 'b': 1, 'd': 4}, speed=1.2)
    
    # 不再进行资源验证和自动调整
    
    return planner

def adjust_resources(planner, suggestions):
    """根据建议自动调整无人机资源
    :param planner: UAVPathPlanning实例
    :param suggestions: 资源调整建议
    """
    print("\n===== 自动调整资源 =====")
    
    # 为每种资源不足的情况，选择一个合适的无人机增加资源
    for resource, amount_needed in suggestions.items():
        print(f"需要增加资源 {resource}: {amount_needed} 个")
        
        # 找出已经携带该资源的无人机
        carriers = []
        for uav_id, uav_info in planner.uavs.items():
            if resource in uav_info['resources_carried']:
                carriers.append(uav_id)
        
        if carriers:
            # 选择第一个携带该资源的无人机增加资源
            uav_id = carriers[0]
            planner.uavs[uav_id]['resources_carried'][resource] += amount_needed
            planner.uavs[uav_id]['remaining_resources'][resource] += amount_needed
            print(f"  - 增加无人机 {uav_id} 的资源 {resource}: +{amount_needed} 个")
        else:
            # 如果没有无人机携带该资源，选择第一个无人机添加该资源
            uav_id = list(planner.uavs.keys())[0]
            if resource not in planner.uavs[uav_id]['resources_carried']:
                planner.uavs[uav_id]['resources_carried'][resource] = amount_needed
                planner.uavs[uav_id]['remaining_resources'][resource] = amount_needed
            else:
                planner.uavs[uav_id]['resources_carried'][resource] += amount_needed
                planner.uavs[uav_id]['remaining_resources'][resource] += amount_needed
            print(f"  - 为无人机 {uav_id} 添加资源 {resource}: {amount_needed} 个")

def main():
    """主函数"""
    print("无人机辅助移动边缘计算路径规划测试")
    
    # 测试不同算法
    algorithms = ["启发式算法", "随机策略"]
    results = {}
    
    for algorithm in algorithms:
        # 为每个算法创建新的场景实例
        planner = setup_scenario()
        total_time = test_algorithm(algorithm, planner)
        results[algorithm] = total_time
    
    # 比较不同算法的性能
    print("\n===== 算法性能比较 =====")
    for algorithm, total_time in results.items():
        print(f"{algorithm}: 总时间 {total_time:.2f}秒")
    
    # 绘制比较图表
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('不同算法的总服务时间比较')
    plt.ylabel('总时间（秒）')
    plt.savefig('algorithm_comparison.png')
    plt.close()  # 关闭图形，而不是显示，避免阻塞

if __name__ == "__main__":
    main()