from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class Resource:
    type_id: str
    amount: int

@dataclass
class TimeWindow:
    start_time: float
    end_time: float

@dataclass
class Location:
    id: int
    position: Tuple[float, float, float]
    demands: Dict[str, int]
    time_window: TimeWindow
    service_time: float
    is_served: bool = False

@dataclass
class DroneEnergyParams:
    mass: float
    rotor_area: float
    battery_capacity: float
    hover_power: float
    max_speed: float
    blade_drag_coef: float

class UAV:
    def __init__(self, 
                 id: int,
                 position: Tuple[float, float, float],
                 resources: Dict[str, int],
                 energy_params: DroneEnergyParams):
        self.id = id
        self.position = position
        self.resources = resources.copy()
        self.initial_resources = resources.copy()
        self.energy_params = energy_params
        self.remaining_energy = energy_params.battery_capacity
        self.velocity = 0.0
        self.path = []
        
    def can_serve(self, location: Location) -> bool:
        return all(
            self.resources.get(r_type, 0) >= amount 
            for r_type, amount in location.demands.items()
        )
    
    def serve_location(self, location: Location) -> bool:
        if not self.can_serve(location):
            return False
        
        for r_type, amount in location.demands.items():
            self.resources[r_type] -= amount
        return True