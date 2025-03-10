import numpy as np

class EnergyModel:
    def __init__(self, params: 'DroneEnergyParams'):
        self.params = params
        self.air_density = 1.225
        
    def calculate_hover_power(self, payload: float = 0) -> float:
        total_mass = self.params.mass + payload
        thrust = total_mass * 9.81
        return (thrust ** 1.5) / (2 * self.air_density * self.params.rotor_area) ** 0.5
    
    def calculate_flight_power(self, velocity: float, payload: float = 0) -> float:
        hover_power = self.calculate_hover_power(payload)
        forward_power = self.params.blade_drag_coef * (velocity ** 3)
        return hover_power + forward_power
    
    def calculate_energy_consumption(self, 
                                  distance: float, 
                                  velocity: float,
                                  hover_time: float,
                                  payload: float = 0) -> float:
        flight_time = distance / velocity if velocity > 0 else 0
        flight_power = self.calculate_flight_power(velocity, payload)
        hover_power = self.calculate_hover_power(payload)
        
        return (flight_power * flight_time + hover_power * hover_time) / 3600