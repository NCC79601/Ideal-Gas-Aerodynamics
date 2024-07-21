import json
import numpy as np
from typing import Union
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(os.path.dirname(__file__), 'constants.json'), 'r') as f:
    constants = json.load(f)

k = constants['k'] # Boltzmann constant
m_1_12_C12 = constants['m_1_12_C12'] # mass of 1/12 of C12

def ptcl_sampler(
    ptcl_count: int,
    map_size: Union[list, tuple],
    temperature: float,
    relative_mass: float,
    max_velocity: float,
    flow_velocity: float = 0.0
) -> tuple:
    '''
    function to sample particles, velocities from Maxwell-Boltzmann distribution, and positions from uniform distribution

    Parameters:
    - ptcl_count: number of particles to sample
    - map_size: size of the map [w, h]
    - temperature: temperature of the system
    - relative_mass: relative mass of the particles
    - max_velocity: maximum thermal velocity of the particles
    - flow_velocity: velocity of the flow

    Returns:
    - `ptcl_pos, ptcl_v`: sampled positions and velocities
    '''
    # sample position
    sampled_x = np.random.uniform(0, map_size[0], ptcl_count)
    sampled_y = np.random.uniform(0, map_size[1], ptcl_count)
    ptcl_pos = np.array([sampled_x, sampled_y]).T

    # sample velocity from Maxwell-Boltzmann distribution
    mass = relative_mass * m_1_12_C12
    scale = np.sqrt(k * temperature / mass)
    sampled_v = np.random.rayleigh(scale, ptcl_count)
    sampled_v[sampled_v > max_velocity] = max_velocity
    sampled_angle = np.random.uniform(0, 2 * np.pi, ptcl_count)
    sampled_v_x = sampled_v * np.cos(sampled_angle) + flow_velocity
    sampled_v_y = sampled_v * np.sin(sampled_angle)
    ptcl_v = np.array([sampled_v_x, sampled_v_y]).T

    v_thermal_mean = np.sqrt(2 * np.pi * k * temperature / mass)
    print(f'mean thermal velocity: {v_thermal_mean}')

    return ptcl_pos, ptcl_v


def ptcl_new_sampler(
    ptcl_count: int,
    ptcl_gen_area_size: Union[list, tuple],
    temperature: float,
    relative_mass: float,
    max_velocity: float,
    flow_velocity: float = 0.0
) -> tuple:
    '''
    function to generate new particles, using `ptcl_sampler` function

    Parameters:
    - ptcl_count: number of particles to sample
    - ptcl_gen_area_size: size of the area to generate particles [w, h]
    - temperature: temperature of the system
    - relative_mass: relative mass of the particles
    - max_velocity: maximum thermal velocity of the particles
    - flow_velocity: velocity of the flow

    Returns:
    - `ptcl_new_pos, ptcl_new_v, t_to_enter`: sampled positions and velocities, and time for each particle to enter the map
    '''
    ptcl_new_pos, ptcl_new_v = ptcl_sampler(
        ptcl_count,
        ptcl_gen_area_size,
        temperature,
        relative_mass,
        max_velocity,
        flow_velocity
    )
    ptcl_new_pos[:, 0] -= ptcl_gen_area_size[0]
    t_to_enter = np.abs(ptcl_new_pos[:, 0]) / flow_velocity

    # sort particles by time to enter
    sorted_indices = np.argsort(t_to_enter)
    ptcl_new_pos = ptcl_new_pos[sorted_indices]
    ptcl_new_v = ptcl_new_v[sorted_indices]
    t_to_enter = t_to_enter[sorted_indices]
    
    return ptcl_new_pos, ptcl_new_v, t_to_enter