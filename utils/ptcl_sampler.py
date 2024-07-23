import json
import numpy as np
from typing import Union
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(os.path.dirname(__file__), 'constants.json'), 'r') as f:
    constants = json.load(f)

k          = constants['k'] # Boltzmann constant
Ru         = constants['Ru'] # universal gas constant
m_1_12_C12 = constants['m_1_12_C12'] # mass of 1/12 of C12


def maxwell_boltzmann_dist(v: float, temperature: float, molar_mass: float) -> float:
    '''
    Maxwell-Boltzmann distribution function

    Parameters:
    - v: velocity (m/s)
    - temperature: temperature of the system (K)
    - molar_mass: molar mass of the particle (g)

    Returns:
    - float, the value of Maxwell-Boltzmann distribution function
    '''
    return 4.0 * np.pi * ((molar_mass / (2 * np.pi * Ru * temperature)) ** 1.5) * (v ** 2) * np.exp(-(molar_mass * (v ** 2)) / (2 * Ru * temperature))


def maxwell_boltzmann_dist_torch(v: torch.Tensor, temperature: float, molar_mass: float) -> torch.Tensor:
    '''
    Maxwell-Boltzmann distribution function for PyTorch tensors

    Parameters:
    - v: velocity (m/s)
    - temperature: temperature of the system (K)
    - molar_mass: molar mass of the particle (g)

    Returns:
    - torch.Tensor, the value of Maxwell-Boltzmann distribution function
    '''
    return 4.0 * torch.pi * ((molar_mass / (2 * torch.pi * Ru * temperature)) ** 1.5) * (v ** 2) * torch.exp(-(molar_mass * (v ** 2)) / (2 * Ru * temperature))


def sample_maxwell_boltzmann_dist(temperature: float, molar_mass: float, sample_count: int, sample_batch_size = 10000) -> np.ndarray:
    '''
    Sample from Maxwell-Boltzmann distribution for PyTorch tensors

    Parameters:
    - temperature: temperature of the system (K)
    - molar_mass: molar mass of the particle (g)
    - count: number of samples to generate

    Returns:
    - torch.Tensor, samples from Maxwell-Boltzmann distribution
    '''
    # Define the maximum value of the PDF for the given parameters
    v_max = torch.sqrt(torch.Tensor([2 * Ru * temperature / molar_mass])).to(device).detach()
    f_max = maxwell_boltzmann_dist_torch(torch.tensor(v_max, device=device), temperature, molar_mass)
    
    samples = []
    
    while len(samples) < sample_count:
        v = torch.rand(sample_batch_size, device=device) * 5 * v_max  # Generate candidate velocities
        f_v = maxwell_boltzmann_dist_torch(v, temperature, molar_mass)  # Calculate the PDF value at this velocity
        u = torch.rand(sample_batch_size, device=device) * f_max  # Generate a uniform random number
        mask = u < f_v
        samples.extend(v[mask].tolist())

    # trim if too many samples
    if len(samples) > sample_count:
        samples = samples[:sample_count]
    
    return np.array(samples)


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
    sampled_v = sample_maxwell_boltzmann_dist(temperature, relative_mass / 1000, ptcl_count)
    sampled_v[sampled_v > max_velocity] = max_velocity
    sampled_angle = np.random.uniform(0, 2 * np.pi, ptcl_count)
    sampled_v_x = sampled_v * np.cos(sampled_angle) + flow_velocity
    sampled_v_y = sampled_v * np.sin(sampled_angle)
    ptcl_v = np.array([sampled_v_x, sampled_v_y]).T

    v_thermal_rms = np.sqrt(3 * Ru * temperature / (relative_mass / 1000))
    print(f'rms thermal velocity: {v_thermal_rms}')

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