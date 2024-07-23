import json
import numpy as np
import matplotlib.pyplot as plt
from utils.ptcl_sampler import ptcl_sampler, maxwell_boltzmann_dist

with open('./config/config.json', 'r') as f:
    config = json.load(f)

T0 = config['T0']
ptcl_relative_mass = config['ptcl_relative_mass']

ptcl_pos, ptcl_v = ptcl_sampler(
    ptcl_count    = 10000,
    map_size      = [100, 100],
    temperature   = T0,
    relative_mass = ptcl_relative_mass,
    max_velocity  = np.inf,
    flow_velocity = 0
)

# Calculate the magnitude of ptcl_v
ptcl_v_mag = np.linalg.norm(ptcl_v, axis=1)

sample_mean    = np.mean(ptcl_v_mag)
sample_rms     = np.sqrt(np.mean(ptcl_v_mag**2))
speed_of_sound = sample_rms * np.sqrt(1.4 / 3)

print(f'Sample mean velocity: {sample_mean}')
print(f'Sample rms  velocity: {sample_rms}')
print(f'Approximated speed of sound: {speed_of_sound}')

# Plot the histogram
plt.hist(ptcl_v_mag, bins=50)

# Plot the Maxwell-Boltzmann Distribution
v = np.linspace(0, 2000, 1000)
plt.plot(v, maxwell_boltzmann_dist(v, T0, ptcl_relative_mass / 1000) * 10000)
plt.legend(['Maxwell-Boltzmann Distribution', 'Histogram'])
plt.grid(True)

plt.xlabel('v')
plt.ylabel('Frequency')
plt.title('Histogram of Speed Distribution')
plt.show()