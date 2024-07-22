import json
import numpy as np
import matplotlib.pyplot as plt
from utils.ptcl_sampler import ptcl_sampler

with open('./config/config.json', 'r') as f:
    config = json.load(f)

T0 = config['T0']
ptcl_relative_mass = config['ptcl_relative_mass']

ptcl_pos, ptcl_v = ptcl_sampler(
    ptcl_count    = 1000,
    map_size      = [100, 100],
    temperature   = T0,
    relative_mass = ptcl_relative_mass,
    max_velocity  = np.inf,
    flow_velocity = 0
)

# Calculate the magnitude of ptcl_v
ptcl_v_mag = np.linalg.norm(ptcl_v, axis=1)

sample_mean = np.mean(ptcl_v_mag)
print(f'Sample mean velocity: {sample_mean}')

# Plot the histogram
plt.hist(ptcl_v_mag, bins=20)
plt.xlabel('Magnitude of ptcl_v')
plt.ylabel('Frequency')
plt.title('Histogram of ptcl_v Magnitude')
plt.show()