import numpy as np
import torch
import json
import os
from tqdm import tqdm

try:
    from .ptcl_sampler import ptcl_sampler, ptcl_new_sampler
except ImportError:
    from ptcl_sampler import ptcl_sampler, ptcl_new_sampler

# use cuda to accelerate simulation if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Map:
    '''
    Map class contains all the particles and walls in the simulation
    '''
    def __init__(self, config_file_path: str = os.path.join(os.path.dirname(__file__), 'config', 'config.json')) -> None:
        '''
        Initialize map with config file

        Parameters:
        - config_file_path: path to the config file
        '''
        assert os.path.exists(config_file_path), f"Config file not found at {config_file_path}"

        # load config file
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        
        # parse config file
        # particle properties
        self.ptcl_radius        = config['ptcl_radius'] # particle radius
        self.ptcl_relative_mass = config['ptcl_relative_mass'] # relative mass of the particles
        # thermal properties
        self.T0              = config['T0'] # static temperature
        self.v0              = config['v0'] # flowing velocity
        self.v_thermal_max   = config['v_thermal_max'] # maximum thermal velocity
        self.spatial_density = config['spatial_density'] # spatial density (number of particles per unit area)
        # spatial geometric properties
        self.map_size           = config['map_size'] # map size [w, h]
        self.width, self.height = self.map_size[0], self.map_size[1]
        self.ptcl_gen_area_size = config['ptcl_gen_area_size'] # particle generation area size [w, h]
        self.ptcl_gen_y_range   = config['ptcl_gen_y_range'] # particle generation y range [y_min, y_max]
        self.walls              = config['walls'] # walls in the map [[[x1, y1], [x2, y2]], ...]
        # simulation time properties
        self.t_lim   = config['t_lim'] # simulation time limit
        self.t_scale = config['t_scale'] # time between frames (default 1/60 s)
        self.dt      = config['dt'] # time step

        # initialize current simulation time
        self.t = 0

        # initialize particles
        self.ptcl_count = int(self.spatial_density * self.map_size[0] * \
                              (self.ptcl_gen_y_range[1] - self.ptcl_gen_y_range[0]))
        print(f'Initializing map [{self.map_size[0]} x {self.map_size[1]}] with {self.ptcl_count} particles...')
        self.ptcl_pos, self.ptcl_v = ptcl_sampler(
            ptcl_count    = self.ptcl_count,
            map_size = [self.map_size[0], self.ptcl_gen_y_range[1] - self.ptcl_gen_y_range[0]],
            temperature   = self.T0,
            max_velocity  = self.v_thermal_max,
            relative_mass = self.ptcl_relative_mass,
            flow_velocity = self.v0
        )
        print(f'Particles sampled.')
        self.ptcl_pos[:, 1] += self.ptcl_gen_y_range[0] # shift particles to the correct y range
        # convert to torch tensor (use cuda if available)
        self.ptcl_pos = torch.tensor(self.ptcl_pos, dtype=torch.float32, device=device).detach() # no need for gradients
        self.ptcl_v   = torch.tensor(self.ptcl_v,   dtype=torch.float32, device=device).detach()
        
        # initialize particle generation area (refreshed in fixed time interval)
        self.ptcl_gen_count = int(self.spatial_density * self.ptcl_gen_area_size[0] * \
                                  (self.ptcl_gen_y_range[1] - self.ptcl_gen_y_range[0]))
        self._refresh_ptcl_gen_area() # fill the area with new generated particles
        print(f'Initial particle generation area created.')
        self.gen_area_refresh_interval = self.ptcl_gen_area_size[0] / self.v0 # time interval to refresh
        self.next_gen_t = self.gen_area_refresh_interval # next time to refresh

        # initialize key frames
        self.keyframes = [] # [{"t": t, "ptcl_pos": ptcl_pos, "ptcl_v": ptcl_v}, ...]

        print('Map initialization complete.')

    def _update_positions(self, elapsed_time: float) -> None:
        '''
        Update particle positions and delete outbounded particles

        Parameters:
        - elapsed_time: elapsed time of this update (usually till the next predicted collision)
        '''
        self.ptcl_pos += self.ptcl_v * elapsed_time
    
    def _refresh_ptcl_gen_area(self) -> None:
        '''
        Refresh particle generation area
        '''
        self.ptcl_new_pos, self.ptcl_new_v, self.ptcl_new_t_to_enter = ptcl_new_sampler(
            ptcl_count    = self.ptcl_gen_count,
            ptcl_gen_area_size = [self.ptcl_gen_area_size[0], self.ptcl_gen_y_range[1] - self.ptcl_gen_y_range[0]],
            temperature   = self.T0,
            relative_mass = self.ptcl_relative_mass,
            max_velocity  = self.v_thermal_max,
            flow_velocity = self.v0
        )
        self.ptcl_new_pos = torch.tensor(self.ptcl_new_pos, dtype=torch.float32, device=device)
        self.ptcl_new_pos[:, 1] += self.ptcl_gen_y_range[0] # shift particles to the correct y range
        self.ptcl_new_v = torch.tensor(self.ptcl_new_v, dtype=torch.float32, device=device)
        self.ptcl_new_t_enter = self.ptcl_new_t_to_enter + self.t
        self.next_enter_id = 0

    def simulate(self) -> None:
        '''
        Perform simulation
        '''
        # record first keyframe
        self.keyframes.append({
            "t": self.t,
            "ptcl_pos": self.ptcl_pos.cpu().numpy(),
            "ptcl_v": self.ptcl_v.cpu().numpy()
        })

        # simulation loop
        pbar = tqdm(total=self.t_lim)
        while self.t < self.t_lim:
            # record keyframes
            if self.t - self.keyframes[-1]['t'] >= self.t_scale:
                self.keyframes.append({
                    "t": self.t,
                    "ptcl_pos": self.ptcl_pos.cpu().numpy(),
                    "ptcl_v": self.ptcl_v.cpu().numpy()
                })
            
            # check if need to refresh particle generation area
            if self.t >= self.next_gen_t:
                self._refresh_ptcl_gen_area()
                self.next_gen_t += self.gen_area_refresh_interval
                print(f'Particle generation area refreshed @ t = {self.t:.2f}.')
            
            # check if new particles enter the map
            while self.next_enter_id < len(self.ptcl_new_t_enter) and self.ptcl_new_t_enter[self.next_enter_id] <= self.t:
                gen_ptcl_pos = self.ptcl_new_pos[self.next_enter_id]
                gen_ptcl_pos[0] = 0 # enter from left boundary
                self.ptcl_pos = torch.cat([
                    self.ptcl_pos,
                    gen_ptcl_pos.unsqueeze(0).to(device)
                ], dim=0)
                self.ptcl_v = torch.cat([
                    self.ptcl_v,
                    self.ptcl_new_v[self.next_enter_id].unsqueeze(0).to(device)
                ], dim=0)
                self.next_enter_id += 1
            
            # step forward time
            self.t += self.dt
            pbar.update(self.dt)
            
            # update particle positions
            self._update_positions(self.dt)

            # check for particles out of map range
            out_of_range  = self.ptcl_pos[:, 0] > self.map_size[0]
            self.ptcl_pos = self.ptcl_pos[~out_of_range]
            self.ptcl_v   = self.ptcl_v[~out_of_range]

            # check for collisions
            # particle-particle collisions
            for i in range(1, self.ptcl_pos.shape[0], 1):
                pos  = self.ptcl_pos[i, :].unsqueeze(0)
                other_ptcl_pos = self.ptcl_pos[:i, :]
                dis2 = torch.sum((other_ptcl_pos - pos) ** 2, dim=1)
                
                collide_id = torch.argmin(dis2)

                if dis2[collide_id] > 4 * self.ptcl_radius ** 2:
                    # no collision for particle i
                    continue

                # handle collision
                p1 = pos.squeeze(0).cpu().numpy()
                v1 = self.ptcl_v[i, :].cpu().numpy()
                p2 = self.ptcl_pos[collide_id, :].cpu().numpy()
                v2 = self.ptcl_v[collide_id, :].cpu().numpy()

                # reference: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
                v1_new = v1 - (np.dot(v1 - v2, p1 - p2) / np.linalg.norm(p1 - p2) ** 2) * (p1 - p2)
                v2_new = v2 - (np.dot(v2 - v1, p2 - p1) / np.linalg.norm(p2 - p1) ** 2) * (p2 - p1)

                # update velocities
                self.ptcl_v[i, :] = torch.tensor(v1_new).to(device)
                self.ptcl_v[collide_id, :] = torch.tensor(v2_new).to(device)

                # if particles are too close, move them apart
                if np.linalg.norm(p1 - p2) < 2 * self.ptcl_radius:
                    center = (p1 + p2) / 2
                    n = (p1 - p2) / np.linalg.norm(p1 - p2)
                    p1_new = center + self.ptcl_radius * n
                    p2_new = center - self.ptcl_radius * n
                    self.ptcl_pos[i, :] = torch.tensor(p1_new).to(device)
                    self.ptcl_pos[collide_id, :] = torch.tensor(p2_new).to(device)
            
            # particle-wall collisions
            for wall in self.walls:
                # compute normal and direction vector of the wall
                p1, p2 = wall[0], wall[1]
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])

                v_dir = np.array([x2 - x1, y2 - y1])

                n = np.array([y1 - y2, x2 - x1], dtype=float)
                n /= np.linalg.norm(n)

                # compute line equation coefficients (A * x + B * y + C = 0)
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2

                # compute distances to the wall
                dis = torch.abs(A * self.ptcl_pos[:, 0] + B * self.ptcl_pos[:, 1] + C) / np.sqrt(A**2 + B**2)
                
                # check if particles move toward the wall
                def cross(v_dir, batched_v):
                    return v_dir[0] * batched_v[:, 1] - v_dir[1] * batched_v[:, 0]
                direction_judge = cross(v_dir, self.ptcl_pos - torch.tensor([x1, y1]).to(device).unsqueeze(0)) * \
                    cross(v_dir, self.ptcl_v)
                moving_toward_line = direction_judge < 0

                # check if particles collide with the wall
                collide_with_line  = dis <= self.ptcl_radius

                # check if particles collide within the segment
                v1 = torch.tensor([x1, y1]).to(device).unsqueeze(0) - self.ptcl_pos
                v2 = torch.tensor([x2, y2]).to(device).unsqueeze(0) - self.ptcl_pos
                dot_product = torch.sum(v1 * v2, dim=1)
                collide_in_segment = dot_product <= 0

                collide_ids = torch.logical_and(
                    moving_toward_line,
                    torch.logical_and(
                        collide_with_line,
                        collide_in_segment
                    )
                )

                if not torch.any(collide_id):
                    # no collision for this wall
                    continue
                
                # handle collisions
                n_tensor = torch.tensor(n).unsqueeze(0).to(device)
                v_tensor = self.ptcl_v[collide_ids, :]
                v_new = v_tensor - 2 * torch.sum(v_tensor * n_tensor, dim=1).unsqueeze(1) * n_tensor

                self.ptcl_v[collide_ids, :] = v_new.type(torch.float32)
                
        
        pbar.close()
