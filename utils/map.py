import numpy as np
import torch
import json
import os
from tqdm import tqdm

try:
    from .ptcl_sampler import ptcl_sampler, ptcl_new_sampler
except ImportError:
    from ptcl_sampler import ptcl_sampler, ptcl_new_sampler

try:
    from .collide import is_collide_ptcl_ptcl, is_collide_ptcl_wall, collide_particle_particle, collide_particle_wall
except ImportError:
    from collide import is_collide_ptcl_ptcl, is_collide_ptcl_wall, collide_particle_particle, collide_particle_wall

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
        
        # parse config
        self.ptcl_radius = config['ptcl_radius'] # particle radius
        self.ptcl_relative_mass = config['ptcl_relative_mass'] # relative mass of the particles
        self.T0 = config['T0'] # static temperature
        self.v0 = config['v0'] # flowing velocity
        self.spatial_density = config['spatial_density'] # spatial density (number of particles per unit area)+
        +.3
        self.map_size = config['map_size'] # map size [w, h]
        self.width, self.height = self.map_size[0], self.map_size[1]
        self.ptcl_gen_area_size = config['ptcl_gen_area_size'] # particle generation area size [w, h]
        self.walls = config['walls'] # walls in the map [[[x1, y1], [x2, y2]], ...]
        self.t_lim = config['t_lim'] # simulation time limit
        self.t_scale = config['t_scale'] # time between frames (default 1/60 s)

        # initialize particles
        ptcl_count = int(self.spatial_density * self.map_size[0] * self.map_size[1])
        print(f'Initializing map [{self.map_size[0]} x {self.map_size[1]}] with {ptcl_count} particles...')
        self.ptcl_pos, self.ptcl_v = ptcl_sampler(ptcl_count, self.map_size, self.T0, self.ptcl_relative_mass, self.v0)
        
        # initialize particle generation area
        ptcl_new_count = int(self.spatial_density * self.ptcl_gen_area_size[0] * self.ptcl_gen_area_size[1])
        self.ptcl_new_pos, self.ptcl_new_v, self.t_to_enter = \
            ptcl_new_sampler(ptcl_new_count, self.ptcl_gen_area_size, self.T0, self.ptcl_relative_mass, self.v0)
        self.t_enter = self.t_to_enter
        self.next_enter_id = 0
        self.gen_area_refresh_interval = self.ptcl_gen_area_size[0] / self.v0 # time interval to refresh particle generation area
        self.next_gen_t = self.gen_area_refresh_interval # next time to refresh particle generation area

        # initialize collision detection
        self.ptcl_collide_t = np.ones(ptcl_count) * np.inf # collision time
        self.ptcl_collide_id = np.zeros(ptcl_count, dtype=int) # negative id for wall collision

        # initialize current simulation time
        self.t = 0

        # initialize last recorded time
        self.t_last_recorded = 0

        # initialize collision predictions
        print('Comupting initial collision predictions...')
        for i in tqdm(range(len(self.ptcl_pos))):
            self._update_collision_predictions(i)

        # initialize key frames
        self.key_frames = [] # [{"t": t, "ptcl_pos": ptcl_pos, "ptcl_v": ptcl_v}, ...]
        print('Map initialized.')

    def _update_positions(self, elapsed_time: float) -> None:
        '''
        Update particle positions and delete outbounded particles

        Parameters:
        - elapsed_time: elapsed time of this update (usually till the next predicted collision)
        '''
        self.ptcl_pos += self.ptcl_v * elapsed_time
        self.ptcl_pos[:, 1] = self.ptcl_pos[:, 1] % self.map_size[1]
    
    def _get_positions(self, t: float) -> np.ndarray:
        '''
        Get particle positions

        Parameters:
        - t: specified time

        Returns:
        - particle positions at time `t`
        '''
        return self.ptcl_pos + self.ptcl_v * (t - self.t)

    def _update_specific_collision_prediciton(self, ptcl_id: int, collide_t: float, ptcl_collide_id: int) -> None:
        '''
        Single-point collision prediction update for specific particle
        
        ** Does not need to be called explicitly **

        Parameters:
        - ptcl_id: id of particle to update (index from 1, to avoid ambiguity)
        - collide_t: collision time
        - ptcl_collide_id: collision id (negative for wall collision)

        Return:
        - True if collision time is updated, False otherwise
        '''
        if collide_t <= self.t:
            return False
        if collide_t < self.ptcl_collide_t[ptcl_id]:
            self.ptcl_collide_t[ptcl_id] = collide_t
            self.ptcl_collide_id[ptcl_id] = ptcl_collide_id
            return True
        return False

    def _update_collision_predictions(self, ptcl_id: int) -> None:
        '''
        Update collision predictions for particle `ptcl_id`
        
        ** Does not need to be called explicitly **

        Parameters:
        - ptcl_id: id of the particle to update (index from 0)
        '''
        # clear original collision predictions
        self.ptcl_collide_t[ptcl_id] = np.inf

        # update particle-particle collisions
        for i in range(len(self.ptcl_pos)):
            is_collide, t = is_collide_ptcl_ptcl(
                self.ptcl_pos[ptcl_id, :], self.ptcl_v[ptcl_id, :],
                self.ptcl_pos[i, :], self.ptcl_v[i, :],
                self.ptcl_radius
            )
            if is_collide:
                self._update_specific_collision_prediciton(ptcl_id, self.t + t, i + 1)
                self._update_specific_collision_prediciton(i, self.t + t, ptcl_id + 1)
        
        # update particle-wall collisions
        for i, wall in enumerate(self.walls):
            is_collide, t = is_collide_ptcl_wall(
                self.ptcl_pos[ptcl_id, :], self.ptcl_v[ptcl_id, :],
                wall[0], wall[1],
                self.ptcl_radius
            )
            if is_collide:
                self._update_specific_collision_prediciton(ptcl_id, self.t + t, -i - 1)

    def _collide_particle_particle(self, ptcl_id1: int, ptcl_id2: int) -> None:
        '''
        Perform collision between particles

        Parameters:
        - ptcl_id1: id of the first particle
        - ptcl_id2: id of the second particle
        '''
        # print(f' > Colliding particle {ptcl_id1} and particle {ptcl_id2}')
        # print(f' > before collision: ptcl_v1: {self.ptcl_v[ptcl_id1]}, ptcl_v2: {self.ptcl_v[ptcl_id2]}')
        self.ptcl_v[ptcl_id1], self.ptcl_v[ptcl_id2] = collide_particle_particle(
            self.ptcl_pos[ptcl_id1, :], self.ptcl_v[ptcl_id1, :],
            self.ptcl_pos[ptcl_id2, :], self.ptcl_v[ptcl_id2, :]
        )
        # print(f' > after collision: ptcl_v1: {self.ptcl_v[ptcl_id1]}, ptcl_v2: {self.ptcl_v[ptcl_id2]}')
        # print(f' > before update, ptcl_id1 collide id: {self.ptcl_collide_id[ptcl_id1]}, ptcl_id2 collide id: {self.ptcl_collide_id[ptcl_id2]}')
        self._update_collision_predictions(ptcl_id1)
        self._update_collision_predictions(ptcl_id2)
        # print(f' > after update, ptcl_id1 collide id: {self.ptcl_collide_id[ptcl_id1]}, ptcl_id2 collide id: {self.ptcl_collide_id[ptcl_id2]}')

    
    def _collide_particle_wall(self, ptcl_id: int, wall_id: int) -> None:
        '''
        Perform collision between particle `ptcl_id` and wall `wall_id`

        Parameters:
        - ptcl_id: id of the particle
        - wall_id: id of the wall
        '''
        self.ptcl_v[ptcl_id, :] = collide_particle_wall(
            self.ptcl_v[ptcl_id, :],
            self.walls[wall_id][0], self.walls[wall_id][1]
        )
        self._update_collision_predictions(ptcl_id)

    def _refresh_ptcl_gen_area(self) -> None:
        '''
        Refresh particle generation area
        '''
        ptcl_new_count = int(self.spatial_density * self.ptcl_gen_area_size[0] * self.ptcl_gen_area_size[1])
        self.ptcl_new_pos, self.ptcl_new_v, self.t_to_enter = \
            ptcl_new_sampler(ptcl_new_count, self.ptcl_gen_area_size, self.T0, self.ptcl_relative_mass, self.v0)
        self.t_enter = self.t + self.t_to_enter
        self.next_enter_id = 0
        self.next_gen_t = self.t + self.gen_area_refresh_interval

    def _append_new_particle(self, ptcl_pos: np.ndarray, ptcl_v: np.ndarray) -> None:
        '''
        Append new particles to the map

        Parameters:
        - ptcl_pos: new particle positions
        - ptcl_v: new particle velocities
        '''
        self.ptcl_pos = np.vstack([self.ptcl_pos, ptcl_pos])
        self.ptcl_v   = np.vstack([self.ptcl_v, ptcl_v])
        self.ptcl_collide_t  = np.append(self.ptcl_collide_t, np.inf)
        self.ptcl_collide_id = np.append(self.ptcl_collide_id, 0)

    def simulate(self):
        '''
        Perform simulation
        '''
        # record the first frame
        self.key_frames.append({
            "t": 0,
            "ptcl_pos": self.ptcl_pos,
            "ptcl_v": self.ptcl_v,
            "ptcl_collide_t": self.ptcl_collide_t,
        })

        # next state of simulation FSM
        sim_next_state = ''

        # calculate the next collision time
        self.next_ptcl_collide_t  = np.min(self.ptcl_collide_t)
        self.next_ptcl_collide_id = np.argmin(self.ptcl_collide_t)

        # calculate the next new particle enter time
        next_enter_t = self.t_enter[self.next_enter_id]

        self.next_gen_t

        # next simulation state time
        next_t = 0

        # construct next state of simulation FSM
        if self.next_ptcl_collide_t < next_enter_t and self.next_ptcl_collide_t < self.next_gen_t:
            sim_next_state = 'collide'
            next_t = self.next_ptcl_collide_t
        elif next_enter_t < self.next_ptcl_collide_t and next_enter_t < self.next_gen_t:
            sim_next_state = 'enter'
            next_t = next_enter_t
        else:
            sim_next_state = 'gen'
            next_t = self.next_gen_t

        # simulation loop (finite state machine)
        print('Running simulation...')

        pbar = tqdm(total=self.t_lim)
        
        while self.t < self.t_lim:
            # print(f'Simulation time: {self.t} s, next state: {sim_next_state}, next time: {next_t} s')
            # FIXME:
            # record key frames
            while next_t - self.t_last_recorded >= self.t_scale:
                self.key_frames.append({
                    "t": self.t_last_recorded + self.t_scale,
                    "ptcl_pos": self._get_positions(self.t_last_recorded + self.t_scale),
                    "ptcl_v": self.ptcl_v,
                })
                self.t_last_recorded = self.t_last_recorded + self.t_scale

            # update positions
            self._update_positions(next_t - self.t)
            
            # update simulation time
            self.t = next_t

            if sim_next_state == 'collide':
                # perform collision
                # HACK: should not!
                if self.ptcl_collide_id[self.next_ptcl_collide_id] < 0:
                    # particle - wall collision
                    self._collide_particle_wall(self.next_ptcl_collide_id, -int(self.ptcl_collide_id[self.next_ptcl_collide_id]) - 1)
                    self.next_ptcl_collide_t = np.min(self.ptcl_collide_t)
                    self.next_ptcl_collide_id = np.argmin(self.ptcl_collide_t)
                else:
                    # particle - particle collision
                    self._collide_particle_particle(self.next_ptcl_collide_id, int(self.ptcl_collide_id[self.next_ptcl_collide_id]) - 1)
                    self.next_ptcl_collide_t = np.min(self.ptcl_collide_t)
                    self.next_ptcl_collide_id = np.argmin(self.ptcl_collide_t)
                    # print(f'particle {next_ptcl_collide_id} collided with particle {int(self.ptcl_collide_id[next_ptcl_collide_id]) - 1}')
                    # print(f'next_ptcl_collide_t: {next_ptcl_collide_t}, next_ptcl_collide_id: {next_ptcl_collide_id}')
            elif sim_next_state == 'enter':
                # append new particles
                self._append_new_particle(
                    self.ptcl_new_pos[self.next_enter_id],
                    self.ptcl_new_v[self.next_enter_id]
                )
                self._update_collision_predictions(len(self.ptcl_pos) - 1)
                if self.next_enter_id + 1 == len(self.t_enter):
                    next_enter_t = np.inf
                else:
                    next_enter_t = self.t_enter[self.next_enter_id]
                # update collision prediction of the newly entered particle
                self.next_enter_id += 1
            elif sim_next_state == 'gen':
                # refresh particle generation area
                self._refresh_ptcl_gen_area()
            
            # refresh next state
            if self.next_ptcl_collide_t < next_enter_t and self.next_ptcl_collide_t < self.next_gen_t:
                sim_next_state = 'collide'
                next_t = self.next_ptcl_collide_t
            elif next_enter_t < self.next_ptcl_collide_t and next_enter_t < self.next_gen_t:
                sim_next_state = 'enter'
                next_t = next_enter_t
            else:
                sim_next_state = 'gen'
                next_t = self.next_gen_t
            
            # update simulation time
            pbar.update(next_t - self.t)
                
        pbar.close()