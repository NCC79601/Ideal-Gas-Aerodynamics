import numpy as np
import torch
import json
import os

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
        
        # parse config
        self.ptcl_radius = config['ptcl_radius'] # particle radius
        self.ptcl_mass = config['ptcl_mass'] # particle mass
        self.T0 = config['T0'] # static temperature
        self.v0 = config['v0'] # flowing velocity
        self.spatial_density = config['spatial_density'] # spatial density (number of particles per unit area)+
        +.3
        self.map_size = config['map_size'] # map size [w, h]
        self.ptcl_gen_area_size = config['ptcl_gen_area_size'] # particle generation area size [w, h]
        self.walls = config['walls'] # walls in the map [[[x1, y1], [x2, y2]], ...]
        self.t_lim = config['t_lim'] # simulation time limit
        self.t_scale = config['t_scale'] # time between frames (default 1/60 s)

        # initialize particles
        ptcl_count = self.spatial_density * self.map_size[0] * self.map_size[1]
        print(f'Initializing map [{self.map_size[0]} x {self.map_size[1]}] with {ptcl_count} particles...')
        self.ptcl_pos, self.ptcl_v = ptcl_sampler(ptcl_count, self.map_size, self.T0, self.ptcl_mass, self.v0)
        
        # initialize particle generation area
        ptcl_new_count = self.spatial_density * self.ptcl_gen_area_size[0] * self.ptcl_gen_area_size[1]
        self.ptcl_new_pos, self.ptcl_new_v, self.t_to_enter = \
            ptcl_new_sampler(ptcl_new_count, self.ptcl_gen_area_size, self.T0, self.ptcl_mass, self.v0)
        self.t_enter = self.t + self.t_to_enter
        self.next_enter_id = 0
        self.gen_area_refresh_interval = self.ptcl_gen_area_size[0] / self.v0 # time interval to refresh particle generation area
        self.next_gen_t = self.t + self.gen_area_refresh_interval # next time to refresh particle generation area

        # initialize collision detection
        self.ptcl_collide_t = np.ones(ptcl_count) * np.inf # collision time
        self.collide_id = np.zeros(ptcl_count) # negative id for wall collision

        # initialize key frames
        self.key_frames = [] # [[t, ptcl_pos, ptcl_v], ...]

        # initialize current simulation time
        self.t = 0

        # initialize last recorded time
        self.t_last_recorded = 0

    def _update_positions(self, elapsed_time: float) -> None:
        '''
        Update particle positions and delete outbounded particles

        Parameters:
        - elapsed_time: elapsed time of this update (usually till the next predicted collision)
        '''
        self.ptcl_pos += self.ptcl_v * elapsed_time
        self.ptcl_pos[:, 1] = self.ptcl_pos[:, 1] % self.map_size[1]
         # delete particles outside the map
        remove_ids = np.where(self.ptcl_pos[:, 0] > self.map_size[0])[0]
        self.ptcl_pos = np.delete(self.ptcl_pos, remove_ids, axis=0)
    
    def _get_positions(self, t: float) -> np.ndarray:
        '''
        Get particle positions

        Parameters:
        - t: specified time

        Returns:
        - particle positions at time `t`
        '''
        return self.ptcl_pos + self.ptcl_v * (t - self.t)

    def _update_specific_collision_prediciton(self, ptcl_id: int, collide_t: float, collide_id: int) -> None:
        '''
        Single-point collision prediction update for specific particle

        Parameters:
        - ptcl_id: id of particle to update (index from 1, to avoid ambiguity)
        - collide_t: collision time
        - collide_id: collision id (negative for wall collision)

        Return:
        - True if collision time is updated, False otherwise
        '''
        if collide_t < self.t:
            return False
        if collide_t < self.ptcl_collide_t[ptcl_id]:
            self.ptcl_collide_t[ptcl_id] = collide_t
            self.collide_id[ptcl_id] = collide_id
            return True
        return False

    def _update_collision_predictions(self) -> None:
        '''
        Update collision predictions for all particles (usually only called when initializing)
        '''
        # update particle-particle collisions
        for i in range(len(self.ptcl_pos)):
            for j in range(i + 1, len(self.ptcl_pos)):
                x1, y1 = self.ptcl_pos[i]
                x2, y2 = self.ptcl_pos[j]
                v1x, v1y = self.ptcl_v[i]
                v2x, v2y = self.ptcl_v[j]
                # compute quadratic equation coefficients
                a = (v1x - v2x) ** 2 + (v1y - v2y) ** 2
                b = 2 * ((x1 - x2) * (v1x - v2x) + (y1 - y2) * (v1y - v2y))
                c = (x1 - x2) ** 2 + (y1 - y2) ** 2 - (2 * self.ptcl_radius) ** 2
                # check if particles collide
                if b ** 2 - 4 * a * c >= 0:
                    t = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                    if t > self.t:
                        self._update_specific_collision_prediciton(i, t, j + 1)
                        self._update_specific_collision_prediciton(j, t, i + 1)
        
        # update particle-wall collisions
        for i in range(len(self.ptcl_pos)):
            x, y = self.ptcl_pos[i]
            vx, vy = self.ptcl_v[i]
            for j, wall in enumerate(self.walls):
                x1, y1 = wall[0]
                x2, y2 = wall[1]
                v_dir = np.array([x2 - x1, y2 - y1])
                # judge whether the particle is moving towards the wall
                if np.cross(v_dir, np.array([x - x1, y - y1])) * np.cross(v_dir, np.array([vx, vy])) > 0:
                    continue
                # compute line equation coefficients (A * x + B * y + C = 0)
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                # compute collision time
                t = (-self.ptcl_radius * np.sqrt(A**2 + B**2) - (A * x + B * y + C)) / (A * vx + B * vy)
                if t > self.t:
                    self._update_specific_collision_prediciton(i, t, -j - 1)

    def _collide_particle_particle(self, i: int, j: int) -> None:
        '''
        Perform collision between particles `i` and `j`

        Parameters:
        - i: id of particle i
        - j: id of particle j
        '''
        # compute relative velocity
        # reference: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        p1 = self.ptcl_pos[i]
        v1 = self.ptcl_v[i]
        p2 = self.ptcl_pos[j]
        v2 = self.ptcl_v[j]

        v1_new = v1 - (np.dot(v1 - v2, p1 - p2) / np.linalg.norm(p1 - p2) ** 2) * (p1 - p2)
        v2_new = v2 - (np.dot(v2 - v1, p2 - p1) / np.linalg.norm(p2 - p1) ** 2) * (p2 - p1)
        
        self.ptcl_v[i] = v1_new
        self.ptcl_v[j] = v2_new

    def _collide_particle_wall(self, i: int, wall_id: int) -> None:
        '''
        Perform collision between particle `i` and wall `wall_id`

        Parameters:
        - i: id of particle i
        - wall_id: id of the wall
        '''
        # compute normal vector of the wall
        wall = self.walls[wall_id]
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        n = np.array([y1 - y2, x2 - x1])
        n /= np.linalg.norm(n)
        # compute new velocity
        v = self.ptcl_v[i]
        self.ptcl_v[i] = v - 2 * np.dot(v, n) * n

    def _refresh_ptcl_gen_area(self) -> None:
        '''
        Refresh particle generation area
        '''
        ptcl_new_count = self.spatial_density * self.ptcl_gen_area_size[0] * self.ptcl_gen_area_size[1]
        self.ptcl_new_pos, self.ptcl_new_v, self.t_to_enter = \
            ptcl_new_sampler(ptcl_new_count, self.ptcl_gen_area_size, self.T0, self.ptcl_mass, self.v0)
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
        self.ptcl_v = np.vstack([self.ptcl_v, ptcl_v])

    def simulate(self):
        '''
        Perform simulation
        '''

        # initialize collision predictions
        self._update_collision_predictions()

        # record the first frame
        self.key_frames.append(
            self._get_positions(0)
        )

        # next state of simulation FSM
        sim_next_state = ''

        # calculate the next collision time
        next_collide_t = np.min(self.ptcl_collide_t)
        next_collide_id = np.argmin(self.ptcl_collide_t)

        # calculate the next new particle enter time
        next_enter_t = self.t_enter[self.next_enter_id]

        next_gen_t = self.next_gen_t

        # next simulation state time
        next_t = 0

        # construct next state of simulation FSM
        if next_collide_t < next_enter_t and next_collide_t < next_gen_t:
            sim_next_state = 'collide'
            next_t = next_collide_t
        elif next_enter_t < next_collide_t and next_enter_t < next_gen_t:
            sim_next_state = 'enter'
            next_t = next_enter_t
        else:
            sim_next_state = 'gen'
            next_t = next_gen_t

        # simulation loop (finite state machine)
        while self.t < self.t_lim:
            # HACK:
            # record key frames
            while self.t - self.t_last_recorded >= self.t_scale:
                self.key_frames.append(
                    self._get_positions(self.t_scale)
                )
                self.t_last_recorded = self.t_last_recorded + self.t_scale

            # update positions
            self._update_positions(next_t - self.t)

            if sim_next_state == 'collide':
                # perform collision
                if next_collide_id < 0:
                    self._collide_particle_wall(-next_collide_id - 1, self.collide_id[next_collide_id])
                    # TODO: update collision prediction of the particle
                    # next_collide_t, next_collide_id
                    pass
                else:
                    self._collide_particle_particle(next_collide_id - 1, self.collide_id[next_collide_id] - 1)
                    # TODO: update collision prediction of the particle
                    # next_collide_t, next_collide_id
                    pass
            elif sim_next_state == 'enter':
                # append new particles
                self._append_new_particle(
                    self.ptcl_new_pos[self.next_enter_id],
                    self.ptcl_new_v[self.next_enter_id]
                )
                self.next_enter_id += 1
                # update collision prediction
                # TODO: update collision prediction of the newly entered particle
                # next_enter_t, next_enter_id
                pass
            elif sim_next_state == 'gen':
                # refresh particle generation area
                self._refresh_ptcl_gen_area()
            
            # refresh next state
            if next_collide_t < next_enter_t and next_collide_t < next_gen_t:
                sim_next_state = 'collide'
                next_t = next_collide_t
            elif next_enter_t < next_collide_t and next_enter_t < next_gen_t:
                sim_next_state = 'enter'
                next_t = next_enter_t
            else:
                sim_next_state = 'gen'
                next_t = next_gen_t
                