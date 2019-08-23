# coding=utf-8
import numpy as np
from gym import spaces
from collections.abc import Iterable
from gym.utils import EzPickle
from ..simulator import Simulator
from .. import logger


class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(
        self,
        gain = 1.0,
        trim = 0.0,
        radius = 0.0318,
        k = 27.0,
        limit = 1.0,
        **kwargs
    ):
        Simulator.__init__(self, **kwargs)
        if self.verbose: logger.info('using DuckietownEnv')

        #self.action_space = spaces.Box(
        #    low=np.array([-1,-1]),
        #    high=np.array([1,1]),
        #    dtype=np.float32
        #)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def step(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        mine = {}
        mine['k'] = self.k
        mine['gain'] = self.gain
        mine['train'] = self.trim
        mine['radius'] = self.radius
        mine['omega_r'] = omega_r
        mine['omega_l'] = omega_l
        info['DuckietownEnv'] = mine
        return obs, reward, done, info


class DuckietownLF(DuckietownEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)
        return obs, reward, done, info


class DuckietownNav(DuckietownEnv, EzPickle):
    """
    Environment for the Duckietown navigation task (NAV)
    """

    def __init__(self, **kwargs):
        self.goal_tile = None
        self.obs_keys = ['cur_pos', 'robot_speed', 'cur_angle']
        DuckietownEnv.__init__(self, **kwargs)
        EzPickle.__init__(self)

    def reset(self):
        DuckietownEnv.reset(self)

        # Find the tile the agent starts on
        start_tile_pos = self.get_grid_coords(self.cur_pos)
        start_tile = self._get_tile(*start_tile_pos)
        
        ## Select a random goal tile to navigate to
        #self.goal_tile = self.drivable_tiles[self.np_random.choice(
        #                        [0, len(self.drivable_tiles) - 1])]
        self.goal_tile = self.drivable_tiles[len(self.drivable_tiles)-1]
        #self.goal_tile = self.drivable_tiles[0]
        info = self.get_agent_info()
        obs = []
        for key in self.obs_keys: 
            information = info['Simulator'][key]
            if key is 'cur_pos':
                obs.append(information[0])
                obs.append(information[2])
            if isinstance(information, Iterable):
                obs.extend(information)
            else:
                obs.append(information)
        return obs


    def _get_manhattan_dist_to_goal(self):
        """
        Returns minimium manhattan distance to closest point on goal tile
        """
        goal_tile_coords = self.goal_tile['coords']
        goal_x_range = np.array([goal_tile_coords[0] * self.road_tile_size,
                                (goal_tile_coords[0] + 1) * self.road_tile_size])
        goal_z_range = np.array([goal_tile_coords[1] * self.road_tile_size,
                                (goal_tile_coords[1] + 1) * self.road_tile_size])
        x_dist = 0
        z_dist = 0
        if self.cur_pos[0] >= goal_x_range[0] and self.cur_pos[0] < goal_x_range[1]:
            x_dist = 0
        else:
            x_dist = np.min(np.abs(self.cur_pos[0] - goal_x_range))
        if self.cur_pos[2] >= goal_z_range[0] and self.cur_pos[2] < goal_z_range[1]:
            z_dist = 0
        else:
            z_dist = np.min(np.abs(self.cur_pos[2] - goal_z_range))
        return x_dist + z_dist

    def step(self, action):
        _, _, done, info = DuckietownEnv.step(self, action)
        obs = []
        for key in self.obs_keys: 
            information = info['Simulator'][key]
            if key is 'cur_pos':
                obs.append(information[0])
                obs.append(information[2])
            if isinstance(information, Iterable):
                obs.extend(information)
            else:
                obs.append(information)
        info['goal_tile'] = self.goal_tile
        goal_distance = self._get_manhattan_dist_to_goal()
        reward = -goal_distance 

        cur_tile_coords = self.get_grid_coords(self.cur_pos)
        cur_tile = self._get_tile(cur_tile_coords[0], cur_tile_coords[1])

        if cur_tile is self.goal_tile:
            done = True
        #    reward = 1000
        #else:
        #    reward = -1
        return np.array(obs), reward, done, info
