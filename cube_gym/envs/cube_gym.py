import gym
from gym import spaces
import pygame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rsc.helper import *

class CubeGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "3d"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 1000  # The size of the PyGame window
        self.fig = plt.figure()


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(3,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(3,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "forward"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0, 0]),     #right
            1: np.array([0, 1, 0]),     #up
            2: np.array([-1, 0, 0]),    #left
            3: np.array([0, -1, 0]),    #down
            4: np.array([0, 0, 1])      #forward
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "3d":
            self.update_3d = True
            self.render_mode = "human"
        else:
            self.update_3d = False


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.canvas = {'x': None, 'y': None, 'z': None}

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._agent_location = np.append(self._agent_location, 0)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )
        self._target_location = np.append(self._target_location, self.size-1)
        #self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=3, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "3d":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.canvas = self.update_canvas(self.update_3d)
            self.update_window()
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def update_window(self):
        self.window.blit(self.canvas.get('x'), self.canvas.get('x').get_rect())
        text, textRect = self.get_cpation('Y-Z Axis')
        self.window.blit(text, textRect)
        self.window.blit(self.canvas.get('y'), self.canvas.get('y').get_rect(x=self.window_size/2, y =0))
        text, textRect = self.get_cpation('X-Z Axis')
        self.window.blit(text, textRect)
        self.window.blit(self.canvas.get('z'), self.canvas.get('z').get_rect(x=self.window_size/2, y =self.window_size/2))
        text, textRect = self.get_cpation('X-Y Axis')
        self.window.blit(text, textRect)
        return
    
    def get_cpation(self, caption):
        font = pygame.font.Font('freesansbold.ttf', 12)
        text = font.render(caption, True, (0, 0, 128), (255, 255, 255))
        textRect = text.get_rect()
        if caption == 'Y-Z Axis':
            textRect.center = (self.window_size/4, self.window_size/2 - 5)
        elif caption == 'X-Z Axis':
            textRect.center = (self.window_size/4 + self.window_size/2, self.window_size/2 - 5)
        elif caption == 'X-Y Axis':
            textRect.center = (self.window_size/4 + self.window_size/2, self.window_size - 5)
        return text, textRect
    
    def update_canvas(self, plot_3d=False):
        sub_size = self.window_size/2 - 10

        self.update_xcanvas(sub_size)
        self.update_ycanvas(sub_size)
        self.update_zcanvas(sub_size)
        if plot_3d:      
            self.plot_3dview(sub_size)
        return self.canvas

    def plot_3dview(self, sub_size=3):
        axes = [self.size, self.size, self.size]
        data = np.ones(axes, dtype=np.bool_)
        alpha = 0.01
        colors = np.empty(axes + [4], dtype=np.float32)
        colors[:] = [1, 1, 1, alpha]  # white


        ax = self.fig.add_subplot(111, projection='3d')
        ax.voxels(data, facecolors=colors, edgecolors='black')
        
        target_cube = draw3d_target_cube(self._target_location, 'red')
        agent_cube = draw3d_target_cube(self._agent_location, 'blue')
        #x, y, z = draw3d_agent_sphere(self._agent_location, 0.5)
        #agent_sphere = ax.plot_surface(x, y, z, color='blue', alpha=0.5)
        #ax.add_collection3d(agent_sphere)
        ax.add_collection3d(agent_cube)
        ax.add_collection3d(target_cube)
        plt.show(block=False)
        plt.pause(0.000001)
        return
        
    

    def update_xcanvas(self, sub_size=3):
        #The following line is for the x-axis canvas
        self.canvas['x'] = pygame.Surface((sub_size, sub_size))
        self.canvas['x'].fill((255, 255, 255))
        pix_square_size = (
            sub_size / (self.size)
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            self.canvas['x'],
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location[1:],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            self.canvas['x'],
            (0, 0, 255),
            (self._agent_location[1:] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                self.canvas['x'],
                0,
                (0, pix_square_size * x),
                (sub_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas['x'],
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, sub_size),
                width=3,
            )
    
    def update_ycanvas(self, sub_size=3):
        #The following line is for the y-axis canvas
        y_target_location = np.array([self._target_location[0], self._target_location[2]])
        y_agent_location = np.array([self._agent_location[0], self._agent_location[2]])
        self.canvas['y'] = pygame.Surface((sub_size, sub_size))
        self.canvas['y'].fill((255, 255, 255))
        pix_square_size = (
            sub_size / (self.size)
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            self.canvas['y'],
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * y_target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            self.canvas['y'],
            (0, 0, 255),
            (y_agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                self.canvas['y'],
                0,
                (0, pix_square_size * x),
                (sub_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas['y'],
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, sub_size),
                width=3,
            )
    
    def update_zcanvas(self, sub_size=3):
        #The following line is for the z-axis canvas
        self.canvas['z'] = pygame.Surface((sub_size, sub_size))
        self.canvas['z'].fill((255, 255, 255))
        pix_square_size = (
            sub_size / (self.size)
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            self.canvas['z'],
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location[:-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            self.canvas['z'],
            (0, 0, 255),
            (self._agent_location[:-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                self.canvas['z'],
                0,
                (0, pix_square_size * x),
                (sub_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas['z'],
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, sub_size),
                width=3,
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
