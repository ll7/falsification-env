from gym import Env
from gym.spaces import Box
import numpy as np

import math
import sys

import matplotlib.pyplot as plt

# logging
import logging
#logger = logging.getLogger('SimpleWalk2D')
#logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True # disable font warnings based on https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python



class SimpleWalk2DDynGoal(Env):
    """
    simple walk environment in 2D with a continuous action and state space
    """
    def __init__(self):
        """
        Initialize the environment
        Big square as the environment and sub squares as the goal"""
        
        # set the dimensions of the environment
        # environment is square, therefore x and y are the same
        self.x_min = 0.0
        self.x_max = 20.0
        self.width = self.x_max - self.x_min
        
        logger = logging.getLogger(__name__)
        
        # set maximum number of steps to reach the goal
        # if the maximum number of steps is reached, the episode is over
        # maximum is max steps to cross the field
        self.max_steps = int(math.ceil(self.x_max - self.x_min))
        
        # set goal boarder
        self.goal_boarder = 3.0
        
        # ensure that the goal can not walk out of the environment
        self.goal_max_speed =  min(0.5 * self.width / (self.max_steps + 1), 0.9)
        
        # set the distance when the goal is reached
        self.viable_goal_distance = 0.5
        
        # set the max speed of the agent
        self.agent_max_speed = 1.0
        
        # set spaces
        self.action_space = Box(
            low=-self.agent_max_speed, 
            high=self.agent_max_speed, 
            shape=(2, )
            ) # x and y change of position
        
        # TODO must be extended to 4 goal elements. 2 current and 2 future
        self.observation_space = Box(low=self.x_min, high=self.x_max, shape=(6, )) # x,y position, x,y goal
        
        # safe past states in an array, safe x and y positions
        self.state = np.ndarray(shape=(6,), dtype=np.float32)
        
        # init goal direction
        self.goal_direction = np.array([0.0, 0.0])
        self.angle = 0.0
        
        
    def __out_of_bounds(self):
        """check if the current state is out of bounds"""
        position = self.state[0:2]
        for element in position:
            if not (self.x_min <= element <= self.x_max):
                
                return True
            else:
                return False
            
    def __append_state(self):
        self.state_array[0].append(self.state[0])
        self.state_array[1].append(self.state[1])
        
    def __append_goal(self):
        self.goal_array[0].append(self.state[2])
        self.goal_array[1].append(self.state[3])
        
    def __distance_to_goal(self):
        distance_to_goal = np.linalg.norm(self.state[0:2] - self.state[2:4])
        return distance_to_goal     
        
    def __goal_direction(self):
        """calculate direction from init goal position through the middle of the env for the goal"""
        # calculate the center of the environment
        center_point = self.x_min + self.width / 2
        
        # calculate the direction from the goal to the center of the environment
        direction = np.array([center_point, center_point]) - np.array(self.state[2:4])
        
        if np.linalg.norm(direction) == 0.0:
            # return 0, if vector is zero
            self.goal_direction = 0.0
        else:
            # set the direction for the goal movement
            self.goal_direction = direction / np.linalg.norm(direction)
        
    def __calculate_angle(self, vector_1, vector_2):
        """calculate the angle between two vectors in radians"""
        # https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
        
        
        vector_1_length = np.linalg.norm(vector_1)
        vector_2_length = np.linalg.norm(vector_2)
        
        # catch length of zero vectors
        if (vector_1_length == 0.0) or (vector_2_length == 0.0):
            """if one of the vectors is zero, the angle is undefined
            we return 0.0 in this unlikely edge case"""
            return 0.0
        
        logging.debug("vector 1: {}".format(vector_1))
        logging.debug("vector 2: {}".format(vector_2))

        unit_vector_1 = vector_1 / vector_1_length
        unit_vector_2 = vector_2 / vector_2_length

        dot_product = np.clip(
            np.dot(unit_vector_1, unit_vector_2), 
            -1.0, 
            1.0
            )
        
        angle = np.arccos(dot_product)
        if math.isnan(angle):
            raise Exception('angle is nan')

        return angle
    
    def step(self, action):
        previous_state = self.state
        # update position
        self.state[0] += action[0] # update x
        self.state[1] += action[1] # update y
        self.__append_state()
        new_state = self.state
        self.steps_taken += 1
        distance_to_goal = self.__distance_to_goal()
    
        if self.steps_taken >= self.max_steps:
            # maximum number of steps reached
            logging.debug("maximum number of steps reached")
            reward = -1000.0
            done = True
        elif self.__out_of_bounds():
            # went out of bounds
            logging.debug("out of bounds")
            reward = -1000.0
            done = True
        elif distance_to_goal < self.viable_goal_distance:
            # reached goal
            reward = 1000.0
            logging.debug("reached goal")
            done = True
        else:
            # penalize for not moving towards the goal at for each step
            movement = self.distance_to_goal - distance_to_goal
            logging.debug("movement: {}".format(movement))
            # reward a movement towards the goal
            reward = movement * 80 - 100
            done = False

            # update everything
            # TODO update the last goal
            self.state[4:6] = self.state[2:4]
            
            # multiply the goal direction by the max goal speed and apply to goal for movement
            self.state[2:4] += self.goal_direction * self.goal_max_speed
            
            # save the goal movement
            self.__append_goal()
            
            # update distance to goal
            self.distance_to_goal = distance_to_goal
            
            # TODO some error with calculating the angle
            # punish based on direction change
            if self.steps_taken > 1:
                previous_direction = np.array([
                    self.state_array[0][-2] - self.state_array[0][-3],
                    self.state_array[1][-2] - self.state_array[1][-3]
                ])
                now_direction = np.array([
                    self.state_array[0][-1] - self.state_array[0][-2],
                    self.state_array[1][-1] - self.state_array[1][-2]
                ])
                
                self.angle = self.__calculate_angle(previous_direction, now_direction)
                logging.debug("angle: {}".format(self.angle))
                reward -= abs(self.angle / math.pi) * 30 # penalty of zero to 100
            
        
        info = {'distance_to_goal': self.distance_to_goal, 
                'steps_taken': self.steps_taken,
                'previous_state': previous_state,
                'new_state': new_state}
        
        return self.state, reward, done, info
    
    def reset(self):
        """reset and initialize the environment"""
        logging.debug("")
        logging.debug("reset")
        
        # set x and y of position
        self.state[0:2] = np.random.uniform(low=self.x_min, high=self.x_max, size=(2,))
        
        # set goal to a random value, shrinked from entire environment size
        self.state[2:4] = np.random.uniform(
            low = self.x_min + self.goal_boarder, 
            high = self.x_max - self.goal_boarder,
            size=(2,)
            )
        
        
        
        
        # empty array for saving all visited states
        self.state_array = [[], []] # x, y
        
        # empty array for all movements of the goal
        self.goal_array = [[], []] # x, y
        
        # init append state and goal
        self.__append_state()
        self.__append_goal()
        
        
        self.__goal_direction()
        
        """
        calculate possible previous goal position for initialization
        the previous goal does not exist after initialization, so the init value would be 0
        This violates the Markov property, therefore we caluclate a dummy goal position.
        We go in the oposite direction of the goal direction.
        self.state[2:4] += self.goal_direction * self.goal_max_speed
        """
        self.state[4:6] = self.state[2:4] - self.goal_direction * self.goal_max_speed
        
        # reset steps taken
        self.steps_taken = 0
        
        # estimate distance to goal
        self.distance_to_goal = self.__distance_to_goal()
        
        self.angle = 0.0
        
        return self.state
    
    def render(self):
        
        # logging.debug("visited states: ", self.state_array)
        logging.debug("x: {}".format(self.state_array[0]))
        logging.debug("y: {}".format(self.state_array[1]))
        goal = self.state[2:4]
        logging.debug("goal: {}".format(goal))
        
        
        # plot
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # plot the visited states
        ax.plot(self.state_array[0], self.state_array[1], linewidth=1.0, marker='o', color='b', markersize=1.0)
        ax.grid(True)
        # plot the goal
        # ax.plot(goal[0], goal[1], 'ro')
        circle1 = plt.Circle((goal[0], goal[1]), radius=self.viable_goal_distance, color='r', fill=False)
        ax.add_patch(circle1)
        # TODO add trajectory of the goal
        ax.plot(self.goal_array[0], self.goal_array[1], linewidth=1.0, color='r', marker='*', markersize=1.0)
        
        ax.set(
            xlim=(self.x_min, self.x_max), #xticks=np.arange(1, 8),
            ylim=(self.x_min, self.x_max), #yticks=np.arange(1, 8))
            )
        plt.show()
