'''
API for agent-environment interactions.
'''

from . import episode
from ..utils import metrics

class Env:
    '''
    Abstraction for the environment an RL agent interacts with.
    '''

    def __init__(self, config, metrics_logger=metrics.EmptyLogger()):
        self.episode_configurations = config.get_episodes()
        self.metrics_logger = metrics_logger

        # init environment state
        self.episode_counter = 0
        self.current_episode = None

    def reset(self):
        '''
        Returns the starting state of an episode.
        '''
        self.current_episode = episode.Episode.build(self.episode_configurations[self.episode_counter])
        self.metrics_logger.new_episode(self.current_episode)
        return self.current_episode.reset()

    def step(self, action):
        '''
        Performs the action at the current state.

        Args
            action: agent action

        Returns
            state:      new state
            reward:     reward for action given state transition
            terminal:   boolean indicating whether the new state is terminal
        '''
        assert self.current_episode is not None
        state, reward, terminal = self.current_episode.step(action)

        # log for metrics
        self.metrics_logger.log_reward(reward)

        # handle end of episode
        if terminal:
            self.current_episode = None
            self.episode_counter += 1

        return state, reward, terminal

    def has_next_episode(self):
        '''
        Whether or not there is an additional episode to be played.
        '''
        return self.episode_counter < len(self.episode_configurations)

class Config:

    def __init__(self, episode_configurations):
        self.episode_configurations = episode_configurations

    def get_episodes(self):
        return self.episode_configurations

    @staticmethod
    def parse(path_to_config):
        episode_configs = []
        with open(path_to_config, 'r') as f:
            for line in f.readlines():
                episode_configs.append(episode.Config.parse(line.strip()))
        return Config(episode_configs)
