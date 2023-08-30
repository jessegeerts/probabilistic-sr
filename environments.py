"""
environments.py
====================================
The module containing RL environment objects. They are separated in EstimationTasks, where the agent only needs
to estimate the value function, and ControlTasks, where the agent
"""
import numpy as np
import networkx as nx
import pandas as pd
import random
import os

from scipy import stats
import gym
from gym import spaces
from gym.utils import seeding

import utils
import matplotlib.pyplot as plt
from dynamic_programming_utils import generate_random_policy


class EstimationTask(object):
    def __init__(self):
        pass

    def make_exp_seq(self):
        """Construct the experimental sequence of stimuli and rewards shown to the agent"""
        pass


class SharpeRevaluation(EstimationTask):
    """This class simulates the revaluation experiment ran by Sharpe et al.

    The correspondence of features and feature numbers:
    A = 0
    C = 1
    D = 2
    E = 3
    F = 4
    X = 5
    food = 6

    The full experiment has 5 stages

    1: preconditioning 1
            A  --> X          (x24)
    2: preconditioning 2
            EF --> X
            AD --> X
            AC --> X
            A  --> X          (x8)
    3: Conditioning
            X  --> food       (4x24)
    4: Devaluation
            food --> -10 R    (x1)
    5: Test
            C                 (x6)

    """
    def __init__(self, devalue=False, opto=True):
        super().__init__()
        self.devalue = devalue
        self.opto = opto
        self.nr_features = 7
        self.feature_names = {
            0: 'A',
            1: 'C',
            2: 'D',
            3: 'E',
            4: 'F',
            5: 'X',
            6: 'food'
        }

        identity = np.eye(self.nr_features)
        self.A = identity[0]
        self.C = identity[1]
        self.D = identity[2]
        self.E = identity[3]
        self.F = identity[4]
        self.X = identity[5]
        self.food = identity[6]

        self.A_X = np.stack((self.A, self.X + self.A))
        self.EF_X = np.stack((self.E + self.F, self.X))
        self.AD_X = np.stack((self.A + self.D, self.X))
        self.AC_X = np.stack((self.A + self.C, self.X))
        self.X_food = np.stack((self.X, self.food + self.X))

        self.stim_seq = None
        self.reward_seq = None
        self.opto_seq = None
        self.stage_idx = None
        self.make_exp_seq()

    def make_exp_seq(self):
        """Construct the experimental sequence of stimuli and rewards shown to the agent"""
        # Make stimulus sequence
        stage_1 = self.randomise_trials(24, self.A_X)                           # Preconditioning 1
        stage_2 = self.randomise_trials(8, self.EF_X, self.AD_X, self.AC_X)     # Preconditioning 2
        stage_3 = self.randomise_trials(4*24, self.X_food)                      # Conditioning
        stage_4 = np.tile(self.food, (1, 1))                                    # Devaluation
        stage_5 = np.tile(self.C, (6, 1))                                       # Test

        self.stage_idx = np.concatenate((np.tile(1, len(stage_1)),
                                        np.tile(2, len(stage_2)),
                                        np.tile(3, len(stage_3)),
                                        np.tile(4, len(stage_4)),
                                        np.tile(5, len(stage_5))))

        self.stim_seq = np.vstack((stage_1, stage_2, stage_3, stage_4, stage_5))
        self.reward_seq = np.zeros(len(self.stim_seq))

        idx_c = np.where((self.stim_seq == self.X_food[1]).all(axis=1))[0]
        idx_d = np.where((self.stage_idx == 4))[0][0]

        self.reward_seq[idx_c] = 1.
        if self.devalue:
            self.reward_seq[idx_d] = -5.
        else:
            self.reward_seq[idx_d] = 1.

        self.make_opto_seq()

    def make_opto_seq(self):
        self.opto_seq = np.zeros(len(self.stim_seq))
        if self.opto:
            opto_idx_acx = np.where((self.stim_seq == self.AC_X[0]).all(axis=1))[0] + 1
            opto_idx_adx = np.where((self.stim_seq == self.AD_X[0]).all(axis=1))[0] + 1
            self.opto_seq[opto_idx_acx] = 1
            self.opto_seq[opto_idx_adx] = 1

    def randomise_trials(self, n_trials, *args):
        """Randomise stimulus presentations, adding a vector of zeros between every presentation.

        :param n_trials:
        :param *args: Stimulus sequences to be added. Each stimulus is an [ s X D ] matrix (s = number of states within
        a single stimulus sequence, D = number of features)
        :return:
        """
        n_cues = len(args)

        stimulus_sequence = []
        for trial in range(n_trials):
            stim_order = np.random.permutation(n_cues)

            for cue in stim_order:
                stim_length = args[cue].shape[0]

                for k in range(stim_length):
                    stimulus_sequence.append(args[cue][k])
                stimulus_sequence.append(np.zeros(self.nr_features))

        return np.array(stimulus_sequence)

    def get_stage(self, trial):
        return self.stage_idx[trial]


class HartRevaluation(EstimationTask):
    """This class simulates the revaluation experiment ran by Hart et al. (2020)

    The correspondence of features and feature numbers:
    A = 0
    C = 1
    D = 2
    E = 3
    F = 4
    X = 5
    food = 6

    The full experiment has 4 stages

    1: preconditioning 1
            C  --> X          (x8)
    2: Conditioning
            X  --> food       (4x24)
    3: Devaluation
            food --> -10 R    (x1)
    4: Test
            C                 (x6)

    """
    def __init__(self, devalue=False, opto=True, n_precond_trials=8, n_cond_trials=4*24):
        super().__init__()
        self.devalue = devalue
        self.opto = opto
        self.nr_features = 7
        self.feature_names = {
            0: 'A',
            1: 'C',
            2: 'D',
            3: 'E',
            4: 'F',
            5: 'X',
            6: 'food'
        }

        identity = np.eye(self.nr_features)
        self.A = identity[0]
        self.C = identity[1]
        self.D = identity[2]
        self.E = identity[3]
        self.F = identity[4]
        self.X = identity[5]
        self.food = identity[6]

        self.C_X = np.stack((self.C, self.X))
        self.X_food = np.stack((self.X, self.food))

        self.stim_seq = None
        self.reward_seq = None
        self.opto_seq = None
        self.stage_idx = None
        self.make_exp_seq(n_precond_trials, n_cond_trials)

    def make_exp_seq(self, n_precond_trials=8, n_cond_trials=4*24):
        """Construct the experimental sequence of stimuli and rewards shown to the agent"""
        # Make stimulus sequence
        stage_2 = self.randomise_trials(n_precond_trials, self.C_X)                            # Preconditioning
        stage_3 = self.randomise_trials(n_cond_trials, self.X_food)                      # Conditioning
        stage_4 = np.tile(np.stack([self.food, np.zeros(self.food.shape)]), (1, 1))                                    # Devaluation
        stage_5 = np.tile(self.C, (3, 1))                                       # Test

        self.stage_idx = np.concatenate((np.tile(2, len(stage_2)),
                                        np.tile(3, len(stage_3)),
                                        np.tile(4, len(stage_4)),
                                        np.tile(5, len(stage_5))))

        self.stim_seq = np.vstack((stage_2, stage_3, stage_4, stage_5))
        self.reward_seq = np.zeros(len(self.stim_seq))

        idx_c = np.where((self.stim_seq == self.X_food[1]).all(axis=1))[0]
        idx_d = np.where((self.stage_idx == 4))[0][0]

        self.reward_seq[idx_c] = 1.
        if self.devalue:
            self.reward_seq[idx_d] = -5.
        else:
            self.reward_seq[idx_d] = 1.

        self.make_opto_seq()

    def randomise_trials(self, n_trials, *args):
        """Randomise stimulus presentations, adding a vector of zeros between every presentation.

        :param n_trials:
        :param *args: Stimulus sequences to be added. Each stimulus is an [ s X D ] matrix (s = number of states within
        a single stimulus sequence, D = number of features)
        :return:
        """
        n_cues = len(args)

        stimulus_sequence = []
        for trial in range(n_trials):
            stim_order = np.random.permutation(n_cues)

            for cue in stim_order:
                stim_length = args[cue].shape[0]

                for k in range(stim_length):
                    stimulus_sequence.append(args[cue][k])
                stimulus_sequence.append(np.zeros(self.nr_features))

        return np.array(stimulus_sequence)

    def get_stage(self, trial):
        return self.stage_idx[trial]

    def make_opto_seq(self):
        # no opto in this task
        self.opto_seq = np.zeros(len(self.stim_seq))


class ControlTask(object):
    """Parent class for RL environments holding some general methods.
    """
    def __init__(self):
        self.nr_states = None
        self.nr_actions = None
        self.actions = None
        self.adjacency_mat = None
        self.goal_state = None
        self.reward_func = None
        self.graph = None
        self.n_features = None
        self.rf = None
        self.transition_probabilities = None
        self.terminal_state = None
        self.state_indices = None
        self.current_state = None

    def act(self, action):
        pass

    def get_current_state(self):
        return self.current_state

    def reset(self):
        pass

    def define_adjacency_graph(self):
        pass

    def _fill_adjacency_matrix(self):
        pass

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph

    def create_graph(self):
        """Create networkx graph from adjacency matrix.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix())

    def show_graph(self, map_variable=None, layout=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = nx.spring_layout(self.graph)
        if map_variable is not None:
            categories = pd.Categorical(map_variable)
            node_color = categories
        else:
            node_color = 'b'
        nx.draw(self.graph, with_labels=True, pos=layout, node_color=node_color, node_size=node_size, **kwargs)

    def set_reward_location(self, state_idx, action_idx):
        self.goal_state = state_idx
        action_destination = self.transition_probabilities[state_idx, action_idx]
        self.reward_func = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        self.reward_func[state_idx, action_idx] = action_destination

    def is_terminal(self, state_idx):
        if not self.get_possible_actions(state_idx):
            return True
        else:
            return False

    def get_destination_state(self, current_state, current_action):
        transition_probabilities = self.transition_probabilities[current_state, current_action]
        return np.flatnonzero(transition_probabilities)

    def get_degree_mat(self):
        degree_mat = np.eye(self.nr_states)
        for state, degree in self.graph.degree:
            degree_mat[state, state] = degree
        return degree_mat

    def get_laplacian(self):
        return self.get_degree_mat() - self.adjacency_mat

    def get_normalised_laplacian(self):
        """Return the normalised laplacian.
        """
        D = self.get_degree_mat()
        L = self.get_laplacian()  # TODO: check diff with non normalised laplacian. check adverserial examples
        exp_D = utils.exponentiate(D, -.5)
        return exp_D.dot(L).dot(exp_D)

    def compute_laplacian(self, normalization_method=None):
        """Compute the Laplacian.

        :param normalization_method: Choose None for unnormalized, 'rw' for RW normalized or 'sym' for symmetric.
        :return:
        """
        if normalization_method not in [None, 'rw', 'sym']:
            raise ValueError('Not a valid normalisation method. See help(compute_laplacian) for more info.')

        D = self.get_degree_mat()
        L = D - self.adjacency_mat

        if normalization_method is None:
            return L
        elif normalization_method == 'sym':
            exp_D = utils.exponentiate(D, -.5)
            return exp_D.dot(L).dot(exp_D)
        elif normalization_method == 'rw':
            exp_D = utils.exponentiate(D, -1)
            return exp_D.dot(L)

    def get_possible_actions(self, state_idx):
        pass

    def get_adjacent_states(self, state_idx):
        pass

    def compute_feature_response(self):
        pass

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for action in range(self.nr_actions):
                transition_matrix[state] += self.transition_probabilities[state, action] * policy[state][action]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def frep(self, s):
        return np.identity(self.nr_states)[s]

    def sarep(self, s, a):
        return np.identity(self.nr_states * self.nr_actions)[(a * self.nr_states) + s]

    def get_next_state(self, origin, action):
        pass

    def get_reward(self, state, **args):
        pass

    def get_random_sr(self, gamma=.95):
        random_policy = generate_random_policy(self)
        random_walk_sr = self.get_successor_representation(random_policy, gamma=gamma)
        return random_walk_sr


class SimpleMDP(ControlTask):
    """Very simple MDP with states on a linear track. Agent gets reward of 1 if it reaches last state.
    """
    def __init__(self, nr_states=3, reward_probability=1.):
        ControlTask.__init__(self)
        self.reward_probability = reward_probability
        self.nr_states = nr_states
        self.n_features = nr_states
        self.state_indices = np.arange(self.nr_states)
        self.nr_actions = 2
        self.actions = [0, 1]
        self.action_consequences = {0: -1, 1: +1}
        self.terminal_states = [self.nr_states - 1]

        self.transition_probabilities = self.define_transition_probabilities()
        self.reward_func = np.zeros((self.nr_states, self.nr_actions))
        self.reward_func[self.nr_states-2, 1] = 1
        self.start_state = 0
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state

    def frep(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.is_terminal(state_idx):
            return np.zeros(self.nr_states)
        else:
            return np.eye(self.nr_states)[state_idx]

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :, :] = 0
                continue
            for action_key, consequence in self.action_consequences.items():
                successor = int(predecessor + consequence)
                if successor not in self.state_indices:
                    transition_probabilities[predecessor, action_key, predecessor] = 1  # stay in current state
                else:
                    transition_probabilities[predecessor, action_key, successor] = 1
        return transition_probabilities

    def get_possible_actions(self, state_idx):
        if state_idx in self.terminal_states:
            return []
        else:
            return list(self.action_consequences)

    def define_adjacency_graph(self):
        transitions_under_random_policy = self.transition_probabilities.sum(axis=1)
        adjacency_graph = transitions_under_random_policy != 0
        return adjacency_graph.astype('int')

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for action in range(self.nr_actions):
                transition_matrix[state] += self.transition_probabilities[state, action] * policy[state][action]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_next_state(self, current_state, action):
        next_state = np.flatnonzero(self.transition_probabilities[current_state, action])[0]
        return next_state

    def get_reward(self, current_state, action):
        if np.random.rand() <= self.reward_probability:
            reward = self.reward_func[current_state, action]
        else:
            reward = 0.
        return reward

    def get_next_state_and_reward(self, current_state, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(current_state):
            return current_state, 0

        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(current_state, action)
        return next_state, reward

    def act(self, action):
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        return next_state, reward

    def get_current_state(self):
        """Return current state idx given current position.
        """
        return self.current_state

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states), dtype=np.int)
        for idx in self.state_indices:
            if (idx + 1) in self.state_indices:
                self.adjacency_graph[idx, idx + 1] = 1

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph


class DeterministicTask(ControlTask):
    """This class implements the deterministic two-step task described in Doll et al (2015, Nature Neuroscience).
    """
    output_folder = 'data/DeterministicTask/'

    def __init__(self, n_trials=272):
        ControlTask.__init__(self)
        self.state_names = ['faces', 'tools', 'bodyparts', 'scenes',
                            'terminal1', 'terminal2', 'terminal3', 'terminal4']
        self.actions = np.array(['left', 'right'])
        self.nr_states = len(self.state_names)
        self.n_trials = n_trials
        self.nr_actions = 2
        self.states_actions_outcomes = {
            'faces': {
                'left': ['bodyparts'],
                'right': ['scenes']
            },
            'tools': {
                'left': ['bodyparts'],
                'right': ['scenes']
            },
            'bodyparts': {
                'left': ['terminal1'],
                'right': ['terminal2']
            },
            'scenes': {
                'left': ['terminal3'],
                'right': ['terminal4']
            },
            'terminal1': {},
            'terminal2': {},
            'terminal3': {},
            'terminal4': {}
        }

        self._fill_adjacency_matrix()
        self.set_transition_probabilities()
        self.create_graph()

        self.reward_traces = self.load_reward_traces()
        self.reward_probs = self.reward_traces[:, 0]  # [1, 0, .1, .1]

        self.start_state = 0  # Change this to stochastic 0 or 1
        self.curr_state = self.start_state
        self.curr_action_idx = 0

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))

        for idx in range(self.nr_states):
            state_name = self.state_names[idx]

            actions = self.states_actions_outcomes[state_name]
            for act, destination_list in actions.items():
                for dest in destination_list:
                    destination_idx = list(self.states_actions_outcomes.keys()).index(dest)
                    self.adjacency_graph[idx, destination_idx] = 1

    def set_transition_probabilities(self):
        """Set the transition probability matrix.
        """
        self.transition_probabilities = np.zeros((self.nr_states, self.nr_actions, self.nr_states))
        for state, act_outcome in self.states_actions_outcomes.items():
            s_idx = self.get_state_idx(state)
            for a_idx, (act, possible_destinations) in enumerate(act_outcome.items()):
                for i, d in enumerate(possible_destinations):
                    d_idx = self.get_state_idx(d)
                    if len(possible_destinations) == 1:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = 1

    def get_possible_actions(self, state_idx):
        state_name = self.state_names[state_idx]
        possible_actions = list(self.states_actions_outcomes[state_name].keys())
        return possible_actions

    def get_state_idx(self, state_name):
        return self.state_names.index(state_name)

    def is_terminal(self, state_idx):
        state_name = self.state_names[state_idx]
        return self.states_actions_outcomes[state_name] == {}

    def reset(self):
        self.start_state = np.random.choice([0, 1])
        self.curr_state = self.start_state

    def plot_graph(self, map_variable=None, node_size=1500, **kwargs):
        positions = {0: [0.25, 2], 1: [1.25, 2], 2: [.25, 1], 3: [1.25, 1],
                     4: [0, 0], 5: [.5, 0], 6: [1, 0], 7: [1.5, 0]}
        self.show_graph(map_variable=map_variable, node_size=node_size,
                        layout=positions, **kwargs)

    def generate_reward_traces(self, **kwargs):
        """Generate reward traces per reward port per trial using a Gaussian random walk and save in file.
        :return:
        """
        r1 = bounded_random_walk(self.n_trials, **kwargs)
        r2 = [1-r for r in r1]  # bounded_random_walk(self.n_trials, **kwargs)
        #r2 = bounded_random_walk(self.n_trials, **kwargs)
        rewards = np.array([r1, r2, r1[::-1], r2[::-1]])
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        np.save(file_path, rewards)

    def load_reward_traces(self):
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        try:
            reward_traces = np.load(file_path)
            print('Loaded reward traces from file.')
        except FileNotFoundError:
            print('Warning: No reward traces file was found so I generate a new one.')
            self.generate_reward_traces(avg_stepsize=.05, sigma=.0005)
            reward_traces = np.load(file_path)
        return reward_traces


class StochasticTask(ControlTask):
    """This class implements the stochastic two-step task as described in Daw et al. (2011, Neuron).
    """
    output_folder = 'data/StochasticTask'

    def __init__(self, n_trials=272):
        ControlTask.__init__(self)
        self.state_names = ['initiation', 'left_state', 'right_state',
                            'terminal1', 'terminal2', 'terminal3', 'terminal4']
        self.common_probability = .7
        self.rare_probability = 1 - self.common_probability

        self.actions = np.array(['left', 'right'])
        self.nr_states = len(self.state_names)
        self.n_trials = n_trials
        self.nr_actions = 2
        self.states_actions_outcomes = {
            'initiation': {
                'left': ['left_state', 'right_state'],
                'right': ['right_state', 'left_state']
            },
            'left_state': {
                'left': ['terminal1'],
                'right': ['terminal2']
            },
            'right_state': {
                'left': ['terminal3'],
                'right': ['terminal4']
            },
            'terminal1': {},
            'terminal2': {},
            'terminal3': {},
            'terminal4': {}
        }

        self._fill_adjacency_matrix()
        self.set_transition_probabilities()
        self.create_graph()

        self.reward_traces = self.load_reward_traces()
        self.reward_probs = self.reward_traces[:, 0]  # [1, 0, .1, .1]

        self.start_state = 0
        self.curr_state = self.start_state
        self.curr_action_idx = 0

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))

        for idx in range(self.nr_states):
            state_name = self.state_names[idx]

            actions = self.states_actions_outcomes[state_name]
            for act, destination_list in actions.items():
                for dest in destination_list:
                    destination_idx = list(self.states_actions_outcomes.keys()).index(dest)
                    self.adjacency_graph[idx, destination_idx] = 1

    def set_transition_probabilities(self):
        """Set the transition probability matrix.
        """
        self.transition_probabilities = np.zeros((self.nr_states, self.nr_actions, self.nr_states))
        for state, act_outcome in self.states_actions_outcomes.items():
            s_idx = self.get_state_idx(state)
            for a_idx, (act, possible_destinations) in enumerate(act_outcome.items()):
                for i, d in enumerate(possible_destinations):
                    d_idx = self.get_state_idx(d)
                    if len(possible_destinations) == 1:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = 1
                    elif len(possible_destinations) > 1 and i == 0:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = self.common_probability
                    elif len(possible_destinations) > 1 and i == 1:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = self.rare_probability

    def plot_graph(self, map_variable=None, node_size=1500, **kwargs):
        positions = {0: [0.75, 2], 1: [.25, 1], 2: [1.25, 1],
                     3: [0, 0], 4: [.5, 0], 5: [1, 0], 6: [1.5, 0]}
        self.show_graph(map_variable=map_variable, node_size=node_size,
                        layout=positions, **kwargs)

    def generate_reward_traces(self, **kwargs):
        """Generate reward traces per reward port per trial using a Gaussian random walk and save in file.
        :return:
        """
        r1 = bounded_random_walk(self.n_trials, **kwargs)
        r2 = [1-r for r in r1]  # bounded_random_walk(self.n_trials, **kwargs)
        #r2 = bounded_random_walk(self.n_trials, **kwargs)
        rewards = np.array([r1, r2, r1[::-1], r2[::-1]])
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        np.save(file_path, rewards)

    def load_reward_traces(self):
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        try:
            reward_traces = np.load(file_path)
            print('Loaded reward traces from file.')
        except FileNotFoundError:
            print('Warning: No reward traces file was found so I generate a new one.')
            self.generate_reward_traces(avg_stepsize=.05, sigma=.0005)
            reward_traces = np.load(file_path)
        return reward_traces

    def reset(self):
        self.curr_state = self.start_state

    def get_possible_actions(self, state_idx):
        state_name = self.state_names[state_idx]
        possible_actions = list(self.states_actions_outcomes[state_name].keys())
        return possible_actions

    def get_state_idx(self, state_name):
        return self.state_names.index(state_name)

    def is_terminal(self, state_idx):
        state_name = self.state_names[state_idx]
        return self.states_actions_outcomes[state_name] == {}


class SquareGrid(ControlTask):
    def __init__(self, num_rows=3, num_cols=3):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.original_goal_x = None
        self.original_goal_y = None
        self.goal_x = None
        self.goal_y = None
        self.nr_occupiable_states = None
        self.absorbing_states = []

        self.matrix_MDP = self.get_matrix_MDP()

        self.start_x = 0
        self.start_y = 0

        self.curr_x = self.start_x
        self.curr_y = self.start_y

        self.nr_states = self.num_rows * self.num_cols
        self.state_indices = np.arange(self.nr_states)
        self.actions = ['up', 'right', 'down', 'left']
        self.action_idx = np.arange(len(self.actions))
        self.nr_actions = len(self.actions)

        self.reset_reward_func()

        self._fill_adjacency_matrix()
        self.create_graph()
        self.define_transition_probabilities()
        self.n_features = self.nr_states

    def get_matrix_MDP(self):
        return np.zeros((self.num_rows, self.num_cols))

    def reset_reward_func(self):
        if self.goal_x is not None and self.goal_y is not None:
            self.reward_func = np.zeros(self.nr_states)
            self.goal_state = self.get_state_idx(self.goal_x, self.goal_y)
            self.reward_func[self.goal_state] = 10

    def set_reward_func(self, rewards, set_goal=True):
        """Set the reward function of the environment.

        :param rewards: Vector of length self.nr_states containing the rewards for each states.
        :param set_goal: Reset the terminal state to be at the non-zero position of the reward vecctor.
        :return:
        """
        self.reward_func = rewards
        if set_goal:
            self.goal_state = np.where(rewards)[0][0]
            self.goal_x, self.goal_y = self.get_state_position(self.goal_state)

    def get_state_idx(self, x, y):
        """Given coordinates, return the state index.
        """
        idx = x + y * self.num_rows
        return idx

    def get_state_position(self, idx):
        """Given the state index, return the x, y position.
        """
        x = idx % self.num_rows
        y = (idx - x) / self.num_rows

        return int(x), int(y)

    def get_next_state(self, origin, action):
        x, y = self.get_state_position(origin)
        if self.matrix_MDP[x][y] == -1:
            return origin

        if action == 'up' and y < self.num_rows:
            next_x = x
            next_y = y + 1
        elif action == 'right' and x < self.num_cols:
            next_x = x + 1
            next_y = y
        elif action == 'down' and y >= 0:
            next_x = x
            next_y = y - 1
        elif action == 'left' and x >= 0:
            next_x = x - 1
            next_y = y
        else:  # terminate
            next_state = self.nr_states
            return next_state
        if not self.location_in_maze(next_x, next_y):
            next_x = x
            next_y = y
        if self.matrix_MDP[next_x][next_y] == -1:
            next_x = x
            next_y = y
        next_state = self.get_state_idx(next_x, next_y)
        return next_state

    def get_reward(self, state, **kwargs):
        if self.reward_func is None:
            return None
        reward = self.reward_func[state]
        return reward

    def is_terminal(self, state_idx):
        rewarded_states = np.flatnonzero(self.reward_func)
        return state_idx in rewarded_states

    def current_state_is_terminal(self):
        if self.curr_x == self.goal_x and self.curr_y == self.goal_y:
            return True
        else:
            for loc in self.absorbing_states:
                if loc[0] == self.curr_x and loc[1] == self.curr_y:
                    return True
            return False

    def get_next_state_and_reward(self, origin, action):
        # If current state is terminal absorbing state:
        if origin == self.nr_states:
            return origin, 0

        next_state = self.get_next_state(origin, action)
        reward = self.get_reward(next_state, )
        return next_state, reward

    def get_adjacency_matrix(self):
        if self.adjacency_mat is None:
            self._fill_adjacency_matrix()
        return self.adjacency_mat

    def _fill_adjacency_matrix(self):
        self.adjacency_mat = np.zeros((self.nr_states, self.nr_states), dtype=np.int)
        self.idx_matrix = np.zeros((self.num_rows, self.num_cols), dtype=np.int)

        for row in range(len(self.idx_matrix)):
            for col in range(len(self.idx_matrix[row])):
                self.idx_matrix[row][col] = row * self.num_cols + col

        for row in range(len(self.matrix_MDP)):
            for col in range(len(self.matrix_MDP[row])):

                adj_locs = [[row + 1, col], [row - 1, col], [row, col + 1], [row, col - 1]]
                if self.matrix_MDP[row][col] != -1:
                    for adj_row, adj_col in adj_locs:
                        if self.location_in_maze(adj_row, adj_col):
                            if self.matrix_MDP[adj_row, adj_col] != -1:
                                self.adjacency_mat[self.idx_matrix[row, col]][self.idx_matrix[adj_row, adj_col]] = 1

    def location_in_maze(self, row, col):
        return row >= 0 and row < self.num_rows and col >= 0 and col < self.num_cols

    def get_possible_actions(self, state_idx):
        if state_idx == self.goal_state:
            return []
        else:
            return self.actions #list(self.action_idx)

    def define_transition_probabilities(self):
        self.transition_probabilities = np.zeros([self.nr_states, self.nr_actions, self.nr_states])

        for s in self.state_indices:
            x, y = self.get_state_position(s)
            moves = {0: [x, y + 1], 1: [x + 1, y], 2: [x, y - 1], 3: [x - 1, y]}
            if self.matrix_MDP[x][y] != -1:

                for a, loc in moves.items():
                    if self.location_in_maze(loc[0], loc[1]) and self.matrix_MDP[tuple(loc)] != -1:
                        sprime = self.get_state_idx(loc[0], loc[1])
                        self.transition_probabilities[s, a, sprime] = 1
                    else:
                        self.transition_probabilities[s, a, s] = 1

    def reset(self):
        """Reset agent to start position.
        """
        self.curr_x = self.start_x
        self.curr_y = self.start_y

    def get_current_state(self):
        """Return current state idx given current position.
        """
        current_state_idx = self.get_state_idx(self.curr_x, self.curr_y)
        return current_state_idx

    def act(self, action):
        """

        :param action:
        :return:
        """
        current_state = self.get_current_state()
        if self.reward_func is None and self.is_terminal(current_state):
            return 0
        else:
            next_state, reward = self.get_next_state_and_reward(current_state, action)
            next_x, next_y = self.get_state_position(next_state)
            self.curr_x = next_x
            self.curr_y = next_y
            return next_state, reward

    def set_rand_start_location(self):
        x = 0
        y = 0
        while not self.matrix_MDP[x, y] == 0:
            x, y = self.get_state_position(np.random.choice(np.arange(self.nr_states)))
        self.start_x = x
        self.start_y = y


class TransitionRevaluation(ControlTask):
    """This class simulates the transition revaluation experiment designed by Momennejad et al. (2017).
    """
    def __init__(self):
        ControlTask.__init__(self)
        self.nr_states = 7
        self.state_indices = list(range(self.nr_states))
        self.nr_actions = None
        self.actions = None
        self.n_features = self.nr_states

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 0,
            6: 0
        }

        self.reward_function = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 10,
            6: 1
        }

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[5, 0] = 10
        self.reward_function[6, 0] = 1

        self.transition_probabilities = self.define_transition_probabilities()

        self.possible_start_states = [1, 2]
        self.current_state = np.random.choice(self.possible_start_states)

        self._fill_adjacency_matrix()
        self.create_graph()

    def reset(self):
        self.current_state = np.random.choice(self.possible_start_states)

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :] = 0
                continue

            successor = self.transitions[predecessor]

            transition_probabilities[predecessor, successor] = 1
        return transition_probabilities

    def is_terminal(self, state_idx):
        return True if state_idx == 0 else False

    def act(self, action=None):
        """Gets next state given previous state.
        """
        next_state, reward = self.get_next_state_and_reward(self.current_state)
        self.current_state = next_state
        return next_state, reward

    def get_next_state_and_reward(self, current_state):
        next_state = self.get_next_state(current_state)
        reward = self.get_reward(current_state, next_state)
        return next_state, reward

    def get_next_state(self, current_state, action=None):
        return self.transitions[current_state]

    def get_reward(self, state, next_state):
        return self.reward_function[state, next_state]

    def get_current_state(self):
        return self.current_state

    def set_relearning_phase(self):
        self.possible_start_states = [3, 4]

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 6,
            4: 5,
            5: 0,
            6: 0
        }
        self._fill_adjacency_matrix()
        self.create_graph()

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))
        for s, sprime in self.transitions.items():
            if sprime is None:
                continue
            if self.transitions[s] == sprime:
                self.adjacency_graph[s, sprime] = 1

    def create_graph(self):
        """Create networkx graph from adjacency matrix, minus the terminal state.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix()[1:, 1:])

    def show_graph(self, layout=None, color_map=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)}
        if color_map is None:
            color_map = ['g', 'g', 'b', 'b', 'r', 'r']
        labels = {i: i+1 for i in range(self.nr_states-1)}
        nx.draw(self.graph, with_labels=True, pos=layout, node_size=node_size, labels=labels,
                node_color=color_map, **kwargs)


class ExplorationTree(ControlTask):
    """Exploration problem.

    In this problem, there are 2L+1 states and two possible actions that are randomly mapped to movements UP and DOWN
    in each state when the environment is generated. The rewards are always 0 except after reaching state s_2L which
    produces a reward of 1. States with odd indices and s_2L are terminal.
    """
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.nr_states = 2 * size + 1
        self.n_features = self.nr_states
        self.actions = ['up', 'down']
        self.nr_actions = len(self.actions)
        self.current_state = 0

    def reset(self):
        self.current_state = 0

    def get_next_state(self, origin, action):
        if action == 'up':
            return origin + 2
        elif action == 'down':
            return origin + 1
        else:
            raise ValueError('Illegal action')

    def is_terminal(self, state):
        return True if (state % 2 or state == 2 * self.size) else False

    def get_reward(self, state, **args):
        return 1. if state == self.size * 2 else 0.

    def act(self, action):
        next_state = self.get_next_state(self.current_state, action)
        reward = self.get_reward(next_state)
        self.current_state = next_state
        return next_state, reward


def bounded_random_walk(n_trials, lim=(.25, .75), avg_stepsize=.05, sigma=.005):
    rewards = [random.uniform(lim[0], lim[1])]
    for trial in range(n_trials-1):

        stepsize = random.gauss(avg_stepsize, sigma)

        if rewards[trial] + stepsize > lim[1]:
            r = rewards[trial] - stepsize
        elif rewards[trial] + stepsize < lim[0]:
            r = rewards[trial] + stepsize
        elif random.random() >= .5:
            r = rewards[trial] + stepsize
        else:
            r = rewards[trial] - stepsize
        rewards.append(r)
    return rewards


class PlusMaze(ControlTask):
    """Packard & McGaugh experiment: start in same start arm for 3 trials, then probe trial in opposite.
    """
    def __init__(self):
        super().__init__()
        self.actions = ['up', 'right', 'down', 'left']
        self.nr_actions = len(self.actions)
        self.states_actions_outcomes = {
            0: {},
            1: {'up': 2, 'right': 1, 'down': 1, 'left': 1},
            2: {'up': 3, 'right': 2, 'down': 1, 'left': 2},
            3: {'up': 6, 'right': 8, 'down': 2, 'left': 4},
            4: {'up': 4, 'right': 3, 'down': 4, 'left': 5},
            5: {'up': 0, 'right': 0, 'down': 0, 'left': 0},
            6: {'up': 7, 'right': 6, 'down': 3, 'left': 6},
            7: {'up': 0, 'right': 0, 'down': 0, 'left': 0},
            8: {'up': 8, 'right': 9, 'down': 8, 'left': 3},
            9: {'up': 0, 'right': 0, 'down': 0, 'left': 0}
        }
        self.state_locs = {
            0: (np.nan, np.nan),
            1: (2, 0),
            2: (2, 1),
            3: (2, 2),
            4: (1, 2),
            5: (0, 2),
            6: (2, 3),
            7: (2, 4),
            8: (3, 2),
            9: (4, 2)
        }
        self.nr_states = len(self.states_actions_outcomes)
        self.state_indices = np.arange(self.nr_states)
        self.landmark_loc = self.get_state_location(3)

        self._fill_adjacency_matrix()
        self.start_state = 1
        self.current_state = self.start_state
        self.rewarded_terminal = 5
        self.random_sr = self.get_random_sr()

    def reset(self):
        self.current_state = self.start_state

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))

        for origin, outcome in self.states_actions_outcomes.items():
            for action, successor in outcome.items():
                self.adjacency_graph[origin, successor] = 1

    def act(self, action):
        """

        :param action:
        :return:
        """
        current_state = self.get_current_state()
        if self.reward_func is None and self.is_terminal(current_state):
            return 0
        else:
            next_state, reward = self.get_next_state_and_reward(current_state, action)
            self.current_state = next_state
            return next_state, reward

    def get_orientation(self):
        pass

    def get_next_state_and_reward(self, origin, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(origin):
            return origin, 0

        next_state = self.get_next_state(origin, action)
        reward = self.get_reward(origin, next_state)
        return next_state, reward

    def get_state_location(self, state_idx):
        return self.state_locs[state_idx]

    def get_state_idx(self, state_loc):
        for idx, loc in self.state_locs.items():
            if loc == (state_loc[0], state_loc[1]):
                return idx
        raise ValueError('Location does not correspond to state.')

    def get_next_state(self, origin, action):
        return self.states_actions_outcomes[origin][action]

    def get_reward(self, origin, next_state):
        if self.is_terminal(next_state) and origin == self.rewarded_terminal:
            return 1.
        else:
            return 0.

    def frep(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.is_terminal(state_idx):
            return np.zeros(self.nr_states)
        else:
            return np.eye(self.nr_states)[state_idx]

    def get_possible_actions(self, state_idx):
        return list(self.states_actions_outcomes[state_idx].keys())

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])

        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for a, action in enumerate(self.actions):
                successor = self.states_actions_outcomes[state][action]
                transition_matrix[state, successor] += policy[state][a]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m


class RevaluationExperiment(ControlTask):
    """This class contains the transition and reward revaluation experiment designed by Momennejad et al. (2017).
    """
    def __init__(self):
        ControlTask.__init__(self)
        self.nr_states = 7
        self.state_indices = list(range(self.nr_states))
        self.nr_actions = None
        self.actions = None
        self.n_features = self.nr_states

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 0,
            6: 0
        }

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[5, 0] = 10
        self.reward_function[6, 0] = 1

        self.transition_probabilities = self.define_transition_probabilities()

        self.possible_start_states = [1, 2]
        self.current_state = np.random.choice(self.possible_start_states)

        self._fill_adjacency_matrix()
        self.create_graph()

    def reset(self):
        self.current_state = np.random.choice(self.possible_start_states)

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :] = 0
                continue

            successor = self.transitions[predecessor]

            transition_probabilities[predecessor, successor] = 1
        return transition_probabilities

    def is_terminal(self, state_idx):
        return True if state_idx == 0 else False

    def act(self, action=None):
        """Gets next state given previous state.
        """
        next_state, reward = self.get_next_state_and_reward(self.current_state)
        self.current_state = next_state
        return next_state, reward

    def get_next_state_and_reward(self, current_state):
        next_state = self.get_next_state(current_state)
        reward = self.get_reward(current_state, next_state)
        return next_state, reward

    def get_next_state(self, current_state):
        return self.transitions[current_state]

    def get_reward(self, state, next_state):
        return self.reward_function[state, next_state]

    def get_current_state(self):
        return self.current_state

    def set_transition_revaluation_phase(self):
        """Change the transition structure to test transition revaluation.
        """
        self.possible_start_states = [3, 4]

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 6,
            4: 5,
            5: 0,
            6: 0
        }
        self._fill_adjacency_matrix()
        self.create_graph()

    def set_reward_revaluation_phase(self):
        """Change the reward structure to test reward revaluation.
        """
        self.possible_start_states = [3, 4]

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[5, 0] = 1
        self.reward_function[6, 0] = 10

    def set_learning_phase(self):
        """Set the reward and transition structure to the initial learning phase one.
        """
        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 0,
            6: 0
        }

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[5, 0] = 10
        self.reward_function[6, 0] = 1

    def set_mem_phase(self):
        self.possible_start_states = [1, 2]

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }
        self._fill_adjacency_matrix()
        self.create_graph()

    def _fill_adjacency_matrix(self):
        self.adjacency_mat = np.zeros((self.nr_states, self.nr_states))
        for s, sprime in self.transitions.items():
            if sprime is None:
                continue
            if self.transitions[s] == sprime:
                self.adjacency_mat[s, sprime] = 1

    def create_graph(self):
        """Create networkx graph from adjacency matrix, minus the terminal state.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix()[1:, 1:])

    def show_graph(self, layout=None, color_map=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)}
        if color_map is None:
            color_map = ['g', 'g', 'b', 'b', 'r', 'r']
        labels = {i: i+1 for i in range(self.nr_states-1)}
        nx.draw(self.graph, with_labels=True, pos=layout, node_size=node_size, labels=labels,
                node_color=color_map, **kwargs)

    def frep(self, s):
        if self.is_terminal(s):
            return np.zeros(self.nr_states)
        return np.identity(self.nr_states)[s]


class PolicyRevaluation(ControlTask):
    """This class contains the transition and reward revaluation experiment designed by Momennejad et al. (2017).
    """
    def __init__(self):
        ControlTask.__init__(self)
        self.nr_states = 7
        self.state_indices = list(range(self.nr_states))
        self.nr_actions = 2
        self.actions = [0, 1]
        self.n_features = self.nr_states

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: [2, 3],
            2: [4, 5],
            3: [5, 6],
            4: [0, 0],
            5: [0, 0],
            6: [0, 0]
        }

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[5, 0] = 15
        self.reward_function[6, 0] = 30

        self.transition_probabilities = self.define_transition_probabilities()

        self.possible_start_states = [1]
        self.current_state = np.random.choice(self.possible_start_states)

        self._fill_adjacency_matrix()
        self.create_graph()

    def reset(self):
        self.current_state = np.random.choice(self.possible_start_states)

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :] = 0
                continue

            successor = self.transitions[predecessor]

            transition_probabilities[predecessor, successor] = 1
        return transition_probabilities

    def is_terminal(self, state_idx):
        return True if state_idx == 0 else False

    def act(self, action=None):
        """Gets next state given previous state.
        """
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        return next_state, reward

    def get_next_state_and_reward(self, current_state, action):
        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(current_state, next_state)
        return next_state, reward

    def get_next_state(self, current_state, action):
        return self.transitions[current_state][action]

    def get_reward(self, state, next_state):
        return self.reward_function[state, next_state]

    def get_current_state(self):
        return self.current_state

    def set_policy_revaluation_phase(self):
        """Change the transition structure to test transition revaluation.
        """
        self.possible_start_states = [4, 5, 6]

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[4, 0] = 40
        self.reward_function[5, 0] = 15
        self.reward_function[6, 0] = 30

        self.create_graph()

    def set_reward_revaluation_phase(self):
        """Change the reward structure to test reward revaluation.
        """
        self.possible_start_states = [3, 4]

        self.reward_function = np.zeros((self.nr_states, self.nr_states))
        self.reward_function[5, 0] = 1
        self.reward_function[6, 0] = 10

    def set_mem_phase(self):
        self.possible_start_states = [1, 2]

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }
        self._fill_adjacency_matrix()
        self.create_graph()

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))
        for s, sprime in self.transitions.items():
            if sprime is None:
                continue
            if self.transitions[s] == sprime:
                self.adjacency_graph[s, sprime] = 1

    def create_graph(self):
        """Create networkx graph from adjacency matrix, minus the terminal state.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix()[1:, 1:])

    def show_graph(self, layout=None, color_map=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)}
        if color_map is None:
            color_map = ['g', 'g', 'b', 'b', 'r', 'r']
        labels = {i: i+1 for i in range(self.nr_states-1)}
        nx.draw(self.graph, with_labels=True, pos=layout, node_size=node_size, labels=labels,
                node_color=color_map, **kwargs)

    def frep(self, s):
        if self.is_terminal(s):
            return np.zeros(self.nr_states)
        return np.identity(self.nr_states)[s]


class ConditionBox(SquareGrid):
    def __init__(self, num_cols=3, num_rows=3):
        super().__init__(num_cols=num_cols, num_rows=num_rows)
        rewards = np.zeros(self.nr_states)
        self.set_reward_func(rewards, set_goal=False)

        step = 1.2
        xs, ys = np.meshgrid(np.arange(self.num_cols, step=step), np.arange(self.num_rows, step=step))

        self.state_distance = 1
        self.RBF_locs = [[x, y] for x, y in zip(xs.flatten(), ys.flatten())]
        self.RBF = [stats.multivariate_normal(loc, self.state_distance * .4 * np.eye(2)) for loc in self.RBF_locs]
        self.RBF_mat = np.zeros((self.nr_states, len(self.RBF)))
        for s in self.state_indices:
            loc = self.get_state_position(s)
            population_rate = [f.pdf(loc) / f.pdf(f.mean) for f in self.RBF]
            self.RBF_mat[s] = np.array(population_rate)
        self.n_features = len(self.RBF)

    def frep(self, s, rbf=True):
        if rbf:
            return self.RBF_mat[s]
        else:
            return np.identity(self.nr_states)[s]


class NChainEnv(gym.Env):
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """

    def __init__(self, n=5, slip=0.2, small=2, large=10):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            if self.state == 0:
                reward = self.small
                self.state = 0
            else:
                self.state -= 1
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = self.small
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large
            done = True

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

