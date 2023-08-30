"""
agents.py
====================================
The module containing agent objects
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from environments import ControlTask, SharpeRevaluation
from utils import softmax


class KTDV(object):
    """Implementation of the Kalman TD value approximation algorithm with linear function approximation (Geist &
    Pietquin, 2012).

    We ignore control for now. The agent follows a random policy.
    """
    def __init__(self, environment=ControlTask(), gamma=.9, inv_temp=2):
        self.env = environment
        self.actions = self.env.actions

        # Parameters
        self.transition_noise = .005 * np.eye(self.env.nr_states)
        self.gamma = gamma
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = 1

        # Initialise priors
        self.prior_theta = np.zeros(self.env.nr_states)
        self.prior_covariance = np.eye(self.env.nr_states)

        self.theta = self.prior_theta
        self.covariance = self.prior_covariance

    def set_params(self, params):
        for key, value in params.items():
            if key == 'gamma':
                self.gamma = value
            if key == 'transition_var':
                self.transition_var = value
                self.transition_cov = value * np.eye(self.env.n_features)
            if key == 'observation_var':
                self.observation_noise_variance = value
            if key == 'prior_var':
                self.prior_covariance = value * np.eye(self.env.n_features)
                self.covariance = self.prior_covariance

    def train_one_episode(self, random_policy=False, fixed_policy=False):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.env.frep(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # Observe transition and reward;
            if random_policy:
                a = np.random.choice(self.actions)
            elif fixed_policy:
                a = 1
            else:
                a = self.select_action()

            next_state, reward = self.env.act(a)

            next_features = self.env.frep(next_state)
            H = features - self.gamma * next_features  # Temporal difference features

            # Prediction step;
            a_priori_covariance = self.covariance + self.transition_noise

            # Compute statistics of interest;
            #V = np.dot(features, self.theta)
            r_hat = np.dot(H, self.theta)
            delta_t = reward - r_hat  # the "prediction error". Actually expected value of the prediction error
            residual_cov = np.dot(H, np.matmul(a_priori_covariance, H)) + self.observation_noise_variance

            # Correction step;
            kalman_gain = np.matmul(a_priori_covariance, H) * residual_cov**-1
            self.theta = self.theta + kalman_gain * delta_t  # weight update
            self.covariance = a_priori_covariance - np.outer(kalman_gain, residual_cov*kalman_gain)

            # Store results
            results[t] = {'weights': self.theta,
                          'cov': self.covariance,
                          'K': kalman_gain,
                          'dt': delta_t,
                          'r': reward,
                          'state': s,
                          'rhat': r_hat,
                          }

            s = next_state
            features = self.env.frep(s)
            t += 1

        return results

    def select_action(self):
        Q = []
        for idx, a in enumerate(self.actions):
            s_prime = self.env.get_destination_state(self.env.get_current_state(), idx)[0]
            features = self.get_feature_representation(s_prime)
            V = np.dot(features, self.theta)
            if self.env.is_terminal(s_prime):
                V = 1
            Q.append(V)

        action = np.random.choice(self.actions, p=softmax(Q, beta=self.inv_temp))
        return action

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.env.is_terminal(state_idx):
            return np.zeros(self.env.nr_states)
        else:
            return np.eye(self.env.nr_states)[state_idx]

    def get_value_uncertainty(self, state_idx):
        variance = self.get_feature_representation(state_idx) @ self.covariance @ \
                   self.get_feature_representation(state_idx)
        return variance


class KalmanSFControl(object):
    """Implementation of Kalman TD for successor features algorithm with linear function approximation.
    Control is implemented using 1-step lookahead and softmax action selection."""
    def __init__(self, environment=ControlTask(), gamma=.9, inv_temp=2., rw=False):

        self.env = environment
        self.actions = self.env.actions

        self.rw = rw

        # Parameters
        self.transition_cov = .0001 * np.eye(self.env.n_features)
        self.transition_var = None

        self.r_transition_noise = .01 * np.eye(self.env.n_features)

        self.gamma = gamma
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = 1
        self.r_observation_noise_variance = 1

        # Initialise priors
        self.prior_W = np.eye(self.env.n_features)  # np.zeros((self.env.n_features, self.env.n_features))
        self.prior_covariance = np.eye(self.env.n_features) * .1

        self.prior_R = np.zeros(self.env.n_features)
        self.prior_reward_covariance = np.eye(self.env.n_features)

        self.R = self.prior_R

        self.W = self.prior_W
        self.covariance = self.prior_covariance
        self.reward_covariance = self.prior_reward_covariance
        self.results = {}

    def set_params(self, params):
        for key, value in params.items():
            if key == 'gamma':
                self.gamma = value
            if key == 'transition_var':
                self.transition_var = value
                self.transition_cov = value * np.eye(self.env.n_features)
            if key == 'observation_var':
                self.observation_noise_variance = value
            if key == 'prior_var':
                self.prior_covariance = value * np.eye(self.env.n_features)
                self.covariance = self.prior_covariance

    def write_results(self, t, s, r, kalman_gain, successor_error, resid_var):
        self.results[t] = {'weights': self.W,
                           'cov': self.covariance,
                           'k': kalman_gain,
                           'SPE': successor_error,
                           'state': s,
                           'reward': r,
                           'V': self.W.T @ self.R,
                           'resid_var': resid_var}

    def train_one_episode(self, random_policy=False, fixed_policy=False, right_policy=False):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.env.frep(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # Observe transition and reward;
            if random_policy:
                a = np.random.choice(self.actions)
            elif fixed_policy:
                a = 0
            elif right_policy:
                a = 1
            else:
                a = self.select_action(s)

            next_state, reward = self.env.act(a)

            next_features = self.env.frep(next_state)
            H = features - self.gamma * next_features  # Temporal difference features

            a_priori_r_covariance = (self.reward_covariance + self.r_transition_noise)

            residual_cov_rw = features @ a_priori_r_covariance @ features.T + \
                              self.r_observation_noise_variance

            kalman_gain = (self.covariance + self.transition_cov) @ H
            r_kalman_gain = a_priori_r_covariance @ features.T / residual_cov_rw

            lambda_t = np.dot(H, (self.covariance + self.transition_cov) @ H) + self.observation_noise_variance
            delta_t = features - self.W.T @ H
            rpe = reward - np.dot(self.R, features)

            # correction step
            if self.rw:
                self.R += .1 * rpe * features
            else:
                self.R += r_kalman_gain * rpe

            self.W += np.outer(kalman_gain, delta_t)
            self.covariance += self.transition_cov - np.outer(kalman_gain, kalman_gain) * lambda_t
            self.reward_covariance = self.reward_covariance - np.outer(r_kalman_gain, features) @ a_priori_r_covariance


            # Store results
            results[t] = {'weights': self.W,
                          'cov': self.covariance,
                          'K': kalman_gain,
                          'dt': delta_t,
                          'r': reward,
                          'state': s}

            s = next_state
            features = self.env.frep(s)
            t += 1

        return results

    def select_action(self, current_state, policy='softmax'):
        # one step lookahead
        next_states = [self.env.get_next_state(current_state, a) for a in self.env.actions]
        Q_values = [self.R @ self.W.T @ self.env.frep(s) for s in next_states]
        if policy == 'softmax':
            probab = softmax(Q_values, beta=self.inv_temp)
            action = np.random.choice(self.actions, p=probab)
        elif policy == 'e_greedy':
            if np.random.rand() < .4:
                action = np.random.choice(self.actions)
            else:
                action = np.argmax(Q_values)
        return action


class KalmanSFestimation(object):
    """Class implementing Kalman SFs for pure estimation tasks (i.e. no control).
    """
    def __init__(self, env=SharpeRevaluation(), gamma=.95, alpha_rw=.1, q_var=.001, prior_var=1.):
        self.env = env
        self.gamma = gamma
        self.n = self.env.nr_features
        self.alpha_rw = alpha_rw

        self.Q = q_var* np.eye(self.n)
        self.r_transition_noise = .01 * np.eye(self.n)
        self.gamma = gamma
        self.observation_noise_variance = 1

        # Initialise priors
        self.prior_R = np.zeros(self.env.nr_features)

        self.prior_W = np.eye(self.n)
        self.prior_covariance = np.eye(self.n) * prior_var
        self.prior_reward_covariance = np.eye(self.n)

        self.W = self.prior_W
        self.R = self.prior_R
        self.covariance = self.prior_covariance
        self.reward_covariance = self.prior_reward_covariance

        self.m_labels = ['M{}-{}'.format(i, j) for i, j in product(list(range(self.n)), list(range(self.n)))]

    def run_one_experiment(self):
        self.env.make_exp_seq()  # Randomise sequence for every experiment
        X = self.env.stim_seq
        n_time_steps, n_features = X.shape
        X = np.vstack((X, np.zeros(self.n)))

        r = self.env.reward_seq

        opto = self.env.opto_seq

        results = {}
        for t in range(n_time_steps):
            # compute temporal difference features
            if np.all(X[t] == np.zeros(X[t].shape)):
                H = np.zeros(X[t].shape)
            else:
                if np.any(X[t] + X[t+1]> 1) :  # stimulus repetition
                    H = X[t] - self.gamma * (X[t+1] - X[t])
                else:
                    H = X[t] - self.gamma * X[t+1]

            # prediction step
            a_priori_covariance = self.covariance + self.Q
            a_priori_r_covariance = (self.reward_covariance + self.r_transition_noise)

            lambda_t = np.dot(H, (self.covariance + self.Q) @ H) + self.observation_noise_variance
            residual_cov_rw = X[t] @ a_priori_r_covariance @ X[t].T + self.observation_noise_variance

            # Correction step;
            kalman_gain = a_priori_covariance @ H / lambda_t
            r_kalman_gain = a_priori_r_covariance @ X[t].T / residual_cov_rw

            delta_t = (opto[t] * X[t]) + (X[t] - self.W.T @ H)  # optogenetics modulated SPE
            rpe = r[t] - np.dot(self.R, X[t])

            self.W += np.outer(kalman_gain, delta_t)

            self.covariance += self.Q - np.outer(kalman_gain, kalman_gain) / lambda_t
            self.R += self.alpha_rw * rpe * X[t]
            self.reward_covariance = self.reward_covariance - np.outer(r_kalman_gain, X[t]) @ a_priori_r_covariance

            # Store results
            results[t] = {'SR': self.W,
                          'cov': copy.deepcopy(self.covariance),
                          'K': copy.deepcopy(kalman_gain),
                          'dt': delta_t,
                          'rpe': rpe,
                          'V': X[t] @ self.W.reshape(self.n, self.n) @ self.R,
                          'R': X[t] @ self.R}
        return results

    def print_m(self):
        return np.around(self.W.reshape(self.n, self.n), decimals=3)

    def show_sr_mat(self, title=None):
        plt.imshow(self.W); plt.colorbar()
        plt.xticks(range(7), self.env.feature_names.values())
        plt.yticks(range(7), self.env.feature_names.values())
        plt.title(title)
        plt.show()


class LinearSF(object):
    """Class implementing linear successor features for pure estimation tasks (i.e. no control).
    """
    def __init__(self, env=SharpeRevaluation(), gamma=.95, alpha_rw=.1, alpha_sr=.2):
        """
        :param env: Evaluation environment (e.g. SharpeRevaluation)
        :param gamma: discount factor
        :param alpha_rw: learning rate for reward weights
        :param alpha_sr: learning rate for successor features
        """
        self.env = env
        self.gamma = gamma
        self.n = self.env.nr_features
        self.alpha_rw = alpha_rw
        self.learning_rate_sr = alpha_sr

        self.gamma = gamma

        # Initialise priors
        self.prior_R = np.zeros(self.env.nr_features)

        self.prior_W = np.eye(self.n)
        self.prior_covariance = np.eye(self.n) * 1.
        self.prior_reward_covariance = np.eye(self.n)

        self.W = self.prior_W
        self.R = self.prior_R
        self.covariance = self.prior_covariance
        self.reward_covariance = self.prior_reward_covariance

        self.m_labels = ['M{}-{}'.format(i, j) for i, j in product(list(range(self.n)), list(range(self.n)))]

    def run_one_experiment(self):
        self.env.make_exp_seq()  # Randomise sequence for every experiment
        X = self.env.stim_seq
        n_time_steps, n_features = X.shape
        X = np.vstack((X, np.zeros(self.n)))

        r = self.env.reward_seq

        opto = self.env.opto_seq

        results = {}
        for t in range(n_time_steps):

            delta_t = (opto[t] * X[t]) + (X[t] + self.gamma * self.W.T @ X[t+1] - self.W.T @ X[t])  # optogenetics modulated SPE
            rpe = r[t] + X[t+1] @ self.W @ self.R - X[t] @ self.W @ self.R  # reward prediction error

            self.W += self.learning_rate_sr * np.outer(X[t], delta_t)

            self.R += self.alpha_rw * rpe * X[t]

            # Store results
            results[t] = {'SR': self.W,
                          'cov': copy.deepcopy(self.covariance),
                          'K': None,
                          'dt': delta_t,
                          'rpe': rpe,
                          'V': X[t] @ self.W.reshape(self.n, self.n) @ self.R,
                          'R': X[t] @ self.R}
        return results

    def print_m(self):
        return np.around(self.W.reshape(self.n, self.n), decimals=3)

    def show_sr_mat(self, title=None):
        plt.imshow(self.W); plt.colorbar()
        plt.xticks(range(7), self.env.feature_names.values())
        plt.yticks(range(7), self.env.feature_names.values())
        plt.title(title)
        plt.show()
