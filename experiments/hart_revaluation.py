import matplotlib.pyplot as plt
import numpy as np

from agents import KalmanSFestimation, LinearSF
from environments import HartRevaluation
from definitions import RESULTS_FOLDER
from config import ParamConfig
import os


if __name__ == '__main__':
    results_folder = os.path.join(RESULTS_FOLDER, 'hart_revaluation')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    config = ParamConfig(n_precond_trials=6, n_cond_trials=6*6, alpha_sr=.01)

    devaluation_task = HartRevaluation(devalue=True, n_precond_trials=config.n_precond_trials, n_cond_trials=config.n_cond_trials)
    nondevaluation_task = HartRevaluation(devalue=False, n_precond_trials=config.n_precond_trials, n_cond_trials=config.n_cond_trials)

    # KalmanSFEstimation simulation, devaluation condition
    agent_kalman = KalmanSFestimation(env=devaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, q_var=config.q_var, prior_var=config.prior_var)
    results_devalue_kalman = agent_kalman.run_one_experiment()
    V_c_dev_kalman = agent_kalman.env.C @ agent_kalman.W @ agent_kalman.R

    # KalmanSFEstimation simulation, non-devaluation condition
    agent_kalman = KalmanSFestimation(env=nondevaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, q_var=config.q_var, prior_var=config.prior_var)
    results_nondev_kalman = agent_kalman.run_one_experiment()
    V_c_nondev_kalman = agent_kalman.env.C @ agent_kalman.W @ agent_kalman.R

    # LinearSF simulation, devaluation condition
    agent_linear_sf = LinearSF(env=devaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, alpha_sr=config.alpha_sr)
    results_devalue_linear_sf = agent_linear_sf.run_one_experiment()
    V_c_dev_linear_sf = agent_linear_sf.env.C @ agent_linear_sf.W @ agent_linear_sf.R

    # LinearSF simulation, non-devaluation condition
    agent_linear_sf = LinearSF(env=nondevaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, alpha_sr=config.alpha_sr)
    results_nondev_linear_sf = agent_linear_sf.run_one_experiment()
    V_c_nondev_linear_sf = agent_linear_sf.env.C @ agent_linear_sf.W @ agent_linear_sf.R

    np.save(os.path.join(results_folder, 'hart_sim_data.npy'),
            np.array([V_c_dev_kalman, V_c_nondev_kalman, V_c_dev_linear_sf, V_c_nondev_linear_sf]))

    # Plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].bar([0, 1], [V_c_nondev_kalman, V_c_dev_kalman])
    ax[1].bar([0, 1], [V_c_nondev_linear_sf, V_c_dev_linear_sf])
    ax[0].set_xticks([0, 1], ['non-devalued', 'devalued'], rotation=15)
    ax[1].set_xticks([0, 1], ['non-devalued', 'devalued'], rotation=15)
    ax[0].set_title('Kalman SR')
    ax[1].set_title('TD SR')
    plt.tight_layout()
    plt.show()
