import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from agents import KalmanSFestimation, LinearSF
from environments import HartRevaluation
from definitions import RESULTS_FOLDER
from config import ParamConfig
import os


if __name__ == '__main__':
    results_folder = os.path.join(RESULTS_FOLDER, 'precond_revaluation')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    config = ParamConfig(n_precond_trials=6, n_cond_trials=6*6, alpha_sr=.01)

    devaluation_task = HartRevaluation(devalue=True, n_precond_trials=config.n_precond_trials, n_cond_trials=config.n_cond_trials)
    nondevaluation_task = HartRevaluation(devalue=False, n_precond_trials=config.n_precond_trials, n_cond_trials=config.n_cond_trials)

    # KalmanSFEstimation simulation, devaluation condition
    agent_ksf_deval = KalmanSFestimation(env=devaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, q_var=config.q_var, prior_var=config.prior_var)
    results_devalue_kalman = agent_ksf_deval.run_one_experiment()
    V_c_dev_kalman = agent_ksf_deval.env.C @ agent_ksf_deval.W @ agent_ksf_deval.R  # compute value
    agent_ksf_deval.show_sr_mat('Kalman SR, devalued')

    # KalmanSFEstimation simulation, non-devaluation condition
    agent_ksf_nodeval = KalmanSFestimation(env=nondevaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, q_var=config.q_var, prior_var=config.prior_var)
    results_nondev_kalman = agent_ksf_nodeval.run_one_experiment()
    V_c_nondev_kalman = agent_ksf_nodeval.env.C @ agent_ksf_nodeval.W @ agent_ksf_nodeval.R  # compute value
    agent_ksf_nodeval.show_sr_mat('Kalman SR, non-devalued')

    # LinearSF simulation, devaluation condition
    agent_tdsr_deval = LinearSF(env=devaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, alpha_sr=config.alpha_sr)
    results_devalue_linear_sf = agent_tdsr_deval.run_one_experiment()
    V_c_dev_linear_sf = agent_tdsr_deval.env.C @ agent_tdsr_deval.W @ agent_tdsr_deval.R  # compute value
    agent_tdsr_deval.show_sr_mat('TD-SR, devalued')

    # LinearSF simulation, non-devaluation condition
    agent_tdsr_nodeval = LinearSF(env=nondevaluation_task, gamma=config.gamma, alpha_rw=config.alpha_rw, alpha_sr=config.alpha_sr)
    results_nondev_linear_sf = agent_tdsr_nodeval.run_one_experiment()
    V_c_nondev_linear_sf = agent_tdsr_nodeval.env.C @ agent_tdsr_nodeval.W @ agent_tdsr_nodeval.R  # compute value
    agent_tdsr_nodeval.show_sr_mat('TD-SR, non-devalued')

    # save simulation data
    np.save(os.path.join(results_folder, 'hart_sim_data.npy'),
            np.array([V_c_dev_kalman, V_c_nondev_kalman, V_c_dev_linear_sf, V_c_nondev_linear_sf]))

    # dataframes
    ksr_data = pd.DataFrame({'Value': [V_c_nondev_kalman, V_c_dev_kalman], 'Condition': ['Non-devalued', 'Devalued']})
    standard_sr_data = pd.DataFrame({'Value': [V_c_nondev_linear_sf, V_c_dev_linear_sf], 'Condition': ['Non-devalued', 'Devalued']})

    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    plt.sca(ax1)
    plt.text(.5, .5, 'Data')

    plt.sca(ax2)
    sns.barplot(data=standard_sr_data, y='Value', x='Condition')
    plt.title('Standard SR')
    plt.xlabel('')

    plt.sca(ax3)
    sns.barplot(data=ksr_data, y='Value', x='Condition')
    plt.title('Kalman SR')
    plt.xlabel('')

    sns.despine()
    plt.tight_layout()
    plt.show()
