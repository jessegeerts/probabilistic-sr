class ParamConfig:
    def __init__(self, gamma=.95, alpha_rw=.1, alpha_sr=.2, q_var=.001, obs_var=1., prior_var=1.,
                 n_precond_trials=6, n_cond_trials=6*6):
        self.gamma = gamma  # discount factor
        self.alpha_rw = alpha_rw  # learning rate for reward weights
        self.alpha_sr = alpha_sr  # learning rate for sr weights (for TDSR)
        self.q_var = q_var  # transition noise variance
        self.obs_var = obs_var  # observation noise variance
        self.prior_var = prior_var # prior variance for sr weights
        self.n_precond_trials = n_precond_trials # 8
        self.n_cond_trials = n_cond_trials #4*24