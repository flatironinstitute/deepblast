import numpy as np
from scipy.stats import multivariate_normal


def sample(transition_matrix,
           means, covs,
           start_state, n_samples,
           random_state):
    n_states, n_features, _ = covs.shape
    states = np.zeros(n_samples, dtype='int')
    emissions = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i == 0:
            prev_state = start_state
        else:
            prev_state = states[i - 1]
        state = random_state.choice(n_states,
                                    p=transition_matrix[:, prev_state])
        emissions[i] = random_state.multivariate_normal(means[state],
                                                        covs[state])
        states[i] = state
    return emissions, states


def make_data(T=20):
    """
    Sample data from a HMM model and compute associated CRF potentials.
    """

    random_state = np.random.RandomState(0)
    d = 0.2
    e = 0.1
    transition_matrix = np.array([[1 - 2*d, d, d],
                                  [1 - e, e, 0],
                                  [1 - e, 0, e]])
    means = np.array([[0, 0],
                      [10, 0],
                      [5, -5]])
    covs = np.array([[[1, 0],
                      [0, 1]],
                     [[.2, 0],
                      [0, .3]],
                     [[2, 0],
                      [0, 1]]])
    start_state = 0

    emissions, states = sample(transition_matrix, means, covs, start_state,
                               n_samples=T, random_state=random_state)
    emission_log_likelihood = []
    for mean, cov in zip(means, covs):
        rv = multivariate_normal(mean, cov)
        emission_log_likelihood.append(rv.logpdf(emissions)[:, np.newaxis])
    emission_log_likelihood = np.concatenate(emission_log_likelihood, axis=1)
    log_transition_matrix = np.log(transition_matrix)

    # CRF potential from HMM model
    theta = emission_log_likelihood[:, :, np.newaxis] \
            + log_transition_matrix[np.newaxis, :, :]

    return states, emissions, theta
