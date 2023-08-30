import numpy as np


def exponentiate(M, exp):
    """Exponentiate a matrix element-wise. For a diagonal matrix, this is equivalent to matrix exponentiation.

    :param M:
    :param exp:
    :return:
    """
    num_rows = len(M)
    num_cols = len(M[0])
    exp_m = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            if M[i][j] != 0:
                exp_m[i][j] = M[i][j] ** exp

    return exp_m


def random_argmax(x):
    """Argmax operation, but if there are multiple maxima, return one randomly.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    b = np.flatnonzero(arg_maxes)
    choice = np.random.choice(b)
    return choice


def all_argmax(x):
    """Argmax operation, but if there are multiple maxima, return all.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    indices = np.flatnonzero(arg_maxes)
    return indices


def softmax(x, beta=2):
    """Compute the softmax function.

    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))


def to_agent_reference_frame(object_location, agent_location, agent_direction):
    """Shift reference frame to agent's current location and direction.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    translate = np.array(object_location) - np.array(agent_location)
    rotation_mat = rotation_matrix_2d(agent_direction).T
    result = rotation_mat.dot(translate)
    return np.asarray(result).squeeze()


def to_ext_reference_frame(object_location, agent_location, agent_direction):
    """Shift reference frame from agent's to world reference frame.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    rotation_mat = rotation_matrix_2d(agent_direction)
    rotated = rotation_mat.dot(object_location)
    rotated = np.asarray(rotated).squeeze()
    translate = np.add(rotated.T, np.array(agent_location))
    return translate


def rotation_matrix_2d(angle):
    """

    :param angle: In radians
    :return:
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.matrix([[c, -s], [s, c]])


def make_pd(covmat):
    """To avoid numerical instabilities, make covariance matrix positive definite.
    """
    eps = 10e-6
    zero = 10e-10

    sigma = covmat
    # TODO: Finish this.
    eigvals, eigvectors = np.linalg.eig(sigma)

    for i in range(len(eigvals)):
        if eigvals[i] <= zero:
            eigvals[i] = eps

    sigma = eigvectors @ np.diag(eigvals) @ eigvectors.T
    return sigma
