import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def rollout_policy_fn(board):
    """
    rollout_policy_fn -- a coarse, ]
    fast version of policy_fn used in the rollout phase.
    """
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def pure_policy_value_fn(board):
    """
    a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    """
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0
