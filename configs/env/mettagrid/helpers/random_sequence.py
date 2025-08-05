import random


def sample_sequence():
    """Return a random permutation of the four colors [0,1,2,3]."""
    return random.sample([0, 1, 2, 3], 4)
