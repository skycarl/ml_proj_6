from src.track import Track
from src.q_learn import QLearning
import numpy as np


# Create track and race objects (doesn't matter that it's a Q-learning object
# because racing functionality is the same for all)
track = Track('L-track.txt')
race = QLearning(track, gamma=0.5, eta=None)

# Race with the specified policy
filename = 'arrays/name_here.npy'
policy = np.load(filename)
race.race(policy=policy)
