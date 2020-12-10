from src.track import Track
from src.val_iter import ValueIteration
import numpy as np


gamma = 0.9

track = Track('L-track.txt')
race = ValueIteration(track, gamma, max_iter=10, crash_cost=-9.9, track_cost=-1)

# policy = race.train()

filename = 'test.npy'
# np.save(filename, policy)

policy = np.load(filename)
# policy = np.load('example_pol2.npy')

race.race(policy=policy)
