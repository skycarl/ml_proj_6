from src.track import Track
from src.val_iter import ValueIteration


gamma = 0.95

track = Track('L-track.txt')
race = ValueIteration(track, gamma, max_iter=30, crash_cost=10, track_cost=1, tol=0.001)

policy = race.train(learn_curve_str=f'VI_{track.name}')
