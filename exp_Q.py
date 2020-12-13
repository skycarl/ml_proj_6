from src.track import Track
from src.q_learn import QLearning


gamma = 0.95
eta = 0.25

track = Track('L-track.txt')
race = QLearning(track,
                 gamma,
                 eta=eta,
                 eps=1,
                 max_iter=1000000,
                 k_decay=0.000005,
                 episode_steps=25,
                 tol=0.001,
                 err_dec=10000,
                 train_race_steps=50)

policy = race.train(learn_curve_str=f'Q_{track.name}')
