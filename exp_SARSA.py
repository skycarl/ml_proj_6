from src.track import Track
from src.sarsa import SARSA


gamma = 0.95
eta = 0.25

track = Track('L-track.txt')
race = SARSA(track,
             gamma,
             eta=eta,
             eps=1,
             max_iter=10000000,
             k_decay=0.0000006,
             episode_steps=25,
             tol=0.001,
             err_dec=10000,
             train_race_steps=50)

policy = race.train(learn_curve_str=f'SARSA_{track.name}')
