"""Implements the SARSA algorithm."""

from src.race import *
from copy import deepcopy
import numpy as np
from itertools import product


class QLearning(Race):

    """Implements the SARSA algorithm for reinforcement learning.

    Attributes
    ----------
    verbose : bool
        Verbosity switch

    learn_curve : list
        Learning curve for training session

    Methods
    -------
    train()

    race()

    evaluate()
    """

    def __init__(self,
                 track,
                 gamma,
                 bad_crash=False,
                 velocity_range=(-5, 5),
                 accel_succ_prob=0.8,
                 accel=[-1, 0, 1],
                 crash_cost=10,
                 track_cost=1,
                 fin_cost=0,
                 max_iter=50,
                 tol=0.001,
                 verbose=True):
        """Initializes an object.

        Parameters
        ----------
        track : Track
            Track to train on

        gamma : float
            Discount rate

        bad_crash : bool, optional
            Whether to return to starting line when a crash
            occurs, by default False

        velocity_range : tuple, optional
            Limits for velocity, by default (-5, 5)

        accel_succ_prob : float, optional
            Probability that an acceleration will succeed, by default 0.8

        accel : list
            Acceptable acceleration options

        crash_cost : int
            Cost for crashing; default 5

        track_cost : int
            Cost for moving; default 1

        fin_cost : int
            Cost for crossing finish line; default 0

        max_iter : int
            Maximum number of iterations

        tol : float
            Tolerance for stopping

        verbose: bool
            Verbosity switch

        """
        super().__init__(track,
                         gamma,
                         bad_crash,
                         velocity_range,
                         accel_succ_prob,
                         accel,
                         crash_cost,
                         track_cost,
                         fin_cost,
                         max_iter,
                         tol,
                         verbose)

    def find_policy(self, gen_learn_curve):
        """Finds a policy using the SARSA algorithm

        Returns
        -------
        policy : np.array
            Policy found by SARSA

        gen_learn_curve : bool
            Whether to generate learning curve data
        """

        # Initialize v, pi, and Q(s, a)
        v = self.init_states()
        policy = self.init_policy()
        q_s_a = self.init_q()

        converged = False
        t = 0
        self.learn_curve = []

        while not converged:
            t += 1
            v_last = deepcopy(v)

            if self.verbose:
                print(f'\nEpoch = {t}')

            """
            # Find the best Q
            pi_loc = np.argmax(q_s_a[loc])
            policy[loc] = self.poss_actions[pi_loc]
            loc_q = (y_pos, x_pos, y_vel, x_vel, pi_loc)
            v[loc] = q_s_a[loc_q]
            """

            # Collect current performance for learning curve
            if gen_learn_curve:
                races = self.evaluate(policy=policy)
                self.learn_curve.append(np.mean(races))

            # Check if converged
            max_delta_v = np.max(np.abs(v - v_last))
            if self.verbose:
                print(f'Current max_delta_v = {max_delta_v}')

            if max_delta_v < self.tol:
                print('Stopped because training converged')
                converged = True

            if t >= self.max_iter:
                print(f'Stopped; max iters of {self.max_iter} reached')
                converged = True

        return policy

    def train(self, gen_learn_curve=False):
        """Develops a policy with the Value Iteration algorithm

        Parameters
        ----------
        gen_learn_curve : bool
            Whether to generate learning curve data

        Returns
        -------
        np.array
            Learned policy
        """

        assert type(gen_learn_curve) is bool, 'Must be boolean'

        # Generate the set of possible acceleration actions in all directions
        self.poss_actions = list(product(self.accel, repeat=2))

        self.policy = self.find_policy(gen_learn_curve)

        if gen_learn_curve:
            np.save(
                f'Learn_curve_{self.track.name}_{self.gamma}.npy', self.learn_curve)

        return self.policy
