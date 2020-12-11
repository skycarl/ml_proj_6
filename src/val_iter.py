"""Implements the Value Iteration algorithm."""

from src.race import *
from copy import deepcopy
import numpy as np
from itertools import product


class ValueIteration(Race):

    """Implements the Value Iteration algorithm for reinforcement learning.

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
        super().__init__(track=track,
                         gamma=gamma,
                         bad_crash=bad_crash,
                         velocity_range=velocity_range,
                         accel_succ_prob=accel_succ_prob,
                         accel=accel,
                         crash_cost=crash_cost,
                         track_cost=track_cost,
                         fin_cost=fin_cost,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose)

    def init_q(self):
        """Initializes Q(s, a) of empty values

        Returns
        -------
        np.array
            Empty Q(s, a) array; shape is (rows, cols, velocity_range,
            velocity_range, possible acceleration options)
        """

        q_s_a = np.empty((self.track.dims[0],
                          self.track.dims[1],
                          len(self.velocity_range),
                          len(self.velocity_range),
                          len(self.poss_actions)))

        return q_s_a

    def find_policy(self, gen_learn_curve):
        """Finds a policy using the value iteration algorithm

        Returns
        -------
        policy : np.array
            Policy found by Value Iteration

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

            # For all s in S
            for y_pos in range(v.shape[0]):
                for x_pos in range(v.shape[1]):

                    # Fill wall points with crash cost
                    if self.track.get_point((y_pos, x_pos)) == '#':
                        v[y_pos, x_pos, :, :] = self.crash_cost
                        continue

                    for y_vel in self.velocity_range:
                        for x_vel in self.velocity_range:
                            loc = (y_pos, x_pos, y_vel, x_vel)

                            # For all a in A:
                            for idx_act, accel in enumerate(self.poss_actions):

                                # Generate an action
                                pos = (y_pos, x_pos)
                                vel = (y_vel, x_vel)
                                action = self.generate_action(pos, vel, accel)
                                pos_new = action[0:2]
                                vel_new = action[2:4]

                                if self.check_trajectory(pos, pos_new, 'F'):
                                    rew = self.fin_cost
                                else:
                                    rew = self.track_cost

                                # Get the values associated with the possible
                                # outcome, if it succeeds
                                loc_new = (
                                    pos_new[0], pos_new[1], vel_new[0], vel_new[1])
                                val_succ = v[loc_new]

                                # Find value if the action fails
                                fail_action = self.generate_action(pos, vel, (0, 0))
                                fail_pos = fail_action[0:2]
                                fail_vel = fail_action[2:4]
                                fail_loc = (
                                    fail_pos[0], fail_pos[1], fail_vel[0], fail_vel[1])
                                fail_val = v_last[fail_loc]

                                # Calculate the expected value of the possible outcomes
                                sum_poss = (self.accel_succ_prob * val_succ) + \
                                    (((1-self.accel_succ_prob)) * (fail_val))

                                # Get Q(s, a)
                                loc_act = (y_pos, x_pos, y_vel, x_vel, idx_act)
                                q_s_a[loc_act] = rew + (self.gamma * sum_poss)

                            # Find the best Q
                            pi_loc = np.argmax(q_s_a[loc])
                            policy[loc] = self.poss_actions[pi_loc]
                            loc_q = (y_pos, x_pos, y_vel, x_vel, pi_loc)
                            v[loc] = q_s_a[loc_q]

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

        assert type(gen_learn_curve) is bool, 'Must be boolean '

        # Generate the set of possible acceleration actions in all directions
        self.poss_actions = list(product(self.accel, repeat=2))

        self.policy = self.find_policy(gen_learn_curve)

        if gen_learn_curve:
            np.save(
                f'Learn_curve_{self.track.name}_{self.gamma}.npy', self.learn_curve)

        return self.policy
