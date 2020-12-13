"""Implements the SARSA algorithm."""

from src.q_learn import QLearning
import numpy as np
from copy import deepcopy


class SARSA(QLearning):

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
                 eta,
                 episode_steps=10,
                 err_dec=100,
                 eps=0.3,
                 k_decay=0.00001,
                 train_race_steps=100,
                 bad_crash=False,
                 velocity_range=(-5, 5),
                 accel_succ_prob=0.8,
                 accel=[-1, 0, 1],
                 crash_cost=10,
                 track_cost=1,
                 fin_cost=0,
                 max_iter=500,
                 tol=0.001,
                 verbose=True):
        """Initializes an object.

        Parameters
        ----------
        track : Track
            Track to train on

        gamma : float
            Discount rate

        eta : float
            Learning rate

        episode_steps : int
            Number of steps per episode

        err_dec : int
            Number of steps for performance to stay the same (per `tol`) to
            consider training to have converged

        eps : float
            Epsilon for epsilon-greedy search

        k_decay : float
            Decay parameter

        train_race_steps : int
            Max number of racing steps to use while assessing performance
            during training

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
                         eta=eta,
                         episode_steps=episode_steps,
                         err_dec=err_dec,
                         eps=eps,
                         k_decay=k_decay,
                         train_race_steps=train_race_steps,
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

    def find_policy(self, learn_curve_str):
        """Finds a policy using the SARSA algorithm

        Returns
        -------
        policy : np.array
            Policy found by SARSA

        learn_curve_str : bool
            Whether to generate learning curve data
        """

        # Initialize Q(s, a) and policy
        q_s_a = self.init_q()
        policy = self.init_policy(q_s_a)

        converged = False
        t = 0
        self.learn_curve = []
        perf_history = []
        perf = np.inf
        eta = deepcopy(self.eta)
        epsilon = deepcopy(self.eps)

        while not converged:
            t += 1
            last_iter_perf = deepcopy(perf)
            q_s_a[self.track.finish[0], self.track.finish[1], :, :, :] = self.fin_cost

            # Check performance every 1k iters
            if t % 1000 == 0:
                perf = np.mean(self.evaluate(policy=policy, max_race_steps=self.train_race_steps))
                self.learn_curve.append(perf)

            # Export the policy and learning curve every 100k iters
            if t % 100000 == 0:
                np.save(f'arrays/policy_{learn_curve_str}_{t}.npy', policy)
                np.save(f'arrays/learn_curve_{learn_curve_str}_{t}.npy', self.learn_curve)

            # Initialize s randomly
            y_pos = np.random.randint(0, self.track.dims[0])
            x_pos = np.random.randint(0, self.track.dims[1])
            y_vel = np.random.randint(self.velocity_range[0], self.velocity_range[1])
            x_vel = np.random.randint(self.velocity_range[0], self.velocity_range[1])
            state = (y_pos, x_pos, y_vel, x_vel)

            # Ignore if we've generated a wall or finish line
            if self.track.get_point((y_pos, x_pos)) == '#':
                continue

            if self.track.get_point((y_pos, x_pos)) == 'F':
                continue

            if self.verbose:
                print(f'Average performance = {perf}')

            if self.verbose:
                print(f'\nEpisode {t}')

            # Decay epsilon
            if epsilon is not None:
                # epsilon *= (1.0 / (1.0 + self.k_decay * t))
                epsilon = 1 * np.exp(-self.k_decay*t)
                print(f'epsilon = {epsilon:.4f}')

            # Epsilon-greedy search for first action
                if np.random.random() < epsilon:
                    # Randomly choose an action
                    act_loc = np.random.randint(0, len(self.poss_actions)-1)
                else:
                    # Use the current Q(s, a) to determine an action
                    act_loc = np.argmax(q_s_a[state])

            # For each episode
            for _ in range(self.episode_steps):

                # Generate an action
                pos = state[0:2]
                vel = state[2:4]
                accel = self.poss_actions[act_loc]
                new_state = self.generate_action(pos, vel, accel, race=True)

                # See if we've crossed the finish line
                if self.check_trajectory(pos, new_state[0:2], 'F'):
                    rew = self.fin_cost
                else:
                    rew = self.track_cost

                # Epsilon-greedy search for second action
                if np.random.random() < epsilon:
                    # Randomly choose an action
                    act_loc_new = np.random.randint(0, len(self.poss_actions)-1)
                else:
                    # Use the current Q(s, a) to determine an action
                    act_loc_new = np.argmax(q_s_a[new_state])

                accel_prime = self.poss_actions[act_loc_new]
                new_state_prime = self.generate_action(new_state[0:2], new_state[2:4], accel_prime, race=True) # should pos, vel be new_state?

                # Update Q(s, a) based on this action
                q_loc_prime = (new_state[0], new_state[1], new_state_prime[2], new_state_prime[3], act_loc_new)
                q_loc = (pos[0], pos[1], vel[0], vel[1], act_loc)
                q_s_a[q_loc] = q_s_a[q_loc] + eta * (rew + self.gamma * q_s_a[q_loc_prime] - q_s_a[q_loc])

                # s <-- s'; a <-- a'
                state = deepcopy(new_state)
                act_loc = deepcopy(act_loc_new)

            # Update the policy
            pi_loc = np.argmax(q_s_a[state])
            policy[state] = self.poss_actions[pi_loc]

            # Check if converged
            if (np.abs(last_iter_perf - perf) < self.tol) and (perf < self.train_race_steps):
                perf_history.append(1)
            else:
                perf_history.append(0)

            last_n = perf_history[-self.err_dec:]
            if all(x == 1 for x in last_n):
                converged = True
                print(f'Converged; average performance was within {self.tol} for last {self.err_dec} steps')

            if t >= self.max_iter:
                print(f'Stopped; max iters of {self.max_iter} reached')
                converged = True

        return policy
