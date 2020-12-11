"""Implements the Value Iteration algorithm."""

from copy import deepcopy
import os
import numpy as np
from itertools import product
from bresenham import bresenham
import time


class ValueIteration():

    """Implements the Value Iteration algorithm for reinforcement learning.

    Attributes
    ----------
    verbose : bool
        Verbosity switch

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
                 vis=True,
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

        vis : bool
            Whether to visualize the track in the console

        verbose: bool
            Verbosity switch

        """

        self.track = track
        self.bad_crash = bad_crash
        self.velocity_range = range(velocity_range[0], velocity_range[1]+1)
        self.accel_succ_prob = accel_succ_prob
        self.accel = accel
        self.vis = vis
        self.crash_cost = -crash_cost
        self.track_cost = -track_cost
        self.max_iter = max_iter
        self.tol = tol
        self.fin_cost = fin_cost
        self.verbose = verbose
        self.policy = None

        assert gamma >= 0 and gamma < 1, 'Discount rate must be between 0 and 1'
        self.gamma = gamma

    def __init_states(self):
        """Initialize the states to arbitrary (zero) values. States are
        position (x, y) and velocity at each position (v_x, v_y)

        Returns
        -------
        np.array
            Shape is (rows, cols, velocity_range, velocity_range)
        """

        
        states = np.zeros((self.track.dims[0],
                           self.track.dims[1],
                           len(self.velocity_range),
                           len(self.velocity_range)))

        states[self.track.finish[0], self.track.finish[1], :, :] = -self.fin_cost
        
        return states

    def __init_policy(self):
        """Initialize the policy to arbitrary (zero) values. States are
        position (x, y) and velocity at each position (v_x, v_y)

        Returns
        -------
        np.array
            Shape is (rows, cols, velocity_range, velocity_range, 2)
        """

        pol = np.zeros((self.track.dims[0],
                        self.track.dims[1],
                        len(self.velocity_range),
                        len(self.velocity_range)), dtype=(int, 2))

        return pol

    def __init_q(self):
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

        q_s_a[self.track.finish[0], self.track.finish[1], :, :, :] = -self.fin_cost
        
        return q_s_a

    def __get_trajectory(self, point1, point2):
        """Gets a trajectory for a given point ('#' for a crash or 'F' for
        crossing the finish line). Uses Bresenham's line generation algorithm;
        https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/.

        Parameters
        ----------
        point1 : tuple
            Starting point

        point2 : tuple
            Proposed ending point

        Returns
        -------
        list
            List of tuples of the trajectory
        """

        # Check points are good
        assert type(point1) == tuple, 'Location must be a tuple'
        assert len(point1) == 2, 'Location length invalid'
        assert type(point2) == tuple, 'Location must be a tuple'
        assert len(point2) == 2, 'Location length invalid'

        return list(bresenham(point1[0], point1[1], point2[0], point2[1]))

    def __check_trajectory(self, point1, point2, pt_type):
        """Checks a trajectory for a given point ('#' for a crash or 'F' for
        crossing the finish line).

        Parameters
        ----------
        point1 : tuple
            Starting point

        point2 : tuple
            Proposed ending point

        pt_type : string
            Point type; must be either '#' or 'F'

        Returns
        -------
        bool
            Whether we have encountered a point of the specified type
        """

        assert pt_type in ['F', '#'], 'Must be either wall or finish'
        crossed = False

        # Check if we've exceeded the bounds of the track accidentally
        if pt_type == '#':
            if point2[0] > self.track.dims[0] or point2[1] > self.track.dims[1]:
                return True

            if point2[0] < 0 or point2[1] < 0:
                return True

        trajec = self.__get_trajectory(point1, point2)

        # Check if any of the points are the point type we're after
        for point in trajec:
            if self.track.get_point(point) == pt_type:
                crossed = True
                break

        return crossed

    def __nearest_point(self, point):
        """Finds the nearest track point with a circular search

        Parameters
        ----------
        point : tuple
            Point to search near
        """

        nearest = None
        found = False
        rad = 0

        while not found:
            rad += 1

            # Generate possibilities in random order to reduce the chances
            # of getting stuck
            poss = (-rad, rad)
            circle = list(product(poss, repeat=2))
            np.random.shuffle(circle)

            # Search the possibilities
            for pt in circle:
                cand_pt = [sum(x) for x in zip(pt, point)]

                # Ignore points outside the track
                if cand_pt[0] < 0 or cand_pt[1] < 0:
                    continue

                if cand_pt[0] >= self.track.dims[0] or cand_pt[1] >= self.track.dims[1]:
                    continue

                pt_tup = (cand_pt[0], cand_pt[1])

                if self.track.get_point(pt_tup) == '.':
                    nearest = (cand_pt[0], cand_pt[1])
                    found = True
                    break

        assert nearest is not None, 'Could not find track point'
        return nearest

    def __nearest_point_along_traj(self, point, trajec):
        """Finds the nearest track point to the passed point, along the
        specified trajectory; this is to prevent "cheating"

        Parameters
        ----------
        point : tuple
            Point to find nearest open point against

        trajec : list
            Trajectory along which to find the open point

        Returns
        -------
        tuple
            Nearest point on the track
        """

        nearest = None

        for pt in reversed(trajec):
            # If an invalid point is provided (i.e. off the track),
            # return the original point
            if pt[0] < 0 or pt[1] < 0:
                nearest = trajec[0]
                break

            if point[0] >= self.track.dims[0] or point[1] >= self.track.dims[1]:
                nearest = trajec[0]
                break

            # Otherwise, find the nearest track point
            if self.track.get_point(pt) == '.':
                nearest = pt
                break

        # Fall back to a circular search (i.e. not along the trajectory) if
        # no track point is found (for the case where the trajectory is
        # completely inside of the wall points while training) or if the
        # discovered point is the original point itself
        if nearest is None or nearest == point:
            nearest = self.__nearest_point(point)

        return nearest

    def __generate_action(self, pt, vel, accel, race=False):
        """Generates an action based on an initial passed point

        Parameters
        ----------
        pt : tuple
            Initial point (row, col)

        vel : int
            Velocity (y_vel, x_vel)

        accel : np.array
            Acceleration (y_acc, x_acc)

        race : bool
            Whether we're in racing mode; if so, actions may fail

        Returns
        -------
        tuple
            New point and velocity (row, col, y_vel, x_vel)
        """

        # The acceleration action may fail
        if race and (self.accel_succ_prob < np.random.random()):
            return pt + vel

        # Generate velocity subject to limits
        v_y = vel[0] + accel[0]
        v_x = vel[1] + accel[1]

        if v_y < self.velocity_range[0]:
            v_y = self.velocity_range[0]

        if v_y > self.velocity_range[-1]:
            v_y = self.velocity_range[-1]

        if v_x < self.velocity_range[0]:
            v_x = self.velocity_range[0]

        if v_x > self.velocity_range[-1]:
            v_x = self.velocity_range[-1]

        # Generate position subject to limits of track
        y = min(max(pt[0] + v_y, 0), self.track.dims[0]-1)
        x = min(max(pt[1] + v_x, 0), self.track.dims[1]-1)

        new_pt = (y, x)
        new_vel = (v_y, v_x)

        assert v_y is not None, 'Something weird happened'
        assert v_x is not None, 'Something weird happened'

        # See if a crash occurred
        if self.__check_trajectory(pt, new_pt, '#'):
            new_vel = (0, 0)

            # Crash behavior depends on parameter
            if self.bad_crash:
                new_pt = self.track.start
            else:
                traj = self.__get_trajectory(pt, new_pt)
                new_pt = self.__nearest_point_along_traj(new_pt, traj)

        return new_pt + new_vel

    def __value_iteration(self):
        """Value iteration algorithm

        Returns
        -------
        policy : np.array
            Policy found by Value Iteration
        """

        # Initialize v and pi to zeros
        v = self.__init_states()
        policy = self.__init_policy()

        # Initialize Q(s, a)
        q_s_a = self.__init_q()

        converged = False
        t = 0

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
                                action = self.__generate_action(pos, vel, accel)
                                pos_new = action[0:2]
                                vel_new = action[2:4]

                                if self.__check_trajectory(pos, pos_new, 'F'):
                                    rew = self.fin_cost
                                else:
                                    rew = self.track_cost

                                # Get the values associated with the possible
                                # outcome, if it succeeds
                                loc_new = (pos_new[0], pos_new[1], vel_new[0], vel_new[1])
                                val_succ = v[loc_new]

                                # Find value if the action fails
                                fail_action = self.__generate_action(pos, vel, (0, 0))
                                fail_pos = fail_action[0:2]
                                fail_vel = fail_action[2:4]
                                fail_loc = (fail_pos[0], fail_pos[1], fail_vel[0], fail_vel[1])
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

            # TODO delete this if it's not right
            # TODO update this for vertical case??
            # Doesn't seem to have done anything
            # Reset finish line values
            v[self.track.finish[0], self.track.finish[1], :, :] = self.fin_cost

            # Check if converged
            max_delta_v = np.max(np.abs(v - v_last))
            if self.verbose:
                print(f'Current max_delta_v = {max_delta_v}')

            # TODO save these values for learning curve

            if max_delta_v < self.tol:
                print('Stopped because training converged')
                converged = True

            if t >= self.max_iter:
                print(f'Stopped; max iters of {self.max_iter} reached')
                converged = True

        return policy

    def train(self):
        """Develops a policy with the Value Iteration algorithm
        """

        # Generate the set of possible acceleration actions in all directions
        self.poss_actions = list(product(self.accel, repeat=2))

        self.policy = self.__value_iteration()

        return self.policy

    def race(self, policy=None, max_race_steps=300):
        """Runs a time trial with the trained policy

        Parameters
        ----------
        policy : np.array
            Policy to use to race; used to speed up experimentation

        max_race_steps : int
            Max number of steps for racing

        Returns
        -------
        int
            Number of moves that the time trial was completed in

        """
        self.max_race_steps = max_race_steps

        if policy is not None:
            self.policy = policy

        finished = False
        steps = 0
        pos = self.track.start
        vel = (0, 0)

        if self.vis:
            self.track.show(pos)

        while not finished:
            steps += 1

            if self.vis:
                os.system('clear')
                print(f'Step: {steps}')
                self.track.show(pos)
                print(f'Velocity = {vel}')
                time.sleep(0.2)

            # Get the acceleration
            acc = self.policy[pos + vel]

            # Get the action that the policy dictates
            new_pos = self.__generate_action(pos, vel, acc, race=True)

            # Check if we've crossed the finish line or crashed
            finished = self.__check_trajectory(pos, new_pos[0:2], 'F')

            # Unpack the tuple for next iter
            pos = new_pos[0:2]
            vel = new_pos[2:4]
            assert pos[0] >= 0, 'Row cannot be less than 0'
            assert pos[1] >= 0, 'Column cannot be less than 0'

            if steps > max_race_steps:
                finished = True
                print(f'Failed to find finish in less than {self.max_race_steps} steps')

        if self.vis:
            os.system('clear')
            self.track.show(pos)

        if self.verbose:
            print(f'\nTime trial completed in {steps} steps')

        return steps

    def evaluate(self, n_races):
        """Evaluates a policy by running n races.

        Parameters
        ----------
        n_races : int
            Number of races to run with the policy
        """        

        #for
        pass 
