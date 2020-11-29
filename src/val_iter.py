"""Implements the Value Iteration algorithm."""

import numpy as np
from itertools import product
from bresenham import bresenham


class ValueIteration():

    """Implements the Value Iteration algorithm for reinforcement learning.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, verbose=False):
        """Initializes an object.

        Parameters
        ----------
        verbose : bool
            Verbosity switch

        """

        self.verbose = verbose

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

        return states

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

        return q_s_a

    def __get_trajectory(self, point1, point2):
        """Gets a trajectory for a given point ('#' for a crash or 'F' for
        crossing the finish line). Uses Bresenham's line generation algorithm;
        adapted from
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

        crossed = False

        # Check points are good
        assert type(point1) == tuple, 'Location must be a tuple'
        assert len(point1) == 2, 'Location length invalid'
        assert type(point2) == tuple, 'Location must be a tuple'
        assert len(point2) == 2, 'Location length invalid'

        # Generate points along the way with Bresenham's algorithm
        m_new = 2 * (point2[0] - point1[0])
        m_err_new = m_new - (point2[1] - point1[1])

        y = point1[0]
        trajec = []
        for x in range(point1[1], point2[1]+1):

            point = (y, x)
            trajec.append(point)
            m_err_new = m_err_new + m_new

            if m_err_new >= 0:
                y += 1
                m_err_new = m_err_new - 2 * (point2[1] - point1[1])

        return trajec

    def __check_trajectory(self, point1, point2, pt_type=None):
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

    def __nearest_track_point(self, point, trajec):
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

        pt = None

        for pt in trajec:
            if self.track.get_point(pt) == '.':
                break

        assert pt is not None, 'Something very strange happened'
        return pt

    def __generate_action(self, pt, vel, accel):
        """Generates an action based on an initial passed point

        Parameters
        ----------
        pt : tuple
            Initial point (row, col)

        vel : int
            Velocity (y_vel, x_vel)

        accel : np.array
            Acceleration (y_acc, x_acc)

        Returns
        -------
        tuple
            New point based on velocity (row, col)
        """

        # The acceleration action may fail
        if self.accel_succ_prob < np.random.random():
            return pt

        # Generate velocity subject to limits
        v_y = vel[0] + accel[0]
        v_x = vel[1] + accel[1]

        if v_y < self.velocity_range[0]:
            v_y = self.velocity_range[0]

        if v_y > self.velocity_range[1]:
            v_y = self.velocity_range[1]

        if v_x < self.velocity_range[0]:
            v_x = self.velocity_range[0]

        if v_x > self.velocity_range[1]:
            v_x = self.velocity_range[1]

        # Generate position
        y = pt[0] + v_y
        x = pt[1] + v_x

        return y, x

    def __value_iteration(self):
        """Value iteration algorithm

        Returns
        -------
        policy : np.array
            Policy found by Value Iteration
        """

        # Initialize states to zeros
        v = self.__init_states()

        # Initialize Q(s, a)
        q_s_a = self.__init_q()

        converged = False
        t = 0
        last_point = self.track.start

        while not converged:
            t += 1

            if self.verbose:
                print(f'Time = {t}\n')
                self.track.show()

            # For all s in S
            for y_pos in range(v.shape[0]):
                for x_pos in range(v.shape[1]):
                    for y_vel in self.velocity_range:
                        for x_vel in self.velocity_range:

                            # Is this right?
                            curr_point = (y_pos, x_pos)

                            # Check if we've crashed
                            # TODO Is this the right spot for this?                            
                            if self.__check_trajectory(last_point,
                                                       curr_point,
                                                       '#'):
                                v[y_pos][x_pos][y_vel][x_vel] = self.crash_cost

                            # Check if we've crossed the finish line
                            if self.__check_trajectory(last_point,
                                                       curr_point,
                                                       'F'):
                                v[y_pos][x_pos][y_vel][x_vel] = self.fin_cost

                            # For all a in A:
                            for idx_act, accel in enumerate(self.poss_actions):

                                # Generate an action
                                pos = (y_pos, x_pos)
                                vel = (y_vel, x_vel)
                                point = self.__generate_action(pos, vel, accel)



                                # q_s_a[y_pos][x_pos][y_vel][x_vel][idx_act] = rew + (self.gamma * expected_value)





                            # last_point = (y_pos, x_pos)

        return # policy

    def train(self,
              track,
              gamma,
              bad_crash=False,
              velocity_range=(-5, 5),
              accel_succ_prob=0.8,
              accel=[-1, 0, 1],
              crash_cost=5,
              track_cost=1,
              fin_cost=0,
              vis=True):
        """Develops a policy with the Value Iteration algorithm

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

        vis : bool
            Whether to visualize the track in the console
        """

        self.track = track
        self.bad_crash = bad_crash
        self.velocity_range = range(velocity_range[0], velocity_range[1]+1)
        self.accel_succ_prob = accel_succ_prob
        self.accel = accel
        self.vis = vis
        self.crash_cost = crash_cost
        self.track_cost = track_cost
        self.fin_cost = fin_cost

        assert gamma >= 0 and gamma < 1, 'Discount rate must be between 0 and 1'
        self.gamma = gamma

        # Generate the set of possible acceleration actions in all directions
        self.poss_actions = list(product(self.accel, repeat=2))

        self.__value_iteration()



    def race(self, track, policy):
        """Predicts using a given policy.

        Parameters
        ----------
        track : Track
            Track on which to race

        policy : np.array
            Policy to use to race

        """

        pass

    def evaluate(self, df, response_col='class'):
        """Evaluates performance on a given dataset.

        Parameters
        ----------
        df : DataFrame
            DataFrame against which to run evaluation

        response_col : string
            Name of the response column in the DataFrame;
            default `class`

        Returns
        -------
        err : int
            Error for the dataset

        """

        # Get the classes
        responses = df[response_col].to_numpy()

        # Predict on the specified dataset
        pred_df = self.predict(df, response_col)
        preds = pred_df['Predicted']

        if self.kind == 'classify':
            # 0-1 loss for classification
            err_array = np.not_equal(preds, responses)
            err = np.sum(err_array)
        else:
            # Mean squared error for regression
            err = (1/len(responses))*np.sum((preds - responses)**2)

        return err
