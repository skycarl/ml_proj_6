"""Class for racetrack"""

import re
import numpy as np
from copy import deepcopy


class Track():
    """Class for racetrack.

    Attributes
    ----------
    dims : list
        Dimensions of track

    track : np.array
        Array representation of track

    start : tuple
        Tuple containing (row, col) of starting line (midpoint is used)

    finish : tuple
        Tuple containing finish line

    Methods
    -------
    get_start_line()
        Gets the starting line of the track denoted by 'S'

    get_finish_line()
        Gets the finish line of the track denoted by 'F'

    show()
        Prints the track

    get_point()
        Returns the char at a specified point

    """

    def __init__(self, filename):
        """Initializes a Track object. Assumes the track file will
        be in `data/` directory.

        Parameters
        ----------
        filename : string
            Filename in data/ directory
        """

        self.filename = filename

        # Read the file
        filename = 'data/' + filename
        with open(filename, 'r') as f:
            track_text = f.read()

        track_split = track_text.split('\n')

        # Get dimensions of track
        dims = track_split[0].split(',')
        self.dims = [int(i) for i in dims]

        self.track = track_split[1:-1]

        # Get starting line and finish line
        self.start = self.get_start_line(self.track)
        self.finish = self.get_finish_line(self.track)

    def get_start_line(self, track):
        """Finds the starting line of the track. Uses the
        midpoint of the line as the actual start.

        Parameters
        ----------
        track : list
            Track to search for line

        Returns
        -------
        tuple
            (row, col) of starting line
        """

        row_list = [idx for idx, ele in enumerate(track) if 'S' in ele]

        if len(row_list) == 1:
            row = row_list[0]
            col_tuple = re.search('S+', track[row]).span()
            col_list = list(range(col_tuple[0], col_tuple[1]))
            col = np.mean(col_list, dtype=np.int)

        else:
            col = track[row_list[0]].find('S')
            row = np.mean(row_list, dtype=np.int)

        return row, col

    def get_finish_line(self, track):
        """Finds the finish line of the track.

        Parameters
        ----------
        track : list
            Track to search for line

        Returns
        -------
        tuple
            Tuple (row, col) contains finish line as lists
        """

        row_list = [idx for idx, ele in enumerate(track) if 'F' in ele]

        if len(row_list) == 1:
            row = row_list[0]
            col_tuple = re.search('F+', track[row]).span()
            col = list(range(col_tuple[0], col_tuple[1]))

        else:
            row = row_list
            col = track[row_list[0]].find('F')

        return row, col

    def show(self, car=None):
        """Shows the track with the car (optional)

        Parameters
        ----------
        car : tuple, optional
            Location of car (row, col), by default None
        """

        if car is not None:
            # Check that the location is valid
            assert type(car) == tuple, 'Location must be a tuple'
            assert len(car) == 2, 'Car location length invalid'
            assert car[0] < self.dims[0], 'Car row invalid'
            assert car[1] < self.dims[1], 'Car column invalid'

            track_car = deepcopy(self.track)

            # Extract the row to a list and assign
            row = list(track_car[car[0]])
            row[car[1]] = 'X'

            # Delete the old row
            del track_car[car[0]]

            # Insert the new row
            row_str = ''.join(row)
            track_car.insert(car[0], row_str)

            [print(row) for row in track_car]
        else:
            [print(row) for row in self.track]

    def get_point(self, point):
        """Gets the character living at a provided point

        Parameters
        ----------
        point : tuple
            Tuple containing (row, col) to look up in the track

        Returns
        -------
        string
            Character at the provided point
        """

        # Check that the location is valid
        assert type(point) == tuple, 'Location must be a tuple'
        assert len(point) == 2, 'Point length invalid'
        assert point[0] < self.dims[0], 'Point row invalid'
        assert point[1] < self.dims[1], 'Point column invalid'
        assert point[0] >= 0, 'Row cannot be less than 0'
        assert point[1] >= 0, 'Column cannot be less than 0'

        return self.track[point[0]][point[1]]
