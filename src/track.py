"""Class for racetrack"""

import re
import numpy as np

class Track():
    """Class for racetrack.

    Attributes
    ----------
    dims : list
        Dimensions of track

    track : np.array
        Numpy array representation of track

    start : tuple
        Tuple containing (rows, cols) of starting line

    finish : 

    Methods
    -------

    """

    def __init__(self, filename, verbose=False):
        """Initializes a Track object. Assumes the track file will
        be in `data/` directory.

        Parameters
        ----------
        filename : string
            Filename in data/ directory

        verbose : bool
            Verbosity switch

        """

        self.filename = filename
        self.verbose = verbose

        # Read the file
        filename = 'data/' + filename
        with open(filename, 'r') as f:
            track_text = f.read()

        track_split = track_text.split('\n')

        # Get dimensions of track
        dims = track_split[0].split(',')
        self.dims = [int(i) for i in dims]

        # Convert to numpy array
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
