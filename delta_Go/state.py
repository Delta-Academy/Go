import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from go_base import BLACK, BOARD_SIZE, EMPTY_BOARD, PlayerMove
from liberty_tracker import LibertyTracker


@dataclass
class State:
    """Describes the State of a game of Go.

    Args:
        board: [BOARD_SIZE, BOARD_SIZE] np array of ints
        komi:  the handicap number of points given to white
        caps: number of captured stones for each player (b, w)
        lib_tracker: a LibertyTracker object. Used for caching available liberties for speedup.
                    Gives a speedup of 5x!
        ko: a tuple (x, y) of the last move that was a ko, or None if no ko
        recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
        board_deltas: a np.array of shape (n, go.N, go.N) representing changes
            made to the board at each move (played move and captures).
            Should satisfy next_pos.board - next_pos.board_deltas[0] == pos.board
        to_play: BLACK or WHITE
        player_color: Keeps track of white color (BLACK or WHITE) you are playing as
    """

    board: np.ndarray = EMPTY_BOARD
    lib_tracker: LibertyTracker = LibertyTracker.from_board(board)
    caps: Tuple[int, int] = (0, 0)
    ko: Optional[Tuple[int, int]] = None
    recent: Tuple[PlayerMove, ...] = tuple()
    board_deltas: np.ndarray = np.zeros([0, BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
    to_play: int = BLACK
    player_color: int = BLACK

    def __deepcopy__(self, memodict: Dict) -> "State":
        new_board = np.copy(self.board)
        new_lib_tracker = copy.deepcopy(self.lib_tracker)
        return State(
            board=new_board,
            lib_tracker=new_lib_tracker,
            caps=self.caps,
            ko=self.ko,
            recent=self.recent,
            board_deltas=self.board_deltas,
            to_play=self.to_play,
            player_color=self.player_color,
        )
