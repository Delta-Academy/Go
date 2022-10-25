# Delta Academy version of PettingZoo go_base. Originally from tensorflow

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code from: https://github.com/tensorflow/minigo

"""A board is a NxN numpy array. A Coordinate is a tuple index into the board. A Move is a
(Coordinate c | None). A PlayerMove is a (Color, Move) tuple.

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
"""
import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

import coords

BOARD_SIZE = 9


# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)

# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

ALL_COORDS = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
EMPTY_BOARD = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)


PASS_MOVE = BOARD_SIZE**2


def _check_bounds(c: Tuple[int, int]) -> bool:
    return 0 <= c[0] < BOARD_SIZE and 0 <= c[1] < BOARD_SIZE


NEIGHBORS = {
    (x, y): list(filter(_check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]))
    for x, y in ALL_COORDS
}
DIAGONALS = {
    (x, y): list(
        filter(
            _check_bounds,
            [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)],
        )
    )
    for x, y in ALL_COORDS
}


class IllegalMove(Exception):
    pass


def int_to_coord(move: int) -> Optional[Tuple[int, int]]:
    return None if move == PASS_MOVE else (move // BOARD_SIZE, move % BOARD_SIZE)


@dataclass
class PlayerMove:
    color: int
    move: int

    def __hash__(self) -> int:
        return hash((self.color, self.move))


class PositionWithContext(namedtuple("SgfPosition", ["position", "next_move", "result"])):
    pass


def place_stones(board: np.ndarray, color: int, stones: Iterable[Tuple[int, int]]) -> None:
    for s in stones:
        board[s] = color


def find_reached(board: np.ndarray, c: Tuple[int, int]) -> Tuple[Set, Set]:
    color = board[c]
    chain = {c}
    reached = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] == color and n not in chain:
                frontier.append(n)
            elif board[n] != color:
                reached.add(n)
    return chain, reached


def is_koish(board, c) -> Optional[int]:
    """Check if c is surrounded on all sides by 1 color, and return that color."""
    if board[c] != EMPTY:
        return None
    neighbors = {board[n] for n in NEIGHBORS[c]}
    if len(neighbors) == 1 and EMPTY not in neighbors:
        return list(neighbors)[0]
    else:
        return None


def is_eyeish(board, c):
    """Check if c is an eye, for the purpose of restricting MC rollouts."""
    # pass is fine.
    if c is None:
        return
    color = is_koish(board, c)
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[c]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if board[d] not in (color, EMPTY):
            diagonal_faults += 1
    return None if diagonal_faults > 1 else color


class Group(namedtuple("Group", ["id", "stones", "liberties", "color"])):
    """
    stones: a frozenset of Coordinates belonging to this group
    liberties: a frozenset of Coordinates that are empty and adjacent to this group.
    color: color of this group
    """

    def __eq__(self, other):
        return (
            self.stones == other.stones
            and self.liberties == other.liberties
            and self.color == other.color
        )


def is_move_legal(
    move: Optional[Tuple[int, int]], board: np.ndarray, ko: Optional[Tuple[int, int]] = None
) -> bool:
    """Checks that a move is on an empty space, not on ko, and not suicide."""
    if move is None:
        return True
    return False if board[move] != EMPTY else move != ko


def all_legal_moves(board: np.ndarray, ko: Optional[Tuple[int, int]]) -> np.ndarray:
    "Returns a np.array of size go.N**2 + 1, with 1 = legal, 0 = illegal"
    # by default, every move is legal
    legal_moves = np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
    # ...unless there is already a stone there
    legal_moves[board != EMPTY] = 0

    # ...and retaking ko is always illegal
    if ko is not None:
        legal_moves[ko] = 0
    # Concat with pass move
    return np.arange(BOARD_SIZE**2 + 1)[
        np.concatenate([legal_moves.ravel(), [1]]).astype(bool)
    ]  # Make better


def pass_move(state, mutate: bool = False):
    pos = state if mutate else copy.deepcopy(state)
    pos.recent += (PlayerMove(pos.to_play, PASS_MOVE),)
    pos.board_deltas = np.concatenate(
        (np.zeros([1, BOARD_SIZE, BOARD_SIZE], dtype=np.int8), pos.board_deltas[:6])
    )
    pos.to_play *= -1
    pos.ko = None
    return pos


def play_move(state, move: int, color=None, mutate=False):
    # Obeys CGOS Rules of Play. In short:
    # No suicides
    # Chinese/area scoring
    # Positional superko (this is very crudely approximate at the moment.)
    if color is None:
        color = state.to_play

    pos = state if mutate else copy.deepcopy(state)

    coord = int_to_coord(move)
    if coord is None:
        pos = pass_move(state, mutate=mutate)
        return pos

    if not is_move_legal(coord, state.board, state.ko):
        raise IllegalMove(
            f'{"Black" if state.to_play == BLACK else "White"} coord at {coords.to_gtp(coord)} is illegal: \n{state}'
        )

    potential_ko = is_koish(state.board, coord)

    place_stones(pos.board, color, [coord])
    captured_stones = pos.lib_tracker.add_stone(color, coord)
    place_stones(pos.board, EMPTY, captured_stones)

    opp_color = color * -1

    new_board_delta = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
    new_board_delta[coord] = color
    place_stones(new_board_delta, color, captured_stones)

    if len(captured_stones) == 1 and potential_ko == opp_color:
        new_ko = list(captured_stones)[0]
    else:
        new_ko = None

    if pos.to_play == BLACK:
        new_caps = (pos.caps[0] + len(captured_stones), pos.caps[1])
    else:
        new_caps = (pos.caps[0], pos.caps[1] + len(captured_stones))

    pos.caps = new_caps
    pos.ko = new_ko
    pos.recent += (PlayerMove(color, move),)

    # keep a rolling history of last 7 deltas - that's all we'll need to
    # extract the last 8 board states.
    pos.board_deltas = np.concatenate(
        (new_board_delta.reshape(1, BOARD_SIZE, BOARD_SIZE), pos.board_deltas[:6])
    )
    pos.to_play *= -1
    return pos


def game_over(recent: Tuple[PlayerMove, ...]) -> bool:
    return len(recent) >= 2 and recent[-1].move == PASS_MOVE and recent[-2].move == PASS_MOVE


def score(board: np.ndarray, komi: float) -> float:
    """Return score from B perspective.

    If W is winning, score is negative.
    """
    working_board = np.copy(board)
    while EMPTY in working_board:
        unassigned_spaces = np.where(working_board == EMPTY)
        c = unassigned_spaces[0][0], unassigned_spaces[1][0]
        territory, borders = find_reached(working_board, c)
        border_colors = {working_board[b] for b in borders}
        X_border = BLACK in border_colors
        O_border = WHITE in border_colors
        if X_border and not O_border:
            territory_color = BLACK
        elif O_border and not X_border:
            territory_color = WHITE
        else:
            territory_color = UNKNOWN  # dame, or seki
        place_stones(working_board, territory_color, territory)

    return (
        np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - komi
    )


def result(board: np.ndarray, komi: float) -> int:
    score_ = score(board, komi)
    if score_ > 0:
        return 1
    elif score_ < 0:
        return -1
    else:
        return 0
