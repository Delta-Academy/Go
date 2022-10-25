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
from typing import Dict, Optional, Tuple

import numpy as np

import coords

BOARD_SIZE = 9

ALLOW_SUICIDE = True

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)

# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

ALL_COORDS = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
EMPTY_BOARD = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)


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


class LibertyTracker:
    @staticmethod
    def from_board(board):
        board = np.copy(board)
        curr_group_id = 0
        lib_tracker = LibertyTracker()
        for color in (WHITE, BLACK):
            while color in board:
                curr_group_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                chain, reached = find_reached(board, coord)
                liberties = frozenset(r for r in reached if board[r] == EMPTY)
                new_group = Group(curr_group_id, frozenset(chain), liberties, color)
                lib_tracker.groups[curr_group_id] = new_group
                for s in chain:
                    lib_tracker.group_index[s] = curr_group_id
                place_stones(board, FILL, chain)

        lib_tracker.max_group_id = curr_group_id

        liberty_counts = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.uint8)
        for group in lib_tracker.groups.values():
            num_libs = len(group.liberties)
            for s in group.stones:
                liberty_counts[s] = num_libs
        lib_tracker.liberty_cache = liberty_counts

        return lib_tracker

    def __init__(self, group_index=None, groups=None, liberty_cache=None, max_group_id=1):
        # group_index: a NxN numpy array of group_ids. -1 means no group
        # groups: a dict of group_id to groups
        # liberty_cache: a NxN numpy array of liberty counts
        self.group_index = (
            group_index
            if group_index is not None
            else -np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int32)
        )
        self.groups = groups or {}
        self.liberty_cache = (
            liberty_cache
            if liberty_cache is not None
            else np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.uint8)
        )
        self.max_group_id = max_group_id

    def __deepcopy__(self, memodict):
        new_group_index = np.copy(self.group_index)
        new_lib_cache = np.copy(self.liberty_cache)
        # shallow copy
        new_groups = copy.copy(self.groups)
        return LibertyTracker(
            new_group_index,
            new_groups,
            liberty_cache=new_lib_cache,
            max_group_id=self.max_group_id,
        )

    def add_stone(self, color, c):
        assert self.group_index[c] == MISSING_GROUP_ID
        captured_stones = set()
        opponent_neighboring_group_ids = set()
        friendly_neighboring_group_ids = set()
        empty_neighbors = set()

        for n in NEIGHBORS[c]:
            neighbor_group_id = self.group_index[n]
            if neighbor_group_id != MISSING_GROUP_ID:
                neighbor_group = self.groups[neighbor_group_id]
                if neighbor_group.color == color:
                    friendly_neighboring_group_ids.add(neighbor_group_id)
                else:
                    opponent_neighboring_group_ids.add(neighbor_group_id)
            else:
                empty_neighbors.add(n)

        new_group = self._merge_from_played(
            color, c, empty_neighbors, friendly_neighboring_group_ids
        )

        # new_group becomes stale as _update_liberties and
        # _handle_captures are called; must refetch with self.groups[new_group.id]
        for group_id in opponent_neighboring_group_ids:
            neighbor_group = self.groups[group_id]
            if len(neighbor_group.liberties) == 1:
                captured = self._capture_group(group_id)
                captured_stones.update(captured)
            else:
                self._update_liberties(group_id, remove={c})

        self._handle_captures(captured_stones)

        if not ALLOW_SUICIDE and len(self.groups[new_group.id].liberties) == 0:
            raise IllegalMove(f"Move at {c} would commit suicide!\n")

        return captured_stones

    def _merge_from_played(self, color, played, libs, other_group_ids):
        stones = {played}
        liberties = set(libs)
        for group_id in other_group_ids:
            other = self.groups.pop(group_id)
            stones.update(other.stones)
            liberties.update(other.liberties)

        if other_group_ids:
            liberties.remove(played)
        assert stones.isdisjoint(liberties)
        self.max_group_id += 1
        result = Group(self.max_group_id, frozenset(stones), frozenset(liberties), color)
        self.groups[result.id] = result

        for s in result.stones:
            self.group_index[s] = result.id
            self.liberty_cache[s] = len(result.liberties)

        return result

    def _capture_group(self, group_id):
        dead_group = self.groups.pop(group_id)
        for s in dead_group.stones:
            self.group_index[s] = MISSING_GROUP_ID
            self.liberty_cache[s] = 0
        return dead_group.stones

    def _update_liberties(self, group_id, add=set(), remove=set()):
        group = self.groups[group_id]
        new_libs = (group.liberties | add) - remove
        self.groups[group_id] = Group(group_id, group.stones, new_libs, group.color)

        new_lib_count = len(new_libs)
        for s in self.groups[group_id].stones:
            self.liberty_cache[s] = new_lib_count

    def _handle_captures(self, captured_stones):
        for s in captured_stones:
            for n in NEIGHBORS[s]:
                group_id = self.group_index[n]
                if group_id != MISSING_GROUP_ID:
                    self._update_liberties(group_id, add={s})


class IllegalMove(Exception):
    pass


@dataclass
class PlayerMove:
    color: int
    move: Optional[Tuple[int, int]]

    def __hash__(self) -> int:
        return hash((self.color, self.move))


class PositionWithContext(namedtuple("SgfPosition", ["position", "next_move", "result"])):
    pass


def place_stones(board, color, stones) -> None:
    for s in stones:
        board[s] = color


def find_reached(board, c):
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


@dataclass
class State:

    """
    TODO: Improve docstring

    board: a numpy array
    n: an int representing moves played so far
    komi: a float, representing points given to the second player.
    caps: a (int, int) tuple of captures for B, W.
    lib_tracker: a LibertyTracker object
    ko: a Move
    recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
    board_deltas: a np.array of shape (n, go.N, go.N) representing changes
        made to the board at each move (played move and captures).
        Should satisfy next_pos.board - next_pos.board_deltas[0] == pos.board
    to_play: BLACK or WHITE
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
    # calculate which spots have 4 stones next to them
    # padding is because the edge always counts as a lost liberty.
    adjacent = np.ones([BOARD_SIZE + 2, BOARD_SIZE + 2], dtype=np.int8)
    adjacent[1:-1, 1:-1] = np.abs(board)
    # Such spots are possibly illegal, unless they are capturing something.
    # Iterate over and manually check each spot.

    # ...and retaking ko is always illegal
    if ko is not None:
        legal_moves[ko] = 0
    # Concat with pass move
    return np.arange(BOARD_SIZE**2 + 1)[
        np.concatenate([legal_moves.ravel(), [1]]).astype(bool)
    ]  # Make better


def pass_move(state, mutate: bool = False):
    pos = state if mutate else copy.deepcopy(state)
    pos.recent += (PlayerMove(pos.to_play, None),)
    pos.board_deltas = np.concatenate(
        (np.zeros([1, BOARD_SIZE, BOARD_SIZE], dtype=np.int8), pos.board_deltas[:6])
    )
    pos.to_play *= -1
    pos.ko = None
    return pos


def flip_playerturn(state, mutate=False):
    pos = state if mutate else copy.deepcopy(state)
    pos.ko = None
    pos.to_play *= -1
    return pos


def get_liberties(self):
    return self.lib_tracker.liberty_cache


def play_move(state, move, color=None, mutate=False):
    # Obeys CGOS Rules of Play. In short:
    # No suicides
    # Chinese/area scoring
    # Positional superko (this is very crudely approximate at the moment.)
    if color is None:
        color = state.to_play

    pos = state if mutate else copy.deepcopy(state)

    if move is None:
        pos = pass_move(state, mutate=mutate)
        return pos

    if not is_move_legal(move, state.board, state.ko):
        raise IllegalMove(
            f'{"Black" if state.to_play == BLACK else "White"} move at {coords.to_gtp(move)} is illegal: \n{state}'
        )

    potential_ko = is_koish(state.board, move)

    place_stones(pos.board, color, [move])
    captured_stones = pos.lib_tracker.add_stone(color, move)
    place_stones(pos.board, EMPTY, captured_stones)

    opp_color = color * -1

    new_board_delta = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
    new_board_delta[move] = color
    place_stones(new_board_delta, color, captured_stones)

    if len(captured_stones) == 1 and potential_ko == opp_color:
        new_ko = list(captured_stones)[0]
    else:
        new_ko = None

    if pos.to_play == BLACK:
        new_caps = (pos.caps[0] + len(captured_stones), pos.caps[1])
    else:
        new_caps = (pos.caps[0], pos.caps[1] + len(captured_stones))

    # pos.n += 1
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
    return len(recent) >= 2 and recent[-1].move is None and recent[-2].move is None


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
