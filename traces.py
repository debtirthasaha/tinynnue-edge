"""Generate feature-delta traces for the C++ core (tests, benchmark, demo).

    python traces.py           # writes traces/special.txt, traces/random.txt, traces/bench.txt

python-chess plays the moves (make AND unmake); for each step we write the
features to subtract, the features to add, the full feature list after the
move, and the Python reference outputs. The C++ side never sees a chess rule.
"""
import csv
import os
import random
from collections import Counter

import chess
import numpy as np
from safetensors.numpy import load_file

from features import MAX_ACTIVE, PAD, active_features, feature_delta
from quantize import fp32_forward, int_forward, load_int8

# (name, start FEN, ops). An op is a UCI move to push, or "pop" to undo the last move.
SPECIAL = [
    ("castling, all four", "r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1",
     ["e1g1", "e8c8", "pop", "pop", "e1c1", "e8g8", "pop", "pop"]),
    ("en passant, white", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
     ["e5f6", "pop", "e5e6", "pop"]),
    ("en passant, black", "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
     ["e4d3", "pop"]),
    ("promotion, quiet + under", "8/P5k1/8/8/8/8/6Kp/8 w - - 0 1",
     ["a7a8q", "h2h1n", "pop", "pop", "a7a8r", "h2h1b", "pop", "pop"]),
    ("promotion with capture", "1n4k1/P7/8/8/8/8/6K1/8 w - - 0 1",
     ["a7b8q", "pop", "a7b8n", "pop", "a7a8q", "pop"]),
    ("captures + recaptures", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     ["f3e5", "c6e5", "c4f7", "e5f7", "pop", "pop", "pop", "pop"]),
]


def move_kind(board: chess.Board, move: chess.Move) -> str:
    if board.is_castling(move):
        return "castling"
    if board.is_en_passant(move):
        return "en_passant"
    if move.promotion:
        return "promotion_capture" if board.is_capture(move) else "promotion"
    return "capture" if board.is_capture(move) else "quiet"


def play(board: chess.Board, ops, kinds: Counter):
    """Run ops on a copy of `board`; return [(removed, added, full)] with removed=None for the begin step."""
    board = board.copy(stack=False)
    cur = active_features(board)
    steps = [(None, None, cur)]
    for op in ops:
        if op == "pop":
            board.pop()
            kinds["undo"] += 1
        else:
            move = chess.Move.from_uci(op) if isinstance(op, str) else op
            assert board.is_legal(move), (board.fen(), op)
            kinds[move_kind(board, move)] += 1
            board.push(move)
        nxt = active_features(board)
        removed, added = feature_delta(cur, nxt)
        steps.append((removed, added, nxt))
        cur = nxt
    return steps


def random_ops(board: chess.Board, n: int, p_undo: float, rng: random.Random):
    """Random legal moves with random undos, like the make/unmake pattern of a search."""
    board = board.copy(stack=False)
    ops, depth = [], 0
    for _ in range(n):
        moves = list(board.legal_moves)
        if depth > 0 and (rng.random() < p_undo or not moves):
            board.pop()
            ops.append("pop")
            depth -= 1
        elif moves:
            m = rng.choice(moves)
            board.push(m)
            ops.append(m)
            depth += 1
    return ops


def write_trace(path, games, q, p):
    """games: list of step lists from play(). Computes reference outputs in one batch."""
    flat = [s for g in games for s in g]
    idx = np.full((len(flat), MAX_ACTIVE), PAD, dtype=np.int64)
    for i, (_, _, full) in enumerate(flat):
        idx[i, :len(full)] = full
    ref_int, ref_fp = int_forward(q, idx), fp32_forward(p, idx)
    fmt = lambda xs: f"{len(xs)} " + " ".join(map(str, xs)) if xs else "0"
    with open(path, "w") as f:
        for (removed, added, full), ri, rf in zip(flat, ref_int, ref_fp):
            if removed is None:
                f.write(f"B {fmt(full)} {ri} {rf:.4f}\n")
            else:
                f.write(f"S {fmt(removed)} {fmt(added)} {fmt(full)} {ri} {rf:.4f}\n")
    return len(flat)


def test_fens(n):
    with open("data/positions.csv") as f:
        return [r["fen"] for r in csv.DictReader(f) if r["split"] == "test"][:n]


def main():
    q = load_int8("export/tinynnue_int8.bin")
    p = load_file("checkpoints/model.safetensors")
    os.makedirs("traces", exist_ok=True)
    rng = random.Random(0)
    fens = test_fens(500)

    kinds = Counter()
    games = [play(chess.Board(fen), ops, kinds) for _, fen, ops in SPECIAL]
    n = write_trace("traces/special.txt", games, q, p)
    print(f"special.txt: {n} steps  {dict(kinds)}")

    kinds = Counter()
    games = []
    for fen in fens[:300]:
        b = chess.Board(fen)
        games.append(play(b, random_ops(b, 60, 0.3, rng), kinds))
    n = write_trace("traces/random.txt", games, q, p)
    print(f"random.txt: {n} steps  {dict(kinds)}")

    kinds = Counter()
    games = []
    for fen in fens:
        b = chess.Board(fen)
        games.append(play(b, random_ops(b, 40, 0.0, rng), kinds))
    n = write_trace("traces/bench.txt", games, q, p)
    print(f"bench.txt: {n} steps  {dict(kinds)}")


if __name__ == "__main__":
    main()
