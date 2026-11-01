"""Sample positions from a Lichess PGN dump and label them with Stockfish.

    python data.py --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst \
                   --engine engine/stockfish/stockfish-windows-x86-64-universal.exe

Design choices:
  * split by GAME (hash of the Lichess game id), never by position: positions
    from one game are highly correlated, so a position-level split would leak.
  * <= 3 positions per game, plies >= 10: skips the near-identical opening
    positions every game shares and keeps per-game correlation low.
  * label = Stockfish score at a fixed node budget, Threads=1, hash cleared per
    position (ucinewgame). Same binary + same nodes => same label, reproducible.
  * score stored from WHITE's point of view. The net has a side-to-move input
    but no board flip, so one fixed perspective keeps the target consistent.
  * dropped: positions in check, positions where the best move is a capture or
    promotion (the static eval cannot see the tactic the search found), and
    mate scores (no finite centipawn value).
"""
import argparse
import csv
import hashlib
import io
import multiprocessing as mp
import random
import time

import chess
import chess.engine
import chess.pgn
import zstandard

from features import material_eval

POSITIONS_PER_GAME = 3
MIN_PLY = 10


def game_split(game_id: str) -> str:
    h = int(hashlib.sha1(game_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 90 else ("val" if h < 95 else "test")


def sample_positions(pgn_path: str, target: int, seed: int):
    """Yield (game_id, split, ply, fen) until `target` candidates are produced."""
    n = 0
    with open(pgn_path, "rb") as fh:
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        while n < target:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            game_id = game.headers.get("Site", "").rsplit("/", 1)[-1]
            moves = list(game.mainline_moves())
            if not game_id or len(moves) < MIN_PLY + 10:
                continue
            rng = random.Random(f"{seed}:{game_id}")
            plies = sorted(rng.sample(range(MIN_PLY, len(moves)), POSITIONS_PER_GAME))
            board = game.board()
            for ply, move in enumerate(moves):
                if ply in plies and not board.is_check():
                    yield game_id, game_split(game_id), ply, board.fen()
                    n += 1
                board.push(move)


_engine = None
_nodes = None


def _init_worker(engine_path: str, nodes: int):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    _engine.configure({"Threads": 1, "Hash": 16})
    _nodes = nodes
    mp.util.Finalize(None, _engine.quit, exitpriority=10)


def _label(item):
    game_id, split, ply, fen = item
    board = chess.Board(fen)
    info = _engine.analyse(board, chess.engine.Limit(nodes=_nodes), game=object())
    score = info["score"].white()
    if score.is_mate():
        return None, "mate"
    best = info["pv"][0]
    if board.is_capture(best) or best.promotion:
        return None, "tactical"
    return (game_id, split, ply, fen, score.score(), material_eval(board), best.uci()), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", default="data/positions.csv")
    ap.add_argument("--target", type=int, default=170_559, help="candidate positions before filtering (170,559 -> the published 121,187 rows)")
    ap.add_argument("--nodes", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    candidates = list(sample_positions(args.pgn, args.target, args.seed))
    print(f"sampled {len(candidates)} candidate positions")

    counts = {"ok": 0, "mate": 0, "tactical": 0}
    t0 = time.time()
    with open(args.out, "w", newline="") as f, \
         mp.Pool(args.workers, _init_worker, (args.engine, args.nodes)) as pool:
        w = csv.writer(f)
        w.writerow(["game_id", "split", "ply", "fen", "cp_white", "material_white", "bestmove"])
        # imap (ordered) keeps the output file order deterministic
        for i, (row, status) in enumerate(pool.imap(_label, candidates, chunksize=64)):
            counts[status] += 1
            if row:
                w.writerow(row)
            if (i + 1) % 10_000 == 0:
                print(f"{i + 1}/{len(candidates)}  {counts}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s  {counts}")


if __name__ == "__main__":
    main()
