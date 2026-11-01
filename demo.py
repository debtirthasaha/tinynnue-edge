"""CLI demo: play a line of moves, evaluate every position with the C++ core.

    python demo.py
    python demo.py --moves "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"
    python demo.py --fen "<fen>" --moves "..."

Python turns moves into feature deltas; the C++ binary updates the int8
accumulator incrementally and prints (int8 cp, fp32 cp). All scores are from
White's point of view. On Windows the Linux binary is run through WSL.
"""
import argparse
import os
import subprocess
from collections import Counter

import chess
from safetensors.numpy import load_file

from features import material_eval
from quantize import load_int8
from traces import play, write_trace

DEFAULT = "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7 Kxf7 Qf3+ Ke6 Nc3 Ncb4 O-O c6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--moves", default=DEFAULT)
    args = ap.parse_args()

    board = chess.Board(args.fen)
    sans, ops, materials = [], [], [material_eval(board)]
    b = board.copy()
    for san in args.moves.split():
        m = b.parse_san(san)
        sans.append(b.san(m))
        ops.append(m)
        b.push(m)
        materials.append(material_eval(b))

    steps = play(board, ops, Counter())
    write_trace("traces/demo.txt", [steps], load_int8("export/tinynnue_int8.bin"),
                load_file("checkpoints/model.safetensors"))
    cmd = ["./build/nnue", "eval", "export/tinynnue_int8.bin", "export/tinynnue_fp32.bin", "traces/demo.txt"]
    if os.name == "nt":
        cmd = ["wsl"] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split("\n")

    print(f"{'ply':>3}  {'move':<8} {'int8 (incr)':>11} {'fp32':>8} {'material':>8}  changed features")
    for i, line in enumerate(l for l in out if l):
        q, f = line.split()
        move = sans[i - 1] if i else "(start)"
        changed = "" if i == 0 else f"-{len(steps[i][0])} +{len(steps[i][1])}"
        print(f"{i:>3}  {move:<8} {int(q):>11} {float(f):>8.1f} {materials[i]:>8}  {changed}")


if __name__ == "__main__":
    main()
