---
license: mit
tags:
- chess
- nnue
- quantization
- int8
- edge
datasets:
- MR0b0t/tinynnue-stockfish-labels
---

# TinyNNUE Edge

A 769 → 128 → 32 → 1 NNUE-style chess position evaluator (102,721 parameters), trained on Stockfish 19 labels and post-training-quantized to int8 weights with int32 accumulators. An integer-inference case study, not a chess engine: no search, not compatible with Stockfish `.nnue` files.

Code, tests and the C++ inference core: [github.com/debtirthasaha/tinynnue-edge](https://github.com/debtirthasaha/tinynnue-edge). Writeup: [debtirthasaha.github.io/blog/2026/tinynnue-edge](https://debtirthasaha.github.io/blog/2026/tinynnue-edge/).

## Files

| file | contents |
|---|---|
| `model.safetensors` | fp32 weights: `W1` [769,128] (feature-major), `b1` [128], `W2` [32,128], `b2` [32], `W3` [32], `b3` [1] |
| `tinynnue_int8.bin` | int8 weights, int32 biases, fixed-point multipliers; layout in `quantize.py:export_int8` |
| `tinynnue_fp32.bin` | the same fp32 weights in the raw layout `nnue.cpp` reads |
| `results/` | all metrics as JSON + `RESULTS.md` |

## Input / output

Input: 769 binary features. Index `plane*64 + square` for each piece (planes 0–5 white P,N,B,R,Q,K; 6–11 black; square a1=0 … h8=63), plus index 768 when White is to move. Output: centipawns from White's point of view (fp32: `400 * net(x)`; int8 path returns integer centipawns directly).

## Results (test split: 6,072 positions from held-out games)

| model | WDL-space MSE | cp MAE (\|label\| ≤ 1000) |
|---|---|---|
| material count | 0.01737 | 195.2 |
| fp32 | 0.01086 | 146.9 |
| int8 | 0.01089 | 147.1 |

int8 vs fp32 output: 7.1 cp mean, 25.4 cp p99, 39.2 cp max. Size: 410,900 B fp32 → 103,244 B int8.

Latency per evaluation, AMD Ryzen 5 4600H, single thread, g++ 13.3 (WSL2), medians: with -O3 -march=native, fp32 full 3,830 ns and int8 incremental 535 ns; adding -ffast-math (lets the compiler vectorize fp32 sums), fp32 incremental 192 ns and int8 incremental 472 ns. The int8 path is not faster than fp32 in this scalar C++ code; its benefits here are the 4x smaller file and exact incremental updates.

Integer incremental updates match full recomputation bit for bit over 18,342 test steps (castling, en passant, promotions, undo).

## Limitations

- 147 cp mean error vs Stockfish: weak as a chess evaluator. Trained on 108,960 quiet positions; overfits (train 0.0049 vs val 0.0111 WDL-MSE).
- Labels are 10k-node Stockfish searches (noisy); positions are from 2013 Lichess games; positions in check, positions whose best move is a capture or promotion, and mate scores are excluded.
- Per-tensor symmetric PTQ only. Latency was measured on one laptop CPU under WSL2.

## Attribution

Positions: Lichess open database (CC0). Labels: Stockfish 19 (GPLv3), used as a tool, not redistributed. Design follows the [nnue-pytorch documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), much simplified.
