# TinyNNUE Edge

A 769 → 128 → 32 → 1 NNUE-style chess evaluator trained on Stockfish labels, quantized to int8 weights / int32 accumulators, and run by a ~340-line C++ core that does real integer arithmetic and incremental first-layer updates.

This is an integer-inference case study with chess as the test bed. It is not a chess engine (no search), not a Stockfish replacement, and not compatible with Stockfish `.nnue` files.

Writeup: [debtirthasaha.github.io/blog/2026/tinynnue-edge](https://debtirthasaha.github.io/blog/2026/tinynnue-edge/) · Weights: [MR0b0t/tinynnue-edge](https://huggingface.co/MR0b0t/tinynnue-edge) · Data: [MR0b0t/tinynnue-stockfish-labels](https://huggingface.co/datasets/MR0b0t/tinynnue-stockfish-labels)

## Results

Test split = 6,072 positions from games never seen in training. Labels are Stockfish 19 scores at 10,000 nodes.

| model | WDL-space MSE | cp MAE (\|label\| ≤ 1000) |
|---|---|---|
| always 0 | 0.05367 | 323.1 |
| material count | 0.01737 | 195.2 |
| TinyNNUE fp32 | 0.01086 | 146.9 |
| TinyNNUE int8 (per-tensor, shipped) | 0.01089 | 147.1 |

| | fp32 | int8 |
|---|---|---|
| weight file | 410,900 B | 103,244 B |
| output vs fp32, mean / p99 / max abs | — | 7.1 / 25.4 / 39.2 cp |

Latency per evaluation, AMD Ryzen 5 4600H (WSL2 Ubuntu 24.04, g++ 13.3, single thread), 20,500 positions × 20 reps:

| path | -O2 | -O3 -march=native | -O3 -march=native -ffast-math |
|---|---|---|---|
| fp32 full recompute | 3,934 / 4,685 | 3,830 / 4,815 | 1,115 / 1,439 |
| fp32 incremental | 3,072 / 3,533 | 2,850 / 3,300 | **192 / 264** |
| int8 full recompute | 2,473 / 3,663 | 1,572 / 2,022 | 1,415 / 1,813 |
| int8 incremental | 1,345 / 1,947 | 535 / 685 | 472 / 533 |

(median / p95 ns)

**int8 is not faster than fp32 here.** With strict IEEE float semantics the compiler may not reorder the fp32 dot-product sums, so they run as serial dependency chains, and int8 looks 2.9x (-O2) to 7.2x (-O3 native) faster. Allow reassociation with `-ffast-math` and fp32 incremental (192 ns) beats int8 incremental (472 ns). In this scalar C++ code, int8 values are widened to int32 before the arithmetic, so a 256-bit register holds 8 lanes on both paths. int8 gets no extra lanes and pays for the widening and for a 64-bit multiply in every requantization. The int8 payoff needs instructions that multiply 8-bit pairs directly (`vpmaddubsw`, VNNI, or an accelerator's int8 MAC array); that requires hand-written SIMD, which is out of scope here. What int8 does deliver: a 4x smaller model, and exact, drift-free incremental updates.

Correctness (`traces/special.txt` + `traces/random.txt`, 18,342 steps including castling, en passant, promotions with and without capture, under-promotion, and 5,316 undos): integer incremental == integer full recompute on every accumulator element and every output; C++ int output == Python int reference on every step.

All numbers come from one laptop CPU; no edge device was measured, so this is a CPU prototype.

Full tables, including the error breakdown and accumulator bounds: [results/RESULTS.md](results/RESULTS.md).

## How it works

```
Lichess PGN ─► data.py ─► positions.csv ─► train.py ─► model.safetensors
 (CC0)         Stockfish 19, 10k nodes      fp32, PyTorch        │
                                                                  ▼
               traces.py ◄── tinynnue_int8.bin ◄── quantize.py (PTQ, overflow bounds,
               python-chess makes/unmakes moves      bit-exact Python int reference)
               and writes feature deltas
                    │
                    ▼
               nnue.cpp  test | bench | eval     (no chess rules in C++)
```

**Features.** 12 piece planes × 64 squares + 1 side-to-move bit = 769 binary inputs, at most 33 active.

**Integer pipeline.**

```
acc1 = b1q + Σ_{active f} W1q[f]           int8 rows summed into int32
a1   = clamp((acc1·M1 + 2^23) >> 24, 0, 127)
acc2 = b2q + W2q · a1                      int8 × int8 widened to int32
a2   = clamp((acc2·M2 + 2^23) >> 24, 0, 127)
cp   = (b3q + W3q · a2) · M3 >> 24          integer centipawns
```

Every scale comes from the weights or from the fixed [0, 1] range of the clipped ReLU, never from the position. So `acc1` is a plain integer sum, and subtracting removed rows and adding new rows gives exactly the same bits as recomputing it. The float path drifts instead (max 0.00085 cp here) because float addition is not associative.

**Overflow.** Worst case over any input, not just the data: |acc1| ≤ 2,534 (fits int16 with lots of headroom), |acc2| ≤ 591,379, |acc3| ≤ 264,318 (need int32).

**Where the error comes from.** Simulating one rounding step at a time: layer-1 weights alone give 7.15 cp mean error, layer-2 weights 1.46, layer-3 weights 0.37, activations 2.35. Each evaluation sums about 23 rounded W1 rows, so their errors add up. The per-tensor W1 scale is also set by an outlier (max |w| = 0.62, p99 = 0.34), which wastes about half of the int8 range.

**Where the time goes.** On the int8 path (-O3 native), the incremental accumulator update is 70 ns and layers 2–3 are 402 ns. Once layer 1 is incremental, the dense 128×32 layer and its requantization are the cost. `-fopt-info-vec` shows that at -O2 GCC vectorizes only the layer-2 dot product (16-byte vectors), not the accumulator row adds or the requantization; at -O3 -march=native all of them use 32-byte vectors.

## Reproduce

```sh
conda create -n nnue python=3.11 -y && conda activate nnue
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chess numpy safetensors zstandard tqdm huggingface_hub

# data: Lichess 2013-01 dump + Stockfish 19 (stockfishchess.org)
curl -LO https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
python data.py --pgn lichess_db_standard_rated_2013-01.pgn.zst --engine <stockfish binary>   # ~40 min, 10 workers
python train.py        # ~1 min on CPU
python quantize.py     # export/ + results/quant_metrics.json
python traces.py       # traces/special.txt, random.txt, bench.txt

./run_cpp.sh           # Linux/WSL: build, test, bench
# the other two columns of the latency table:
make clean && CXXFLAGS="-O3 -march=native -std=c++17" TAG=x86_64_O3native ./run_cpp.sh
make clean && CXXFLAGS="-O3 -march=native -ffast-math -std=c++17" TAG=x86_64_O3native_fastmath ./run_cpp.sh
python report.py       # results/RESULTS.md
python demo.py --moves "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"
```

`data.py --target 170559` reproduces the exact 121,187 rows in `data/positions.csv` (single-threaded search at a fixed node budget with the hash cleared per position, so labels are deterministic for a given Stockfish binary).

## Files

| file | what |
|---|---|
| `features.py` | 769-feature encoding, feature diff, material baseline |
| `data.py` | PGN sampling, game-level split, Stockfish labeling and filters |
| `train.py` | fp32 model (EmbeddingBag first layer), WDL-space loss, metrics |
| `quantize.py` | PTQ, fixed-point multipliers, export, bit-exact int reference, error breakdown, overflow bounds |
| `traces.py` | python-chess plays moves/undos and writes feature-delta traces |
| `nnue.cpp` | C++ core: fp32/int8 × full/incremental, `test` / `bench` / `eval` |
| `demo.py` | CLI: evaluate a move sequence with the C++ core |
| `report.py` | results JSON → results/RESULTS.md |
| `prepare_hf.py` | stage (and optionally upload) the HF model and dataset repos |

## Limitations

- Accuracy is weak in absolute terms: 147 cp mean error against Stockfish. 100k positions and no search cannot do better, and accuracy was not the goal.
- Train loss ends at 0.0049, validation at 0.0111. The model overfits; the best-validation checkpoint is the one kept.
- Labels from 10k-node searches are noisy, positions come from 2013 Lichess games (many low-rated), and tactical positions were filtered out, so the test set is quiet positions only.
- Per-tensor symmetric quantization, PTQ only. Per-channel is evaluated in Python (6.7 vs 7.1 cp mean error) but not implemented in C++.
- Latency was measured under WSL2 on one laptop CPU (no edge device), single thread, with no core pinning. Medians are stable to a few percent across runs; p95 is noisier.
- No hand-written SIMD. Every speed comparison above is really a comparison of what the compiler auto-vectorizes, so it says little about int8 on hardware with native int8 dot-product instructions or MAC arrays.

## License and attribution

Code: MIT. Training positions: [Lichess open database](https://database.lichess.org/) (CC0). Labels produced with [Stockfish](https://stockfishchess.org/) 19 (GPLv3); Stockfish itself is not redistributed here. Architecture follows the ideas in the [nnue-pytorch documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), much simplified.
