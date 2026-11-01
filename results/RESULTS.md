# Results

## Accuracy vs Stockfish (test split, positions from held-out games)

| model | WDL-space MSE | cp MAE (\|label\| <= 1000) |
|---|---|---|
| always 0 | 0.05367 | 323.1 |
| material count | 0.01737 | 195.2 |
| TinyNNUE fp32 | 0.01086 | 146.9 |
| TinyNNUE int8 per-tensor (shipped) | 0.01089 | 147.1 |
| TinyNNUE int8 per-channel (analysis only) | 0.01087 | 147.1 |

n = 6072 test positions.

## Quantization error (int8 output vs fp32 output, centipawns)

| scheme | mean abs | p99 abs | max abs |
|---|---|---|---|
| per-tensor | 7.10 | 25.40 | 39.22 |
| per-channel | 6.70 | 21.73 | 30.46 |

Error by source (float simulation, only the named rounding step applied):

| quantized | mean abs cp | max abs cp |
|---|---|---|
| weights_only | 6.79 | 38.86 |
| activations_only | 2.35 | 12.13 |
| both_simulated | 7.16 | 38.29 |
| W1_only | 7.15 | 37.85 |
| W2_only | 1.46 | 6.03 |
| W3_only | 0.37 | 1.17 |

| weights | max abs | p99 abs | mean abs |
|---|---|---|---|
| W1 | 0.620 | 0.343 | 0.102 |
| W2 | 0.286 | 0.174 | 0.063 |
| W3 | 0.425 | 0.404 | 0.212 |

## Size

| file | bytes |
|---|---|
| fp32 weights | 410,900 |
| int8 weights + int32 biases | 103,244 |

## Accumulator bounds

| accumulator | worst case (any input) | observed max (test set) |
|---|---|---|
| acc1 (layer 1) | 2,534 (loose: 4,338) | 582 |
| acc2 (layer 2) | 591,379 (loose: 2,069,500) | 170,100 |
| acc3 (layer 3) | 264,318 | 111,367 |

## Latency per evaluation: `x86_64`

Hardware:  AMD Ryzen 5 4600H with Radeon Graphics, Linux x86_64, g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 -O2, 12 cores

20500 positions x 20 reps, timed in chunks of 64; avg 22.57 active features, avg 3.12 features changed per move.

| path | median ns | p95 ns |
|---|---|---|
| fp32_full | 3934 | 4685 |
| fp32_incremental | 3072 | 3533 |
| int8_full | 2473 | 3663 |
| int8_incremental | 1345 | 1947 |
| int8_acc_refresh_only | 1323 | 1920 |
| int8_acc_update_only | 198 | 224 |
| int8_layers_2_3_only | 1146 | 1428 |

## Latency per evaluation: `x86_64_O3native`

Hardware:  AMD Ryzen 5 4600H with Radeon Graphics, Linux x86_64, g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 -O3 -march=native, 12 cores

20500 positions x 20 reps, timed in chunks of 64; avg 22.57 active features, avg 3.12 features changed per move.

| path | median ns | p95 ns |
|---|---|---|
| fp32_full | 3830 | 4815 |
| fp32_incremental | 2850 | 3300 |
| int8_full | 1572 | 2022 |
| int8_incremental | 535 | 685 |
| int8_acc_refresh_only | 1002 | 1369 |
| int8_acc_update_only | 70 | 90 |
| int8_layers_2_3_only | 402 | 504 |

## Latency per evaluation: `x86_64_O3native_fastmath`

Hardware:  AMD Ryzen 5 4600H with Radeon Graphics, Linux x86_64, g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 -O3 -march=native -ffast-math, 12 cores

20500 positions x 20 reps, timed in chunks of 64; avg 22.57 active features, avg 3.12 features changed per move.

| path | median ns | p95 ns |
|---|---|---|
| fp32_full | 1115 | 1438 |
| fp32_incremental | 192 | 264 |
| int8_full | 1415 | 1813 |
| int8_incremental | 472 | 533 |
| int8_acc_refresh_only | 1009 | 1314 |
| int8_acc_update_only | 71 | 91 |
| int8_layers_2_3_only | 409 | 516 |

- `random_x86_64`: 18300 steps, incremental-vs-full mismatches 0 (acc) / 0 (eval), C++-vs-Python int mismatches 0, fp32 incremental drift max 0.000854 cp, pass=True
- `special_x86_64`: 42 steps, incremental-vs-full mismatches 0 (acc) / 0 (eval), C++-vs-Python int mismatches 0, fp32 incremental drift max 0.000277 cp, pass=True
