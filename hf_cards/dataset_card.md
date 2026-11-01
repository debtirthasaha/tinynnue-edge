---
license: cc0-1.0
tags:
- chess
- stockfish
- nnue
size_categories:
- 100K<n<1M
---

# TinyNNUE Stockfish labels

121,187 chess positions from the Lichess January 2013 rated-games database, each labeled with a Stockfish 19 evaluation. Used to train [MR0b0t/tinynnue-edge](https://huggingface.co/MR0b0t/tinynnue-edge).

## Columns (`positions.csv`)

| column | meaning |
|---|---|
| `game_id` | Lichess game id (`https://lichess.org/<game_id>`) |
| `split` | `train` / `val` / `test`, assigned per game by SHA-1 of the game id (90/5/5) |
| `ply` | half-move number in the game |
| `fen` | position |
| `cp_white` | Stockfish score in centipawns, White's point of view |
| `material_white` | material count P=100 N=320 B=330 R=500 Q=900, White's point of view |
| `bestmove` | Stockfish best move (UCI) |

Splits: 108,960 train / 6,155 val / 6,072 test. No game appears in more than one split.

## How it was made

- Up to 3 positions per game, sampled from ply ≥ 10 of games with ≥ 20 plies; positions in check skipped.
- Stockfish 19 (`stockfish-windows-x86-64-universal`), `Threads=1`, `Hash=16`, hash cleared before every position, fixed budget of 10,000 nodes. Deterministic for a given binary.
- 170,559 candidates searched, 49,372 dropped: mate scores (no finite centipawn value) and positions whose best move is a capture or promotion, since a static evaluator cannot see the tactic the search found. At the 170,000-candidate log mark the split was 5,132 mate / 44,055 tactical.
- Script: `data.py --target 170559` in [github.com/debtirthasaha/tinynnue-edge](https://github.com/debtirthasaha/tinynnue-edge).

## Limitations

10k-node scores are noisy, especially in sharp positions. 2013 Lichess games skew toward lower-rated play. The filters make this a quiet-position dataset.

## License and attribution

Games: [Lichess open database](https://database.lichess.org/), CC0. Evaluations produced with [Stockfish](https://stockfishchess.org/) 19 (GPLv3), used as a tool; no Stockfish code or network is included.
