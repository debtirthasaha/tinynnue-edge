"""Train the FP32 TinyNNUE (769 -> 128 -> 32 -> 1) on Stockfish labels.

    python train.py            # reads data/positions.csv, writes checkpoints/ and results/

Loss is MSE in "win-probability space": sigmoid(pred / 400) vs sigmoid(cp / 400).
A +900 vs +1200 disagreement barely matters (both are winning), a +0 vs +300
disagreement matters a lot. This is the standard NNUE training target shape.
"""
import csv
import json
import os
import time

import chess
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

from features import MAX_ACTIVE, N_FEATURES, PAD, active_features

CP_SCALE = 400.0  # net output o is in units of 400 cp; cp = 400 * o
H1, H2 = 128, 32


def load_dataset(path="data/positions.csv"):
    """Return dict split -> (idx [N,33] int64 padded with PAD, cp [N] float32, material [N] float32)."""
    cache = path.replace(".csv", ".npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        z = np.load(cache)
        return {s: (z[f"{s}_idx"], z[f"{s}_cp"], z[f"{s}_mat"]) for s in ("train", "val", "test")}
    rows = {"train": [], "val": [], "test": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            feats = active_features(chess.Board(r["fen"]))
            rows[r["split"]].append((feats, float(r["cp_white"]), float(r["material_white"])))
    out, save = {}, {}
    for s, rs in rows.items():
        idx = np.full((len(rs), MAX_ACTIVE), PAD, dtype=np.int64)
        for i, (feats, _, _) in enumerate(rs):
            idx[i, :len(feats)] = feats
        cp = np.array([r[1] for r in rs], dtype=np.float32)
        mat = np.array([r[2] for r in rs], dtype=np.float32)
        out[s] = (idx, cp, mat)
        save.update({f"{s}_idx": idx, f"{s}_cp": cp, f"{s}_mat": mat})
    np.savez(cache, **save)
    return out


class TinyNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # Sparse first layer: summing rows of a [770,128] table == W1 @ x for binary x.
        # Row PAD (=769) is a fixed zero row so fixed-width index arrays can be padded.
        self.l1 = nn.EmbeddingBag(N_FEATURES + 1, H1, mode="sum", padding_idx=PAD)
        self.b1 = nn.Parameter(torch.zeros(H1))
        self.l2 = nn.Linear(H1, H2)
        self.l3 = nn.Linear(H2, 1)
        nn.init.normal_(self.l1.weight, std=0.1)
        with torch.no_grad():
            self.l1.weight[PAD].zero_()
            self.b1.fill_(0.5)

    def forward(self, idx):
        h1 = torch.clamp(self.l1(idx) + self.b1, 0.0, 1.0)  # clipped ReLU
        h2 = torch.clamp(self.l2(h1), 0.0, 1.0)
        return self.l3(h2).squeeze(-1)  # in units of CP_SCALE


def wdl(cp):
    return torch.sigmoid(torch.as_tensor(cp) / CP_SCALE)


def metrics(pred_cp, true_cp):
    """wdl_mse: the training objective. cp_mae: on |label| <= 1000 only, where centipawns are meaningful."""
    pred_cp, true_cp = np.asarray(pred_cp, np.float64), np.asarray(true_cp, np.float64)
    wdl_mse = float(np.mean((1 / (1 + np.exp(-pred_cp / CP_SCALE)) - 1 / (1 + np.exp(-true_cp / CP_SCALE))) ** 2))
    m = np.abs(true_cp) <= 1000
    return {"wdl_mse": wdl_mse,
            "cp_mae": float(np.mean(np.abs(pred_cp[m] - true_cp[m]))),
            "n": int(len(true_cp))}


@torch.no_grad()
def predict_cp(model, idx, bs=8192):
    model.eval()
    out = [model(torch.from_numpy(idx[i:i + bs])) for i in range(0, len(idx), bs)]
    return (torch.cat(out) * CP_SCALE).numpy()


def main(data_path="data/positions.csv", epochs=40, bs=1024, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_dataset(data_path)
    for s, (idx, _, _) in data.items():
        print(f"{s}: {len(idx)} positions")
    tr_idx, tr_cp, _ = data["train"]
    tr_idx, tr_target = torch.from_numpy(tr_idx), wdl(tr_cp)

    model = TinyNNUE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, history = float("inf"), []
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = ((torch.sigmoid(model(tr_idx[b])) - tr_target[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        tr_m = metrics(predict_cp(model, data["train"][0]), data["train"][1])
        va_m = metrics(predict_cp(model, data["val"][0]), data["val"][1])
        history.append({"epoch": ep + 1, "train_wdl_mse": tr_m["wdl_mse"], "val_wdl_mse": va_m["wdl_mse"],
                        "val_cp_mae": va_m["cp_mae"]})
        print(f"ep {ep + 1:2d}  train {tr_m['wdl_mse']:.5f}  val {va_m['wdl_mse']:.5f}  "
              f"val_cp_mae {va_m['cp_mae']:.1f}  {time.time() - t0:.0f}s", flush=True)
        if va_m["wdl_mse"] < best_val:
            best_val = va_m["wdl_mse"]
            torch.save(model.state_dict(), "checkpoints/tinynnue_fp32.pt")

    model.load_state_dict(torch.load("checkpoints/tinynnue_fp32.pt"))
    sd = model.state_dict()
    save_file({"W1": sd["l1.weight"][:N_FEATURES].contiguous(),  # [769,128] feature-major
               "b1": sd["b1"], "W2": sd["l2.weight"], "b2": sd["l2.bias"],
               "W3": sd["l3.weight"].reshape(-1).contiguous(), "b3": sd["l3.bias"]},
              "checkpoints/model.safetensors")

    te_idx, te_cp, te_mat = data["test"]
    result = {
        "test_net_fp32": metrics(predict_cp(model, te_idx), te_cp),
        "test_material": metrics(te_mat, te_cp),
        "test_zero": metrics(np.zeros_like(te_cp), te_cp),
        "best_val_wdl_mse": best_val,
        "history": history,
        "hparams": {"epochs": epochs, "batch_size": bs, "lr": lr, "seed": seed, "cp_scale": CP_SCALE},
        "params": sum(p.numel() for p in model.parameters()) - H1,  # minus the PAD row
    }
    with open("results/train_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    for k in ("test_net_fp32", "test_material", "test_zero"):
        print(k, result[k])


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
