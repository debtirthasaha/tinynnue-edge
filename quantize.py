"""Post-training quantization of TinyNNUE + the bit-exact Python integer reference.

    python quantize.py         # reads checkpoints/model.safetensors, writes export/ and results/

Integer pipeline (per-tensor symmetric, A = 127 activation steps):

  layer 1   acc1 = b1q + sum_{f active} W1q[f]            int8 rows summed into int32
            a1   = clamp(requant(acc1, M1), 0, 127)       a1 / 127 ~ clippedReLU(W1 x + b1)
  layer 2   acc2 = b2q + W2q @ a1                         int8 * int8 widened to int32
            a2   = clamp(requant(acc2, M2), 0, 127)
  layer 3   acc3 = b3q + W3q . a2
            cp   = requant(acc3, M3)                      final output in integer centipawns

  requant(x, M) = (x * M + 2^(S-1)) >> S                  fixed-point multiply by M / 2^S

Scales:
  s_k = max|W_k| / 127                     weight scale (per tensor)
  layer-1 input is binary, so its scale is 1 and acc1 has scale s_1
  layer-2/3 inputs are a / 127, so acc_k has scale s_k / 127
  bias is quantized with the ACCUMULATOR's scale so it can be added directly
  M1 = s_1 * 127, M2 = s_2, M3 = 400 * s_3 / 127   (all times 2^S, rounded)

None of these scales depend on the position. That is what makes incremental
updates exact: acc1 is a plain integer sum, so add/subtract in any order gives
bit-identical results to recomputing it from scratch.
"""
import json
import os
import struct

import numpy as np
from safetensors.numpy import load_file

from features import N_FEATURES, PAD
from train import CP_SCALE, load_dataset, metrics

A = 127      # activation levels: clipped ReLU output [0, 1] -> integers [0, 127]
S = 24       # fixed-point shift: M = round(m * 2^24) keeps even the smallest multiplier (~1e-3) to ~1e-4 relative error
MAGIC = b"TNUE"


# ---------------------------------------------------------------- quantization
def quantize_weights(w, axis=None):
    """Symmetric int8. axis=None: one scale per tensor. axis=k: one scale per slice along k."""
    amax = np.abs(w).max() if axis is None else np.abs(w).max(axis=axis, keepdims=True)
    scale = amax / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def fixed_point(m):
    """Real multiplier m -> integer M with m ~ M / 2^S."""
    M = np.round(np.asarray(m, np.float64) * (1 << S)).astype(np.int64)
    assert np.all(M > 0) and np.all(M < 2 ** 31), "multiplier does not fit int32"
    return M


def quantize_net(p, per_channel=False):
    """p: float params dict. Returns integer params. per_channel=True is the analysis-only variant
    (one scale per output unit); the exported/C++ model is per-tensor."""
    W1q, s1 = quantize_weights(p["W1"], axis=0 if per_channel else None)       # [769,128], s1 [1,128] or scalar
    W2q, s2 = quantize_weights(p["W2"], axis=1 if per_channel else None)       # [32,128],  s2 [32,1]  or scalar
    W3q, s3 = quantize_weights(p["W3"])                                        # [32]
    s1, s2 = np.reshape(s1, -1), np.reshape(s2, -1)
    return {
        "W1q": W1q, "b1q": np.round(p["b1"] / s1).astype(np.int32),
        "W2q": W2q, "b2q": np.round(p["b2"] * A / s2).astype(np.int32),
        "W3q": W3q, "b3q": np.round(p["b3"] * A / s3).astype(np.int32),
        "M1": fixed_point(s1 * A) * np.ones(p["W1"].shape[1], np.int64),
        "M2": fixed_point(s2) * np.ones(p["W2"].shape[0], np.int64),
        "M3": fixed_point(CP_SCALE * s3 / A),
        "s": (s1, s2, s3),
    }


# ------------------------------------------------------------- forward passes
def fp32_forward(p, idx):
    """Float reference. idx [N,33] padded with PAD. Returns cp (float)."""
    W1 = np.vstack([p["W1"], np.zeros((1, p["W1"].shape[1]), np.float32)])  # PAD row
    h1 = np.clip(W1[idx].sum(1) + p["b1"], 0, 1)
    h2 = np.clip(h1 @ p["W2"].T + p["b2"], 0, 1)
    return (h2 @ p["W3"] + p["b3"]) * CP_SCALE


def requant(x, M):
    return (x.astype(np.int64) * M + (1 << (S - 1))) >> S  # numpy >> on int64 is arithmetic, like C++


def int_forward(q, idx, return_acc1=False):
    """Integer reference that the C++ core must match bit for bit."""
    W1q = np.vstack([q["W1q"], np.zeros((1, q["W1q"].shape[1]), np.int8)]).astype(np.int64)
    acc1 = q["b1q"].astype(np.int64) + W1q[idx].sum(1)
    a1 = np.clip(requant(acc1, q["M1"]), 0, A)
    acc2 = q["b2q"].astype(np.int64) + a1 @ q["W2q"].T.astype(np.int64)
    a2 = np.clip(requant(acc2, q["M2"]), 0, A)
    acc3 = q["b3q"].astype(np.int64) + a2 @ q["W3q"].astype(np.int64)
    cp = requant(acc3, q["M3"]).reshape(-1)
    return (cp, acc1, acc2, acc3) if return_acc1 else cp


def fake_quant_forward(p, idx, quant_weights, quant_acts):
    """Float forward with only some quantization steps simulated, to split the error by source."""
    if quant_weights:
        p = dict(p)
        for k in ("W1", "W2", "W3"):
            q, s = quantize_weights(p[k])
            p[k] = q.astype(np.float32) * s
    W1 = np.vstack([p["W1"], np.zeros((1, p["W1"].shape[1]), np.float32)])
    act = (lambda h: np.round(h * A) / A) if quant_acts else (lambda h: h)
    h1 = act(np.clip(W1[idx].sum(1) + p["b1"], 0, 1))
    h2 = act(np.clip(h1 @ p["W2"].T + p["b2"], 0, 1))
    return (h2 @ p["W3"] + p["b3"]) * CP_SCALE


def err_stats(a, b):
    d = np.abs(np.asarray(a, np.float64) - b)
    return {"mae": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": float(d.max())}


# --------------------------------------------------------------------- export
def export_int8(q, path):
    """Little-endian layout read by nnue.cpp (see load_int8 there)."""
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<6i", 1, N_FEATURES, 128, 32, S, 0))
        f.write(struct.pack("<3i", int(q["M1"][0]), int(q["M2"][0]), int(q["M3"])))
        f.write(q["W1q"].tobytes())                         # [769][128] int8, one row per feature
        f.write(q["b1q"].astype("<i4").tobytes())
        f.write(q["W2q"].tobytes())                         # [32][128] int8
        f.write(q["b2q"].astype("<i4").tobytes())
        f.write(q["W3q"].tobytes())                         # [32] int8
        f.write(q["b3q"].astype("<i4").tobytes())


def load_int8(path):
    """Read the exported file back; the Python reference runs on exactly these bytes."""
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC
        _, nf, h1, h2, s, _ = struct.unpack("<6i", f.read(24))
        assert s == S
        M1, M2, M3 = struct.unpack("<3i", f.read(12))
        rd = lambda n, dt: np.frombuffer(f.read(n * np.dtype(dt).itemsize), dtype=dt)
        return {"W1q": rd(nf * h1, np.int8).reshape(nf, h1), "b1q": rd(h1, "<i4"),
                "W2q": rd(h2 * h1, np.int8).reshape(h2, h1), "b2q": rd(h2, "<i4"),
                "W3q": rd(h2, np.int8), "b3q": rd(1, "<i4"),
                "M1": np.int64(M1), "M2": np.int64(M2), "M3": np.int64(M3)}


def export_fp32(p, path):
    with open(path, "wb") as f:
        f.write(b"TNUF" + struct.pack("<3i", N_FEATURES, 128, 32))
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            f.write(p[k].astype("<f4").tobytes())


# ------------------------------------------------------------ overflow bounds
def overflow_bounds(q):
    """Worst-case |accumulator| over ANY input, not just the data we saw."""
    W1 = np.abs(q["W1q"].astype(np.int64))
    top32 = np.sort(W1[:N_FEATURES - 1], axis=0)[-32:].sum(0)        # 32 largest piece rows per unit
    acc1 = np.abs(q["b1q"]).astype(np.int64) + top32 + W1[N_FEATURES - 1]  # + side-to-move row
    acc2 = np.abs(q["b2q"]).astype(np.int64) + np.abs(q["W2q"].astype(np.int64)).sum(1) * A
    acc3 = int(np.abs(q["b3q"]).astype(np.int64).sum() + np.abs(q["W3q"].astype(np.int64)).sum() * A)
    loose1 = int(np.abs(q["b1q"]).max()) + 33 * 127
    return {"acc1_worst": int(acc1.max()), "acc1_loose_33x127_plus_bias": loose1,
            "acc2_worst": int(acc2.max()), "acc2_loose_128x127x127_plus_bias": int(np.abs(q["b2q"]).max()) + 128 * 127 * 127,
            "acc3_worst": acc3, "int16_max": 32767, "int32_max": 2 ** 31 - 1}


def main():
    p = load_file("checkpoints/model.safetensors")
    data = load_dataset()
    te_idx, te_cp, _ = data["test"]
    os.makedirs("export", exist_ok=True)

    q = quantize_net(p)
    export_int8(q, "export/tinynnue_int8.bin")
    export_fp32(p, "export/tinynnue_fp32.bin")
    qf = load_int8("export/tinynnue_int8.bin")

    fp = fp32_forward(p, te_idx)
    it, acc1, acc2, acc3 = int_forward(qf, te_idx, return_acc1=True)
    pc = int_forward(quantize_net(p, per_channel=True), te_idx)

    # multipliers: how far is M / 2^S from the real multiplier it stands for
    s1, s2, s3 = q["s"]
    mult_err = {k: float(abs(M / 2 ** S - m) / m) for k, M, m in
                (("M1", qf["M1"], s1[0] * A), ("M2", qf["M2"], s2[0]), ("M3", qf["M3"], CP_SCALE * s3 / A))}
    w_err = {k: float(np.abs(p[k] - q[k + "q"].astype(np.float32) * sc).max())
             for k, sc in (("W1", s1[0]), ("W2", s2[0]), ("W3", s3))}

    res = {
        "scales": {"s1": float(s1[0]), "s2": float(s2[0]), "s3": float(s3), "S": S, "A": A,
                   "M1": int(qf["M1"]), "M2": int(qf["M2"]), "M3": int(qf["M3"])},
        "multiplier_rel_error": mult_err,
        "weight_max_abs_error": w_err,
        "sizes_bytes": {"fp32_bin": os.path.getsize("export/tinynnue_fp32.bin"),
                        "int8_bin": os.path.getsize("export/tinynnue_int8.bin")},
        "overflow_bounds": overflow_bounds(qf),
        "observed_max_abs": {"acc1": int(np.abs(acc1).max()), "acc2": int(np.abs(acc2).max()),
                             "acc3": int(np.abs(acc3).max())},
        "int8_vs_fp32_cp": err_stats(it, fp),
        "per_channel_vs_fp32_cp": err_stats(pc, fp),
        # which rounding step causes the error: weights only, activations only, both (float-simulated)
        "error_breakdown_vs_fp32_cp": {
            "weights_only": err_stats(fake_quant_forward(p, te_idx, True, False), fp),
            "activations_only": err_stats(fake_quant_forward(p, te_idx, False, True), fp),
            "both_simulated": err_stats(fake_quant_forward(p, te_idx, True, True), fp),
            **{f"{k}_only": err_stats(fp32_forward({**p, k: quantize_weights(p[k])[0].astype(np.float32)
                                                   * quantize_weights(p[k])[1]}, te_idx), fp)
               for k in ("W1", "W2", "W3")},
        },
        "weight_abs_stats": {k: {"max": float(np.abs(p[k]).max()), "p99": float(np.percentile(np.abs(p[k]), 99)),
                                 "mean": float(np.abs(p[k]).mean())} for k in ("W1", "W2", "W3")},
        "test_vs_stockfish": {"fp32": metrics(fp, te_cp), "int8_per_tensor": metrics(it, te_cp),
                              "int8_per_channel": metrics(pc, te_cp)},
    }
    with open("results/quant_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
