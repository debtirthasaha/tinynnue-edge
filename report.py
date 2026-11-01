"""Collect results/*.json into results/RESULTS.md (the tables used by the README and model card).

    python report.py
"""
import glob
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    tr = load("results/train_metrics.json")
    qm = load("results/quant_metrics.json")
    out = ["# Results", ""]

    out += ["## Accuracy vs Stockfish (test split, positions from held-out games)", "",
            "| model | WDL-space MSE | cp MAE (\\|label\\| <= 1000) |", "|---|---|---|"]
    rows = [("always 0", tr["test_zero"]), ("material count", tr["test_material"]),
            ("TinyNNUE fp32", qm["test_vs_stockfish"]["fp32"]),
            ("TinyNNUE int8 per-tensor (shipped)", qm["test_vs_stockfish"]["int8_per_tensor"]),
            ("TinyNNUE int8 per-channel (analysis only)", qm["test_vs_stockfish"]["int8_per_channel"])]
    out += [f"| {n} | {m['wdl_mse']:.5f} | {m['cp_mae']:.1f} |" for n, m in rows]
    out += ["", f"n = {tr['test_net_fp32']['n']} test positions.", ""]

    e, pc = qm["int8_vs_fp32_cp"], qm["per_channel_vs_fp32_cp"]
    out += ["## Quantization error (int8 output vs fp32 output, centipawns)", "",
            "| scheme | mean abs | p99 abs | max abs |", "|---|---|---|---|",
            f"| per-tensor | {e['mae']:.2f} | {e['p99']:.2f} | {e['max']:.2f} |",
            f"| per-channel | {pc['mae']:.2f} | {pc['p99']:.2f} | {pc['max']:.2f} |", "",
            "Error by source (float simulation, only the named rounding step applied):", "",
            "| quantized | mean abs cp | max abs cp |", "|---|---|---|"]
    out += [f"| {k} | {v['mae']:.2f} | {v['max']:.2f} |" for k, v in qm["error_breakdown_vs_fp32_cp"].items()]
    out += ["", "| weights | max abs | p99 abs | mean abs |", "|---|---|---|---|"]
    out += [f"| {k} | {v['max']:.3f} | {v['p99']:.3f} | {v['mean']:.3f} |" for k, v in qm["weight_abs_stats"].items()]
    out.append("")

    s = qm["sizes_bytes"]
    out += ["## Size", "", "| file | bytes |", "|---|---|",
            f"| fp32 weights | {s['fp32_bin']:,} |", f"| int8 weights + int32 biases | {s['int8_bin']:,} |", ""]

    ob, mx = qm["overflow_bounds"], qm["observed_max_abs"]
    out += ["## Accumulator bounds", "", "| accumulator | worst case (any input) | observed max (test set) |",
            "|---|---|---|",
            f"| acc1 (layer 1) | {ob['acc1_worst']:,} (loose: {ob['acc1_loose_33x127_plus_bias']:,}) | {mx['acc1']:,} |",
            f"| acc2 (layer 2) | {ob['acc2_worst']:,} (loose: {ob['acc2_loose_128x127x127_plus_bias']:,}) | {mx['acc2']:,} |",
            f"| acc3 (layer 3) | {ob['acc3_worst']:,} | {mx['acc3']:,} |", ""]

    for path in sorted(glob.glob("results/bench_*.json")):
        tag = os.path.basename(path)[6:-5]
        if tag == "smoke":
            continue
        b = load(path)
        hw = load(f"results/hardware_{tag}.json") if os.path.exists(f"results/hardware_{tag}.json") else {}
        out += [f"## Latency per evaluation: `{tag}`", "", f"Hardware: {hw.get('description', 'see README')}", "",
                f"{b['steps']} positions x {b['reps']} reps, timed in chunks of {b['chunk']}; "
                f"avg {b['avg_active_features']} active features, "
                f"avg {b['avg_features_changed_per_move']} features changed per move.", "",
                "| path | median ns | p95 ns |", "|---|---|---|"]
        out += [f"| {k} | {v['median_ns']:.0f} | {v['p95_ns']:.0f} |" for k, v in b["latency_per_eval"].items()]
        out.append("")

    for path in sorted(glob.glob("results/test_*.json")):
        tag = os.path.basename(path)[5:-5]
        if tag.endswith("smoke"):
            continue
        t = load(path)
        out.append(f"- `{tag}`: {t['steps']} steps, incremental-vs-full mismatches "
                   f"{t['int_acc_incremental_vs_full_mismatches']} (acc) / {t['int_eval_incremental_vs_full_mismatches']} (eval), "
                   f"C++-vs-Python int mismatches {t['int_eval_vs_python_mismatches']}, "
                   f"fp32 incremental drift max {t['fp32_incremental_drift_max_abs_cp']:.6f} cp, pass={t['pass']}")
    with open("results/RESULTS.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
