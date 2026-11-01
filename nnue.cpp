// TinyNNUE inference core: FP32 and integer (int8 weights, int32 accumulators),
// full recomputation and incremental first-layer updates.
//
//   nnue test  <int8.bin> <fp32.bin> <trace>        correctness: incremental == full == Python
//   nnue bench <int8.bin> <fp32.bin> <trace> [reps] latency, prints one JSON object
//   nnue eval  <int8.bin> <fp32.bin> <trace>        per-step evals (used by demo.py)
//
// Chess rules live in Python. This file only sees feature indices:
//   B n f1..fn                          begin: refresh the accumulator from a feature list
//   S nr r1.. na a1.. nf f1.. int fp    step: remove/add features (incremental), the full
//                                       feature list after the move, and the Python
//                                       reference outputs (int cp, fp32 cp)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

constexpr int NF = 769, H1 = 128, H2 = 32, A = 127;

// ------------------------------------------------------------------ networks
struct Int8Net {
    int S;
    int32_t M1, M2, M3;           // fixed-point requantization multipliers (value = M / 2^S)
    std::vector<int8_t> W1;       // [NF][H1], one contiguous row per feature
    int32_t b1[H1];
    int8_t W2[H2][H1];
    int32_t b2[H2];
    int8_t W3[H2];
    int32_t b3;
};

struct FloatNet {
    std::vector<float> W1;        // [NF][H1]
    float b1[H1], W2[H2][H1], b2[H2], W3[H2], b3;
};

static void read_exact(FILE* f, void* dst, size_t n, const char* what) {
    if (fread(dst, 1, n, f) != n) { fprintf(stderr, "short read: %s\n", what); exit(1); }
}

static Int8Net load_int8(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[4];
    int32_t hdr[6];
    read_exact(f, magic, 4, "magic");
    read_exact(f, hdr, sizeof hdr, "header");
    if (memcmp(magic, "TNUE", 4) || hdr[1] != NF || hdr[2] != H1 || hdr[3] != H2) {
        fprintf(stderr, "bad int8 file\n"); exit(1);
    }
    Int8Net n;
    n.S = hdr[4];
    read_exact(f, &n.M1, 4, "M1"); read_exact(f, &n.M2, 4, "M2"); read_exact(f, &n.M3, 4, "M3");
    n.W1.resize(NF * H1);
    read_exact(f, n.W1.data(), NF * H1, "W1");
    read_exact(f, n.b1, sizeof n.b1, "b1");
    read_exact(f, n.W2, sizeof n.W2, "W2");
    read_exact(f, n.b2, sizeof n.b2, "b2");
    read_exact(f, n.W3, sizeof n.W3, "W3");
    read_exact(f, &n.b3, 4, "b3");
    fclose(f);
    return n;
}

static FloatNet load_fp32(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[4];
    int32_t hdr[3];
    read_exact(f, magic, 4, "magic");
    read_exact(f, hdr, sizeof hdr, "header");
    if (memcmp(magic, "TNUF", 4) || hdr[0] != NF || hdr[1] != H1 || hdr[2] != H2) {
        fprintf(stderr, "bad fp32 file\n"); exit(1);
    }
    FloatNet n;
    n.W1.resize(NF * H1);
    read_exact(f, n.W1.data(), NF * H1 * 4, "W1");
    read_exact(f, n.b1, sizeof n.b1, "b1");
    read_exact(f, n.W2, sizeof n.W2, "W2");
    read_exact(f, n.b2, sizeof n.b2, "b2");
    read_exact(f, n.W3, sizeof n.W3, "W3");
    read_exact(f, &n.b3, 4, "b3");
    fclose(f);
    return n;
}

// ------------------------------------------------------------ integer path
// (x * M + 2^(S-1)) >> S : multiply by the real number M / 2^S, round half up.
// x * M can exceed int32 (e.g. 2e6 * 2e7), so the product is done in int64.
static inline int32_t requant(int32_t x, int32_t M, int S) {
    return (int32_t)(((int64_t)x * M + (int64_t(1) << (S - 1))) >> S);
}
static inline int32_t clamp_act(int32_t x) { return std::min(std::max(x, 0), A); }

struct IntAcc { int32_t v[H1]; };

static void refresh(const Int8Net& n, const std::vector<int>& feats, IntAcc& acc) {
    std::copy(n.b1, n.b1 + H1, acc.v);
    for (int f : feats) {
        const int8_t* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] += row[i];   // int8 widened to int32
    }
}

static void update(const Int8Net& n, const std::vector<int>& removed, const std::vector<int>& added, IntAcc& acc) {
    for (int f : removed) {
        const int8_t* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] -= row[i];
    }
    for (int f : added) {
        const int8_t* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] += row[i];
    }
}

static int32_t evaluate(const Int8Net& n, const IntAcc& acc) {
    int32_t a1[H1], a2[H2];
    for (int i = 0; i < H1; i++) a1[i] = clamp_act(requant(acc.v[i], n.M1, n.S));
    for (int o = 0; o < H2; o++) {
        int32_t s = n.b2[o];
        for (int i = 0; i < H1; i++) s += (int32_t)n.W2[o][i] * a1[i];  // <= 128*127*127, fits int32
        a2[o] = clamp_act(requant(s, n.M2, n.S));
    }
    int32_t s = n.b3;
    for (int o = 0; o < H2; o++) s += (int32_t)n.W3[o] * a2[o];
    return requant(s, n.M3, n.S);   // integer centipawns
}

// -------------------------------------------------------------- float path
struct FloatAcc { float v[H1]; };

static void refresh(const FloatNet& n, const std::vector<int>& feats, FloatAcc& acc) {
    std::copy(n.b1, n.b1 + H1, acc.v);
    for (int f : feats) {
        const float* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] += row[i];
    }
}

static void update(const FloatNet& n, const std::vector<int>& removed, const std::vector<int>& added, FloatAcc& acc) {
    for (int f : removed) {
        const float* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] -= row[i];
    }
    for (int f : added) {
        const float* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] += row[i];
    }
}

static float evaluate(const FloatNet& n, const FloatAcc& acc) {
    float h1[H1], h2[H2];
    for (int i = 0; i < H1; i++) h1[i] = std::min(std::max(acc.v[i], 0.f), 1.f);
    for (int o = 0; o < H2; o++) {
        float s = n.b2[o];
        for (int i = 0; i < H1; i++) s += n.W2[o][i] * h1[i];
        h2[o] = std::min(std::max(s, 0.f), 1.f);
    }
    float s = n.b3;
    for (int o = 0; o < H2; o++) s += n.W3[o] * h2[o];
    return 400.f * s;
}

// ------------------------------------------------------------------- trace
struct Step {
    bool begin;
    std::vector<int> removed, added, full;
    int32_t ref_int;
    float ref_fp;
};

static std::vector<int> read_list(std::istringstream& in) {
    int n; in >> n;
    std::vector<int> v(n);
    for (int& x : v) in >> x;
    return v;
}

static std::vector<Step> load_trace(const char* path) {
    std::ifstream f(path);
    if (!f) { perror(path); exit(1); }
    std::vector<Step> steps;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        char tag; in >> tag;
        Step s{};
        if (tag == 'B') {
            s.begin = true;
            s.full = read_list(in);
        } else {
            s.removed = read_list(in);
            s.added = read_list(in);
            s.full = read_list(in);
        }
        in >> s.ref_int >> s.ref_fp;
        steps.push_back(std::move(s));
    }
    return steps;
}

// -------------------------------------------------------------------- test
static int cmd_test(const Int8Net& qn, const FloatNet& fn, const std::vector<Step>& steps) {
    IntAcc inc, full;
    FloatAcc finc, ffull;
    long n = 0, acc_mismatch = 0, eval_mismatch = 0, ref_mismatch = 0, fp_ref_bad = 0;
    int32_t max_acc1 = 0;
    double max_fp_drift = 0, max_fp_ref_diff = 0;
    for (const Step& s : steps) {
        refresh(qn, s.full, full);
        refresh(fn, s.full, ffull);
        if (s.begin) { inc = full; finc = ffull; }
        else { update(qn, s.removed, s.added, inc); update(fn, s.removed, s.added, finc); }
        n++;
        if (memcmp(inc.v, full.v, sizeof inc.v) != 0) acc_mismatch++;
        int32_t e_inc = evaluate(qn, inc), e_full = evaluate(qn, full);
        if (e_inc != e_full) eval_mismatch++;
        if (e_full != s.ref_int) ref_mismatch++;
        for (int i = 0; i < H1; i++) max_acc1 = std::max(max_acc1, std::abs(full.v[i]));
        float f_full = evaluate(fn, ffull), f_inc = evaluate(fn, finc);
        max_fp_drift = std::max(max_fp_drift, (double)std::fabs(f_inc - f_full));
        double d = std::fabs(f_full - s.ref_fp);
        max_fp_ref_diff = std::max(max_fp_ref_diff, d);
        if (d > 0.05) fp_ref_bad++;
    }
    bool ok = acc_mismatch == 0 && eval_mismatch == 0 && ref_mismatch == 0 && fp_ref_bad == 0;
    printf("{\"steps\": %ld, \"int_acc_incremental_vs_full_mismatches\": %ld, "
           "\"int_eval_incremental_vs_full_mismatches\": %ld, \"int_eval_vs_python_mismatches\": %ld, "
           "\"fp32_vs_python_over_0.05cp\": %ld, \"fp32_vs_python_max_abs_cp\": %.6f, "
           "\"fp32_incremental_drift_max_abs_cp\": %.6f, \"max_abs_acc1\": %d, \"pass\": %s}\n",
           n, acc_mismatch, eval_mismatch, ref_mismatch, fp_ref_bad, max_fp_ref_diff, max_fp_drift,
           max_acc1, ok ? "true" : "false");
    return ok ? 0 : 1;
}

// ------------------------------------------------------------------- bench
// One eval takes well under a microsecond, close to the clock's own overhead, so
// we time CHUNK consecutive steps and divide. median/p95 are over chunks.
constexpr int CHUNK = 64;

template <class F>
static void time_chunks(const std::vector<Step>& steps, int reps, F&& run_step, std::vector<double>& ns_per_eval) {
    using clk = std::chrono::steady_clock;
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i + CHUNK <= steps.size(); i += CHUNK) {
            auto t0 = clk::now();
            for (size_t j = i; j < i + CHUNK; j++) run_step(steps[j]);
            auto t1 = clk::now();
            ns_per_eval.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / CHUNK);
        }
    }
}

static void stats(const char* name, std::vector<double> v, bool last) {
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
    printf("    \"%s\": {\"median_ns\": %.1f, \"p95_ns\": %.1f, \"chunks\": %zu}%s\n",
           name, pct(0.5), pct(0.95), v.size(), last ? "" : ",");
}

static int cmd_bench(const Int8Net& qn, const FloatNet& fn, const std::vector<Step>& steps, int reps) {
    volatile int64_t sink = 0;   // keeps the compiler from deleting the work
    IntAcc acc;
    FloatAcc facc;
    std::vector<double> t_fp_full, t_fp_inc, t_q_full, t_q_inc, t_q_refresh, t_q_update, t_q_eval;

    // warm-up pass
    for (const Step& s : steps) { refresh(qn, s.full, acc); sink += evaluate(qn, acc); }

    time_chunks(steps, reps, [&](const Step& s) { refresh(fn, s.full, facc); sink += (int64_t)evaluate(fn, facc); }, t_fp_full);
    time_chunks(steps, reps, [&](const Step& s) {
        if (s.begin) refresh(fn, s.full, facc); else update(fn, s.removed, s.added, facc);
        sink += (int64_t)evaluate(fn, facc); }, t_fp_inc);
    time_chunks(steps, reps, [&](const Step& s) { refresh(qn, s.full, acc); sink += evaluate(qn, acc); }, t_q_full);
    time_chunks(steps, reps, [&](const Step& s) {
        if (s.begin) refresh(qn, s.full, acc); else update(qn, s.removed, s.added, acc);
        sink += evaluate(qn, acc); }, t_q_inc);
    // breakdown of the int8 path
    time_chunks(steps, reps, [&](const Step& s) { refresh(qn, s.full, acc); sink += acc.v[0]; }, t_q_refresh);
    time_chunks(steps, reps, [&](const Step& s) {
        if (s.begin) refresh(qn, s.full, acc); else update(qn, s.removed, s.added, acc);
        sink += acc.v[0]; }, t_q_update);
    refresh(qn, steps[0].full, acc);
    time_chunks(steps, reps, [&](const Step&) { sink += evaluate(qn, acc); }, t_q_eval);

    long removed = 0, added = 0, full = 0, upd = 0;
    for (const Step& s : steps) {
        full += s.full.size();
        if (!s.begin) { removed += s.removed.size(); added += s.added.size(); upd++; }
    }
    printf("{\n  \"steps\": %zu, \"reps\": %d, \"chunk\": %d,\n", steps.size(), reps, CHUNK);
    printf("  \"avg_active_features\": %.2f, \"avg_features_changed_per_move\": %.2f,\n",
           (double)full / steps.size(), (double)(removed + added) / std::max(upd, 1L));
    printf("  \"latency_per_eval\": {\n");
    stats("fp32_full", t_fp_full, false);
    stats("fp32_incremental", t_fp_inc, false);
    stats("int8_full", t_q_full, false);
    stats("int8_incremental", t_q_inc, false);
    stats("int8_acc_refresh_only", t_q_refresh, false);
    stats("int8_acc_update_only", t_q_update, false);
    stats("int8_layers_2_3_only", t_q_eval, true);
    printf("  },\n  \"sink\": %lld\n}\n", (long long)sink);
    return 0;
}

// -------------------------------------------------------------------- eval
static int cmd_eval(const Int8Net& qn, const FloatNet& fn, const std::vector<Step>& steps) {
    IntAcc acc;
    FloatAcc full;
    for (const Step& s : steps) {
        if (s.begin) refresh(qn, s.full, acc); else update(qn, s.removed, s.added, acc);
        refresh(fn, s.full, full);
        printf("%d %.2f\n", evaluate(qn, acc), evaluate(fn, full));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: nnue {test|bench|eval} <int8.bin> <fp32.bin> <trace> [reps]\n");
        return 2;
    }
    Int8Net qn = load_int8(argv[2]);
    FloatNet fn = load_fp32(argv[3]);
    std::vector<Step> steps = load_trace(argv[4]);
    std::string cmd = argv[1];
    if (cmd == "test") return cmd_test(qn, fn, steps);
    if (cmd == "bench") return cmd_bench(qn, fn, steps, argc > 5 ? atoi(argv[5]) : 20);
    if (cmd == "eval") return cmd_eval(qn, fn, steps);
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 2;
}
