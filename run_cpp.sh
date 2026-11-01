#!/bin/sh
# Build the C++ core, run the correctness tests, then the benchmark.
# Linux or WSL. Override compiler/flags with CXX=... CXXFLAGS=... and the results tag with TAG=...
set -e
cd "$(dirname "$0")"
make -s
mkdir -p results
TAG=${TAG:-$(uname -m)}
CPU=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
COMP=$(${CXX:-g++} --version | head -1)
printf '{"description": "%s, %s, %s %s, %s cores"}\n' "$CPU" "$(uname -sm)" "$COMP" "${CXXFLAGS:--O2}" "$(nproc)" > results/hardware_$TAG.json
for t in special random; do
    echo "== test $t"
    ./build/nnue test export/tinynnue_int8.bin export/tinynnue_fp32.bin traces/$t.txt | tee results/test_${t}_$TAG.json
done
echo "== bench"
./build/nnue bench export/tinynnue_int8.bin export/tinynnue_fp32.bin traces/bench.txt ${REPS:-20} | tee results/bench_$TAG.json
