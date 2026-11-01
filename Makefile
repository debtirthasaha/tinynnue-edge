# Plain scalar C++; -O2 lets GCC auto-vectorize simple loops, no hand-written SIMD.
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

build/nnue: nnue.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf build
