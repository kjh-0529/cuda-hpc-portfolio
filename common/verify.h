// Host-side helpers: random initialization and result comparison.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

inline void fill_random(std::vector<float>& v, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// Compares `got` against `ref` with a mixed absolute/relative tolerance.
// Prints the first mismatch and returns false if any element is out of tolerance.
inline bool all_close(const std::vector<float>& got, const std::vector<float>& ref,
                      float atol = 1e-3f, float rtol = 1e-3f) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "size mismatch: %zu vs %zu\n", got.size(), ref.size());
        return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - ref[i]);
        if (!(diff <= atol + rtol * std::fabs(ref[i]))) {  // also catches NaN
            std::fprintf(stderr, "mismatch at %zu: got %f, expected %f\n", i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}
