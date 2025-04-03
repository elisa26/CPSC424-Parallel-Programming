#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

// Scalar implementation: uses a simple loop
void scalar_add(const float* A, const float* B, float* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Vectorized implementation: uses AVX intrinsics to add 8 floats at a time
void vectorized_add(const float* A, const float* B, float* C, size_t N) {
    size_t i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&C[i], c);
    }
    for (; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const size_t N = 10000000;
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0.0f);

    scalar_add(A.data(), B.data(), C.data(), N);

    auto start = std::chrono::steady_clock::now();
    scalar_add(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> scalar_duration = end - start;
    std::cout << "Scalar addition took " << scalar_duration.count() << " seconds.\n";

    vectorized_add(A.data(), B.data(), C.data(), N);

    start = std::chrono::steady_clock::now();
    vectorized_add(A.data(), B.data(), C.data(), N);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> vectorized_duration = end - start;
    std::cout << "Vectorized addition took " << vectorized_duration.count() << " seconds.\n";

    return 0;
}

