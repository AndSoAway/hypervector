/*
 * Debug tool: compare RaBitQ estimated distances against exact L2.
 */

#include <index/flat/index_flat.h>
#include <quantization/rabitq/rabitq.h>
#include <utils/distances/distances.h>
#include <utils/structures/random.h>

#include <cstdio>
#include <cmath>
#include <vector>

using namespace hypervec;

int main() {
    const int d = 16, B = 1;
    const int n = 200;

    // Generate clustered data
    RandomGenerator rng(42);
    std::vector<float> train(n * d);
    for (int i = 0; i < n; i++) {
        float cluster = (i / 50) % 2 == 0 ? 10.0f : -10.0f;
        for (int j = 0; j < d; j++) {
            train[i * d + j] = cluster + 2.0f * rng.rand_float() - 1.0f;
        }
    }

    // Train RaBitQ quantizer
    RaBitQQuantizer rabitq(d, B);
    rabitq.Train(n, train.data());

    // Encode all data vectors
    std::vector<uint8_t> codes(n * rabitq.code_size);
    std::vector<float> inner_products(n);
    std::vector<float> norms(n);
    for (int i = 0; i < n; i++) {
        rabitq.ComputeCode(train.data() + i * d, codes.data() + i * rabitq.code_size);
        // Decode
        std::vector<float> decoded(d);
        rabitq.Decode(codes.data() + i * rabitq.code_size, decoded.data());
        float ip = 0;
        for (int j = 0; j < d; j++) ip += decoded[j] * train[i * d + j];
        inner_products[i] = ip;
        norms[i] = 1.0f; // placeholder
    }

    // Pick first data vector as "query"
    int qi = 0;
    const float* q = train.data();
    float q_norm = 0;
    for (int j = 0; j < d; j++) q_norm += q[j] * q[j];
    q_norm = std::sqrt(q_norm);

    // Preprocess query through RaBitQ pipeline
    // normalize and inverse transform
    std::vector<float> q_norm_vec(d);
    for (int j = 0; j < d; j++) q_norm_vec[j] = q[j] / q_norm;
    std::vector<float> q_transformed(d);
    rabitq.rot.InverseTransform(1, q_norm_vec.data(), q_transformed.data());

    float v_l, delta;
    std::vector<uint8_t> q_quantized(d);
    rabitq.QuantizeQuery(q_transformed.data(), q_quantized.data(), v_l, delta);

    uint8_t* bit_planes[4];
    uint8_t bp_mem[4 * 4];
    for (int bp = 0; bp < 4; bp++) bit_planes[bp] = bp_mem + bp * 2;
    rabitq.ComputeBitPlanes(q_quantized.data(), bit_planes, d);

    int sum_q = 0;
    for (int j = 0; j < d; j++) sum_q += q_quantized[j];

    // Compare estimated vs true distances
    printf("Comparing RaBitQ estimates vs exact L2 distances...\n");
    printf("Query vector index: 0\n\n");
    printf("%4s | %10s | %10s | %10s | %6s\n",
           "idx", "est_dist", "true_dist", "ratio", "same_dir");
    printf("------|------------|------------|------------|--------\n");

    double max_rel_err = 0;
    double sum_rel_err = 0;
    int n_correct = 0;

    for (int i = 0; i < n; i++) {
        int popcnt = 0;
        for (size_t b = 0; b < rabitq.code_size; b++) {
            popcnt += __builtin_popcount(codes[i * rabitq.code_size + b]);
        }

        float ip_q_o = rabitq.ComputeSingleCode(
            codes.data() + i * rabitq.code_size,
            const_cast<const uint8_t**>(bit_planes),
            sum_q, v_l, delta, popcnt);

        float ip_est = (inner_products[i] > 1e-10f)
            ? (ip_q_o / inner_products[i]) : 0.0f;

        // RaBitQ dist estimate (using norm=1 since data is not centroid-normalized)
        float est_dist = rabitq.EstimateDistance(ip_est, norms[i], q_norm, 0.0f);

        // True distance
        float true_dist = 0;
        for (int j = 0; j < d; j++) {
            float diff = train[i * d + j] - q[j];
            true_dist += diff * diff;
        }

        double ratio = true_dist > 1e-10 ? est_dist / true_dist : 1.0;
        double rel_err = std::abs(ratio - 1.0);

        sum_rel_err += rel_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (ratio > 0.99 && ratio < 1.01) n_correct++;

        if (i < 20 || rel_err > 0.5) {
            printf("%4d | %10.4f | %10.4f | %6.4f |\n",
                   i, est_dist, true_dist, ratio);
        }
    }

    printf("\n--- Summary ---\n");
    printf("Avg relative error: %.4f\n", sum_rel_err / n);
    printf("Max relative error: %.4f\n", max_rel_err);
    printf("Within 1%% accuracy: %d/%d\n", n_correct, n);

    // Check ⟨ō, o⟩ distribution
    printf("\n--- ⟨ō, o⟩ distribution ---\n");
    double sum_ip = 0, min_ip = 1, max_ip = 0;
    for (int i = 0; i < n; i++) {
        sum_ip += inner_products[i];
        if (inner_products[i] < min_ip) min_ip = inner_products[i];
        if (inner_products[i] > max_ip) max_ip = inner_products[i];
    }
    printf("Mean ⟨ō, o⟩: %.4f (paper expects ~0.8)\n", sum_ip / n);
    printf("Min ⟨ō, o⟩: %.4f\n", min_ip);
    printf("Max ⟨ō, o⟩: %.4f\n", max_ip);

    return 0;
}