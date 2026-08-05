/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <gtest/gtest.h>

#include <index/flat/index_flat.h>
#include <quantization/pq/pq.h>
#include <utils/distances/distances.h>
#include <utils/log/exception.h>
#include <utils/structures/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// Synthetic dataset of `k_true` Gaussian-jittered clusters arranged on a
// hypercube vertex pattern. Reused across PQ training tests because PQ on
// truly random data has weak structure to exploit.
std::vector<float> MakeClusteredData(hypervec::idx_t d, hypervec::idx_t k_true,
                                     hypervec::idx_t pts_per_cluster,
                                     int64_t seed) {
  hypervec::RandomGenerator rng(seed);
  std::vector<float> x(static_cast<size_t>(k_true) * pts_per_cluster * d);
  std::vector<float> centres(static_cast<size_t>(k_true) * d);
  for (hypervec::idx_t c = 0; c < k_true; c++) {
    for (hypervec::idx_t j = 0; j < d; j++) {
      centres[c * d + j] = (((c >> (j % 6)) & 1) ? 10.0f : -10.0f);
    }
  }
  for (hypervec::idx_t c = 0; c < k_true; c++) {
    for (hypervec::idx_t i = 0; i < pts_per_cluster; i++) {
      const hypervec::idx_t row = c * pts_per_cluster + i;
      for (hypervec::idx_t j = 0; j < d; j++) {
        const float jitter = 2.0f * rng.rand_float() - 1.0f;
        x[row * d + j] = centres[c * d + j] + jitter;
      }
    }
  }
  return x;
}

// recall@k: fraction of ground-truth neighbours present in the result set.
float Recall(const std::vector<hypervec::idx_t>& result,
             const std::vector<hypervec::idx_t>& gt, int nq, int k) {
  int hits = 0;
  for (int i = 0; i < nq; i++) {
    for (int j = 0; j < k; j++) {
      const hypervec::idx_t want = gt[i * k + j];
      for (int l = 0; l < k; l++) {
        if (result[i * k + l] == want) {
          hits++;
          break;
        }
      }
    }
  }
  return static_cast<float>(hits) / static_cast<float>(nq * k);
}

}  // namespace

// ===========================================================================
// Bit-packing roundtrip (independent of full PQ training)
// ===========================================================================

TEST(PQEncoderGeneric, RoundtripAllSupportedNbits) {
  // For every supported nbits, encode a random sequence of M values via the
  // generic packer and decode them; output must equal input. Spans byte
  // boundaries: M=64 with nbits=12 uses 96 bytes with split codes.
  for (int nbits : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
    const int M_test = 64;
    const uint64_t mask = (static_cast<uint64_t>(1) << nbits) - 1;
    const size_t code_size =
      (static_cast<size_t>(M_test) * nbits + 7) / 8;

    hypervec::RandomGenerator rng(42 + nbits);
    std::vector<uint64_t> input(M_test);
    for (int m = 0; m < M_test; m++) {
      // 32-bit random masked to nbits — covers the full code range.
      const uint64_t hi = static_cast<uint64_t>(rng.rand_int());
      const uint64_t lo = static_cast<uint64_t>(rng.rand_int());
      input[m] = ((hi << 31) ^ lo) & mask;
    }

    std::vector<uint8_t> code(code_size, 0);
    {
      hypervec::PQEncoderGeneric enc(code.data(), nbits);
      for (int m = 0; m < M_test; m++) {
        enc.encode(input[m]);
      }
    }  // destructor flushes any partial trailing byte

    std::vector<uint64_t> output(M_test);
    hypervec::PQDecoderGeneric dec(code.data(), nbits);
    for (int m = 0; m < M_test; m++) {
      output[m] = dec.decode();
    }

    for (int m = 0; m < M_test; m++) {
      ASSERT_EQ(input[m], output[m])
        << "nbits=" << nbits << " m=" << m;
    }
  }
}

TEST(PQEncoderFastPath, EightBitMatchesGeneric) {
  // PQEncoder8 / PQDecoder8 must produce byte-identical results to the
  // generic path for nbits=8. Same for nbits=16 below.
  const int nbits = 8;
  const int M_test = 32;
  hypervec::RandomGenerator rng(123);
  std::vector<uint64_t> input(M_test);
  for (int m = 0; m < M_test; m++) {
    input[m] = static_cast<uint64_t>(rng.rand_int(256));
  }

  std::vector<uint8_t> code_fast(M_test, 0);
  std::vector<uint8_t> code_gen(M_test, 0);
  {
    hypervec::PQEncoder8 enc(code_fast.data());
    for (int m = 0; m < M_test; m++) {
      enc.encode(input[m]);
    }
  }
  {
    hypervec::PQEncoderGeneric enc(code_gen.data(), nbits);
    for (int m = 0; m < M_test; m++) {
      enc.encode(input[m]);
    }
  }
  EXPECT_EQ(code_fast, code_gen);

  hypervec::PQDecoder8 dec_fast(code_fast.data());
  hypervec::PQDecoderGeneric dec_gen(code_gen.data(), nbits);
  for (int m = 0; m < M_test; m++) {
    EXPECT_EQ(dec_fast.decode(), input[m]);
    EXPECT_EQ(dec_gen.decode(), input[m]);
  }
}

TEST(PQEncoderFastPath, SixteenBitMatchesGeneric) {
  const int nbits = 16;
  const int M_test = 16;
  hypervec::RandomGenerator rng(456);
  std::vector<uint64_t> input(M_test);
  for (int m = 0; m < M_test; m++) {
    input[m] = static_cast<uint64_t>(rng.rand_int()) & 0xFFFFu;
  }

  std::vector<uint8_t> code_fast(M_test * 2, 0);
  std::vector<uint8_t> code_gen(M_test * 2, 0);
  {
    hypervec::PQEncoder16 enc(code_fast.data());
    for (int m = 0; m < M_test; m++) {
      enc.encode(input[m]);
    }
  }
  {
    hypervec::PQEncoderGeneric enc(code_gen.data(), nbits);
    for (int m = 0; m < M_test; m++) {
      enc.encode(input[m]);
    }
  }
  EXPECT_EQ(code_fast, code_gen);

  hypervec::PQDecoder16 dec_fast(code_fast.data());
  hypervec::PQDecoderGeneric dec_gen(code_gen.data(), nbits);
  for (int m = 0; m < M_test; m++) {
    EXPECT_EQ(dec_fast.decode(), input[m]);
    EXPECT_EQ(dec_gen.decode(), input[m]);
  }
}

// ===========================================================================
// Training & end-to-end encode/decode
// ===========================================================================

TEST(ProductQuantizer, TrainAndReconstructLowMSE) {
  // d=8, M=4 → dsub=2; nbits=4 → ksub=16. With 16 well-separated clusters
  // per subspace (k_true=16, pts_per_cluster=80), each subquantizer should
  // learn the cluster centres, so reconstruction MSE per dimension should
  // be small (< jitter² * something).
  const hypervec::idx_t d = 8, M = 4;
  const int nbits = 4;
  const hypervec::idx_t k_true = 16, pts_per_cluster = 80;
  const hypervec::idx_t n = k_true * pts_per_cluster;

  const std::vector<float> x = MakeClusteredData(d, k_true, pts_per_cluster, 7);

  hypervec::ProductQuantizer pq(d, M, nbits);
  hypervec::PQParameters params;
  params.nredo = 3;
  pq.Train(n, x.data(), params);
  ASSERT_TRUE(pq.is_trained);

  std::vector<uint8_t> codes(static_cast<size_t>(n) * pq.code_size);
  pq.ComputeCodes(n, x.data(), codes.data());

  std::vector<float> recon(static_cast<size_t>(n) * d);
  pq.DecodeBatch(n, codes.data(), recon.data());

  double mse = 0.0;
  for (hypervec::idx_t i = 0; i < n; i++) {
    for (hypervec::idx_t j = 0; j < d; j++) {
      const double diff = static_cast<double>(x[i * d + j]) -
                          static_cast<double>(recon[i * d + j]);
      mse += diff * diff;
    }
  }
  mse /= static_cast<double>(n) * d;

  // Jitter is uniform in [-1, 1], so per-dim variance ≈ 1/3. Reconstruction
  // can't beat the within-cluster jitter, so a generous bound is 2× that.
  EXPECT_LT(mse, 0.7) << "PQ reconstruction MSE too high: " << mse;
}

TEST(ProductQuantizer, EncodeDecodeRoundtripIsCentroidExact) {
  // Decoding a freshly encoded vector must yield the concatenated
  // subcentroids — no loss beyond that. Verify by re-encoding the decoded
  // result: it must produce byte-identical codes (because each subvector
  // is now exactly its own nearest centroid).
  const hypervec::idx_t d = 8, M = 4;
  const int nbits = 5;  // ksub=32, generic bit-packing path
  const hypervec::idx_t n = 200;
  const std::vector<float> x = MakeClusteredData(d, 32, n / 32, 23);

  hypervec::ProductQuantizer pq(d, M, nbits);
  pq.Train(n, x.data());

  std::vector<uint8_t> codes1(static_cast<size_t>(n) * pq.code_size);
  pq.ComputeCodes(n, x.data(), codes1.data());

  std::vector<float> decoded(static_cast<size_t>(n) * d);
  pq.DecodeBatch(n, codes1.data(), decoded.data());

  std::vector<uint8_t> codes2(static_cast<size_t>(n) * pq.code_size);
  pq.ComputeCodes(n, decoded.data(), codes2.data());

  EXPECT_EQ(codes1, codes2);
}

// ===========================================================================
// Distance table correctness
// ===========================================================================

TEST(ProductQuantizer, DistanceTableEqualsFvecL2) {
  // dis_table[m, k] must equal fvec_L2sqr(query_subvector_m, centroid_{m,k}).
  const hypervec::idx_t d = 16, M = 4;
  const int nbits = 6;
  const hypervec::idx_t n_train = 1000;
  const std::vector<float> xt = MakeClusteredData(d, 64, 16, 1);

  hypervec::ProductQuantizer pq(d, M, nbits);
  pq.Train(n_train, xt.data());

  hypervec::RandomGenerator rng(99);
  std::vector<float> q(d);
  for (hypervec::idx_t j = 0; j < d; j++) {
    q[j] = 20.0f * (rng.rand_float() - 0.5f);
  }

  std::vector<float> dis_table(static_cast<size_t>(M) * pq.ksub);
  pq.ComputeDistanceTable(q.data(), dis_table.data());

  for (hypervec::idx_t m = 0; m < M; m++) {
    for (hypervec::idx_t k = 0; k < pq.ksub; k++) {
      const float expected = hypervec::fvec_L2sqr(
        q.data() + m * pq.dsub, pq.GetCentroids(m, k),
        static_cast<size_t>(pq.dsub));
      EXPECT_NEAR(dis_table[m * pq.ksub + k], expected, 1e-4f)
        << "m=" << m << " k=" << k;
    }
  }
}

TEST(ProductQuantizer, AdcSumEqualsL2OfDecodedVector) {
  // The fundamental ADC identity:
  //   Σ_m dis_table[m, code[m]] == fvec_L2sqr(query, decode(code))
  // because each subvector contributes independently to L2².
  const hypervec::idx_t d = 16, M = 4;
  const int nbits = 8;
  const hypervec::idx_t n = 1000;
  const std::vector<float> x = MakeClusteredData(d, 64, n / 64, 41);

  hypervec::ProductQuantizer pq(d, M, nbits);
  pq.Train(n, x.data());

  std::vector<uint8_t> codes(static_cast<size_t>(n) * pq.code_size);
  pq.ComputeCodes(n, x.data(), codes.data());

  hypervec::RandomGenerator rng(77);
  std::vector<float> q(d);
  for (hypervec::idx_t j = 0; j < d; j++) {
    q[j] = 20.0f * (rng.rand_float() - 0.5f);
  }

  std::vector<float> dis_table(static_cast<size_t>(M) * pq.ksub);
  pq.ComputeDistanceTable(q.data(), dis_table.data());

  std::vector<float> decoded(d);
  for (hypervec::idx_t i : {hypervec::idx_t(0), hypervec::idx_t(17),
                            hypervec::idx_t(123), hypervec::idx_t(999)}) {
    pq.Decode(codes.data() + i * pq.code_size, decoded.data());
    const float reference = hypervec::fvec_L2sqr(
      q.data(), decoded.data(), static_cast<size_t>(d));

    float adc_sum = 0.0f;
    for (hypervec::idx_t m = 0; m < M; m++) {
      adc_sum += dis_table[m * pq.ksub + codes[i * pq.code_size + m]];
    }
    EXPECT_NEAR(adc_sum, reference, 1e-3f * std::max(reference, 1.0f))
      << "i=" << i;
  }
}

// ===========================================================================
// SearchL2 vs IndexFlatL2 ground truth
// ===========================================================================

TEST(ProductQuantizer, SearchL2RecallVsBruteForce) {
  // PQ is lossy, so we can't expect 100% recall. With a small d and
  // moderate nbits, recall@10 should still be well above 50% on clustered
  // data. The exact threshold is loose to avoid CI flakiness.
  const hypervec::idx_t d = 16, M = 8;  // dsub=2
  const int nbits = 8;                  // ksub=256
  const hypervec::idx_t nb = 3000;
  const hypervec::idx_t nq = 100;
  const hypervec::idx_t k = 10;

  hypervec::RandomGenerator rng(2026);
  std::vector<float> base(static_cast<size_t>(nb) * d);
  std::vector<float> query(static_cast<size_t>(nq) * d);
  for (auto& v : base) {
    v = 5.0f * rng.rand_float();
  }
  for (auto& v : query) {
    v = 5.0f * rng.rand_float();
  }

  // Ground truth from exact IndexFlatL2.
  hypervec::IndexFlatL2 gt(d);
  gt.Add(nb, base.data());
  std::vector<float> gt_dists(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> gt_labels(static_cast<size_t>(nq) * k);
  gt.Search(nq, query.data(), k, gt_dists.data(), gt_labels.data());

  // Train and encode.
  hypervec::ProductQuantizer pq(d, M, nbits);
  hypervec::PQParameters params;
  params.nredo = 2;
  pq.Train(nb, base.data(), params);
  std::vector<uint8_t> codes(static_cast<size_t>(nb) * pq.code_size);
  pq.ComputeCodes(nb, base.data(), codes.data());

  std::vector<float> pq_dists(static_cast<size_t>(nq) * k);
  std::vector<hypervec::idx_t> pq_labels(static_cast<size_t>(nq) * k);
  pq.SearchL2(nq, query.data(), nb, codes.data(), k, pq_dists.data(),
              pq_labels.data());

  const float recall = Recall(pq_labels, gt_labels, nq, k);
  EXPECT_GT(recall, 0.5f) << "PQ search recall@" << k
                          << " too low: " << recall;
}

// ===========================================================================
// Parameter validation
// ===========================================================================

TEST(ProductQuantizer, RejectsDNotDivisibleByM) {
  EXPECT_THROW(hypervec::ProductQuantizer(10, 3, 8),
               hypervec::HypervecException);
}

TEST(ProductQuantizer, RejectsNbitsOutOfRange) {
  EXPECT_THROW(hypervec::ProductQuantizer(8, 4, 0),
               hypervec::HypervecException);
  EXPECT_THROW(hypervec::ProductQuantizer(8, 4, 17),
               hypervec::HypervecException);
}

TEST(ProductQuantizer, RejectsTooFewTrainingVectors) {
  hypervec::ProductQuantizer pq(8, 4, 8);  // ksub=256
  std::vector<float> x(100 * 8, 0.0f);     // n=100 < ksub=256
  EXPECT_THROW(pq.Train(100, x.data()), hypervec::HypervecException);
}

TEST(ProductQuantizer, ComputeBeforeTrainingThrows) {
  hypervec::ProductQuantizer pq(8, 4, 4);
  std::vector<float> x(8, 0.0f);
  std::vector<uint8_t> code(pq.code_size);
  EXPECT_THROW(pq.ComputeCode(x.data(), code.data()),
               hypervec::HypervecException);
}
