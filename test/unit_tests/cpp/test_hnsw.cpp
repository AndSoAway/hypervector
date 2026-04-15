/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <index/hnsw/index_hnsw.h>
#include <index/hnsw/hnsw.h>
#include <search/result_handler.h>
#include <index/hnsw/visited_table.h>
#include <utils/structures/random.h>

int ReferencePopMin(hypervec::HNSW::MinimaxHeap& heap, float* vmin_out) {
    assert(heap.k > 0);
    // returns min. This is an O(n) operation
    int i = heap.k - 1;
    while (i >= 0) {
        if (heap.ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = heap.dis[i];
    i--;
    while (i >= 0) {
        if (heap.ids[i] != -1 && heap.dis[i] < vmin) {
            vmin = heap.dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = heap.ids[imin];
    heap.ids[imin] = -1;
    --heap.nvalid;

    return ret;
}

void TestPopmin(int heap_size, int amount_to_put) {
    // create a heap
    hypervec::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = hypervec::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);
    std::uniform_real_distribution<float> uf(0, 1);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        float distance = uf(rng);
        if (distance >= 0.7f) {
            // Add infinity values from time to time
            distance = std::numeric_limits<float>::infinity();
        }
        mm_heap.push(index, distance);
    }

    // clone the heap
    hypervec::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference PopMin for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                ReferencePopMin(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.PopMin(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

void TestPopminIdenticalDistances(
        int heap_size,
        int amount_to_put,
        const float distance) {
    // create a heap
    hypervec::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = hypervec::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        mm_heap.push(index, distance);
    }

    // clone the heap
    hypervec::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference PopMin for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                ReferencePopMin(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.PopMin(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

TEST(HNSW, Popmin) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32, 64, 128};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            TestPopmin(size, amount);
        }
    }
}

TEST(HNSW, PopminIdenticalDistances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            TestPopminIdenticalDistances(size, amount, 1.0f);
        }
    }
}

TEST(HNSW, PopminInfiniteDistances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            TestPopminIdenticalDistances(
                    size, amount, std::numeric_limits<float>::infinity());
        }
    }
}

TEST(HNSW, IndexHNSWMetricLp) {
    // Create an HNSW index with kMetricLp and metric_arg = 3
    hypervec::IndexFlat storage_index(1, hypervec::kMetricLp);
    storage_index.metric_arg = 3;
    hypervec::IndexHNSW index(&storage_index, 32);

    // Add a single data point
    float data[1] = {0.0};
    index.Add(1, data);

    // Prepare a query
    float query[1] = {2.0};
    float distance;
    hypervec::idx_t label;

    index.Search(1, query, 1, &distance, &label);

    EXPECT_NEAR(distance, 8.0, 1e-5); // Distance should be 8.0 (2^3)
    EXPECT_EQ(label, 0);              // Label should be 0
}

class HNSWTest : public testing::Test {
   protected:
    HNSWTest() {
        xb = std::make_unique<std::vector<float>>(d * nb);
        xb->reserve(d * nb);
        hypervec::FloatRand(xb->data(), d * nb, 12345);
        index = std::make_unique<hypervec::IndexHNSWFlat>(d, M);
        index->Add(nb, xb->data());
        xq = std::unique_ptr<std::vector<float>>(
                new std::vector<float>(d * nq));
        xq->reserve(d * nq);
        hypervec::FloatRand(xq->data(), d * nq, 12345);
        dis = std::unique_ptr<hypervec::DistanceComputer>(
                index->storage->GetDistanceComputer());
        dis->SetQuery(xq->data() + 0 * index->d);
    }

    const int d = 64;
    const int nb = 2000;
    const int M = 4;
    const int nq = 10;
    const int k = 10;
    std::unique_ptr<std::vector<float>> xb;
    std::unique_ptr<std::vector<float>> xq;
    std::unique_ptr<hypervec::DistanceComputer> dis;
    std::unique_ptr<hypervec::IndexHNSWFlat> index;
};

/** Do a BFS on the candidates list */
int ReferenceSearchFromCandidates(
        const hypervec::HNSW& hnsw,
        hypervec::DistanceComputer& qdis,
        hypervec::ResultHandler& res,
        hypervec::HNSW::MinimaxHeap& candidates,
        hypervec::VisitedTable& vt,
        hypervec::HNSWStats& stats,
        int level,
        int nres_in,
        const hypervec::SearchParametersHNSW* params) {
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by Search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int ef_search = params ? params->ef_search : hnsw.ef_search;
    const hypervec::IDSelector* sel = params ? params->sel : nullptr;

    hypervec::HNSW::C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        hypervec::idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        HYPERVEC_ASSERT(v1 >= 0);
        if (!sel || sel->IsMember(v1)) {
            if (d < threshold) {
                if (res.AddResult(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.PopMin(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.CountBelow(d0);
            if (n_dis_below >= ef_search) {
                break;
            }
        }

        size_t begin, end;
        hnsw.NeighborRange(v0, level, &begin, &end);

        // a reference version
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->IsMember(v1)) {
                if (d < threshold) {
                    if (res.AddResult(d, v1)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }

            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > ef_search) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

hypervec::HNSWStats ReferenceGreedyUpdateNearest(
        const hypervec::HNSW& hnsw,
        hypervec::DistanceComputer& qdis,
        int level,
        hypervec::HNSW::storage_idx_t& nearest,
        float& d_nearest) {
    hypervec::HNSWStats stats;

    for (;;) {
        hypervec::HNSW::storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.NeighborRange(nearest, level, &begin, &end);

        size_t ndis = 0;

        for (size_t i = begin; i < end; i++) {
            hypervec::HNSW::storage_idx_t v = hnsw.neighbors[i];
            if (v < 0) {
                break;
            }
            ndis += 1;
            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

std::priority_queue<hypervec::HNSW::Node> ReferenceSearchFromCandidateUnbounded(
        const hypervec::HNSW& hnsw,
        const hypervec::HNSW::Node& node,
        hypervec::DistanceComputer& qdis,
        int ef,
        hypervec::VisitedTable* vt,
        hypervec::HNSWStats& stats) {
    int ndis = 0;
    std::priority_queue<hypervec::HNSW::Node> top_candidates;
    std::priority_queue<
            hypervec::HNSW::Node,
            std::vector<hypervec::HNSW::Node>,
            std::greater<hypervec::HNSW::Node>>
            candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        hypervec::HNSW::storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.NeighborRange(v0, 0, &begin, &end);

        for (size_t j = begin; j < end; ++j) {
            int v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

TEST_F(HNSWTest, SearchFromCandidateUnbounded) {
    omp_set_num_threads(1);
    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto node = hypervec::HNSW::Node(d_nearest, nearest);
    hypervec::VisitedTable vt(index->n_total);
    hypervec::HNSWStats stats;

    // actual version
    auto top_candidates = hypervec::SearchFromCandidateUnbounded(
            index->hnsw, node, *dis, k, &vt, stats);

    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(nearest);
    auto reference_node =
            hypervec::HNSW::Node(reference_d_nearest, reference_nearest);
    hypervec::VisitedTable reference_vt(index->n_total);
    hypervec::HNSWStats reference_stats;

    // reference version
    auto reference_top_candidates = ReferenceSearchFromCandidateUnbounded(
            index->hnsw,
            reference_node,
            *dis,
            k,
            &reference_vt,
            reference_stats);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_EQ(top_candidates.size(), reference_top_candidates.size());
}

TEST_F(HNSWTest, GreedyUpdateNearest) {
    omp_set_num_threads(1);

    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(reference_nearest);

    // actual version
    auto stats = hypervec::GreedyUpdateNearest(
            index->hnsw, *dis, 0, nearest, d_nearest);

    // reference version
    auto reference_stats = ReferenceGreedyUpdateNearest(
            index->hnsw, *dis, 0, reference_nearest, reference_d_nearest);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_NEAR(d_nearest, reference_d_nearest, 0.01);
    EXPECT_EQ(nearest, reference_nearest);
}

TEST_F(HNSWTest, SearchFromCandidates) {
    omp_set_num_threads(1);

    std::vector<hypervec::idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    std::vector<hypervec::idx_t> reference_I(k * nq);
    std::vector<float> reference_D(k * nq);
    using RH = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;

    hypervec::VisitedTable vt(index->n_total);
    hypervec::VisitedTable reference_vt(index->n_total);
    int num_candidates = 10;
    hypervec::HNSW::MinimaxHeap candidates(num_candidates);
    hypervec::HNSW::MinimaxHeap reference_candidates(num_candidates);

    for (int i = 0; i < num_candidates; i++) {
        vt.set(i);
        reference_vt.set(i);
        candidates.push(i, (*dis)(i));
        reference_candidates.push(i, (*dis)(i));
    }

    hypervec::HNSWStats stats;
    RH bres(nq, D.data(), I.data(), k);
    hypervec::HeapBlockResultHandler<hypervec::HNSW::C>::SingleResultHandler res(
            bres);

    res.begin(0);
    hypervec::SearchFromCandidates(
            index->hnsw, *dis, res, candidates, vt, stats, 0, 0, nullptr);
    res.end();

    hypervec::HNSWStats reference_stats;
    RH reference_bres(nq, reference_D.data(), reference_I.data(), k);
    hypervec::HeapBlockResultHandler<hypervec::HNSW::C>::SingleResultHandler
            reference_res(reference_bres);
    reference_res.begin(0);
    ReferenceSearchFromCandidates(
            index->hnsw,
            *dis,
            reference_res,
            reference_candidates,
            reference_vt,
            reference_stats,
            0,
            0,
            nullptr);
    reference_res.end();
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(I[i * k + j], reference_I[i * k + j], 0.1);
            EXPECT_NEAR(D[i * k + j], reference_D[i * k + j], 0.1);
        }
    }
    EXPECT_EQ(reference_stats.ndis, stats.ndis);
    EXPECT_EQ(reference_stats.nhops, stats.nhops);
    EXPECT_EQ(reference_stats.n1, stats.n1);
    EXPECT_EQ(reference_stats.n2, stats.n2);
}

TEST_F(HNSWTest, SearchNeighborsToAdd) {
    omp_set_num_threads(1);

    hypervec::VisitedTable vt(index->n_total);
    hypervec::VisitedTable reference_vt(index->n_total);

    std::priority_queue<hypervec::HNSW::NodeDistCloser> link_targets;
    std::priority_queue<hypervec::HNSW::NodeDistCloser> reference_link_targets;

    hypervec::SearchNeighborsToAdd(
            index->hnsw,
            *dis,
            link_targets,
            index->hnsw.entry_point,
            (*dis)(index->hnsw.entry_point),
            index->hnsw.max_level,
            vt,
            false);

    hypervec::SearchNeighborsToAdd(
            index->hnsw,
            *dis,
            reference_link_targets,
            index->hnsw.entry_point,
            (*dis)(index->hnsw.entry_point),
            index->hnsw.max_level,
            reference_vt,
            true);

    EXPECT_EQ(link_targets.size(), reference_link_targets.size());
    while (!link_targets.empty()) {
        auto val = link_targets.top();
        auto reference_val = reference_link_targets.top();
        EXPECT_EQ(val.d, reference_val.d);
        EXPECT_EQ(val.id, reference_val.id);
        link_targets.pop();
        reference_link_targets.pop();
    }
}

TEST_F(HNSWTest, NbNeighborsBound) {
    omp_set_num_threads(1);
    EXPECT_EQ(index->hnsw.NbNeighbors(0), 8);
    EXPECT_EQ(index->hnsw.NbNeighbors(1), 4);
    EXPECT_EQ(index->hnsw.NbNeighbors(2), 4);
    EXPECT_EQ(index->hnsw.NbNeighbors(3), 4);
    // picking a large number to trigger an exception based on checking bounds
    EXPECT_THROW(index->hnsw.NbNeighbors(100), hypervec::HypervecException);
}

TEST_F(HNSWTest, SearchLevel0) {
    omp_set_num_threads(1);
    std::vector<hypervec::idx_t> I(k * nq);
    std::vector<float> D(k * nq);

    using RH = hypervec::HeapBlockResultHandler<hypervec::HNSW::C>;
    RH bres1(nq, D.data(), I.data(), k);
    hypervec::HeapBlockResultHandler<hypervec::HNSW::C>::SingleResultHandler res1(
            bres1);
    RH bres2(nq, D.data(), I.data(), k);
    hypervec::HeapBlockResultHandler<hypervec::HNSW::C>::SingleResultHandler res2(
            bres2);

    hypervec::HNSWStats stats1, stats2;
    hypervec::VisitedTable vt1(index->n_total);
    hypervec::VisitedTable vt2(index->n_total);
    auto nprobe = 5;
    const hypervec::HNSW::storage_idx_t values[] = {1, 2, 3, 4, 5};
    const hypervec::HNSW::storage_idx_t* nearest_i = values;
    const float distances[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    const float* nearest_d = distances;

    // search_type == 1
    res1.begin(0);
    index->hnsw.SearchLevel0(
            *dis, res1, nprobe, nearest_i, nearest_d, 1, stats1, vt1, nullptr);
    res1.end();

    // search_type == 2
    res2.begin(0);
    index->hnsw.SearchLevel0(
            *dis, res2, nprobe, nearest_i, nearest_d, 2, stats2, vt2, nullptr);
    res2.end();

    // search_type 1 calls SearchFromCandidates in a loop nprobe times.
    // search_type 2 pushes the candidates and just calls SearchFromCandidates
    // once, so those stats will be much less.
    EXPECT_GT(stats1.ndis, stats2.ndis);
    EXPECT_GT(stats1.nhops, stats2.nhops);
    EXPECT_GT(stats1.n1, stats2.n1);
    EXPECT_GT(stats1.n2, stats2.n2);
}
