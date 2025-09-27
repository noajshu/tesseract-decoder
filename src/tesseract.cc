// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tesseract.h"

#include <algorithm>
#include <boost/functional/hash.hpp>  // For boost::hash_range
#include <cassert>
#include <functional>  // For std::hash (though not strictly necessary here, but good practice)
#include <iostream>
#include <chrono>

namespace {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (auto& x : vec) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    os << x;
  }
  os << "]";
  return os;
}

};  // namespace

namespace std {
template <>
struct hash<boost::dynamic_bitset<>> {
  size_t operator()(const boost::dynamic_bitset<>& bs) const {
    // Delegate to Boost's internal hash_value for dynamic_bitset
    // This is the correct and most efficient way.
    return boost::hash_value(bs);
  }
};
}  // namespace std

std::string TesseractConfig::str() {
  auto& config = *this;
  std::stringstream ss;
  ss << "TesseractConfig(";
  ss << "dem=DetectorErrorModel_Object" << ", ";
  ss << "det_beam=" << config.det_beam << ", ";
  ss << "no_revisit_dets=" << config.no_revisit_dets << ", ";

  ss << "verbose=" << config.verbose << ", ";
  ss << "merge_errors=" << config.merge_errors << ", ";
  ss << "pqlimit=" << config.pqlimit << ", ";
  ss << "det_orders=" << config.det_orders << ", ";
  ss << "det_penalty=" << config.det_penalty << ", ";
  ss << "create_visualization=" << config.create_visualization;
  ss << ")";
  return ss.str();
}

std::string Node::str() const {
  std::stringstream ss;
  auto& self = *this;
  ss << "Node(";
  ss << "errors=" << self.errors << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  ss << "num_fresh_dets=" << self.num_fresh_dets << ", ";
  return ss.str();
}

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  double min_cost = INF;
  double error_cost;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (int ei : d2e[d]) {
    ec = error_costs[ei];
    if (ec.min_cost >= min_cost) break;

    dct = detector_cost_tuples[ei];
    if (!dct.error_blocked) {
      error_cost = ec.likelihood_cost / dct.num_dets;
      if (error_cost < min_cost) {
        min_cost = error_cost;
      }
    }
  }

  return min_cost + config.det_penalty;
}

TesseractDecoder::TesseractDecoder(TesseractConfig config_) : config(config_) {
  if (config.merge_errors) {
    config.dem = common::merge_indistinguishable_errors(config.dem);
  }
  config.dem = common::remove_zero_probability_errors(config.dem);

  if (config.det_orders.empty()) {
    config.det_orders.emplace_back(config.dem.count_detectors());
    std::iota(config.det_orders[0].begin(), config.det_orders[0].end(), 0);
  } else {
    for (size_t i = 0; i < config.det_orders.size(); ++i) {
      if (config.det_orders[i].size() != config.dem.count_detectors()) {
        throw std::invalid_argument(
            "Each detector order list must have a size equal to the number of detectors.");
      }
    }
  }
  if (config.det_orders.empty()) {
    throw std::runtime_error("After initialization, detector orders list must not be empty.");
  }
  errors = get_errors_from_dem(config.dem.flattened());
  if (config.verbose) {
    for (auto& error : errors) {
      std::cout << error.str() << "\n";
    }
    std::cout << std::flush;
  }
  num_detectors = config.dem.count_detectors();
  num_errors = config.dem.count_errors();
  num_observables = config.dem.count_observables();
  initialize_structures(config.dem.count_detectors());
  if (config.create_visualization) {
    visualizer.add_detector_coords(get_detector_coords(config.dem));
    visualizer.add_errors(errors);
  }
}

void TesseractDecoder::initialize_structures(size_t num_detectors) {
  d2e.resize(num_detectors);
  edets.resize(num_errors);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[d].push_back(ei);
    }
  }

  for (size_t i = 0; i < errors.size(); ++i) {
    error_costs.push_back({errors[i].likelihood_cost,
                           errors[i].likelihood_cost / errors[i].symptom.detectors.size()});
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost;
    });
  }

  eneighbors.resize(num_errors);

  std::vector<boost::dynamic_bitset<>> edets_bitsets(num_errors,
                                                     boost::dynamic_bitset<>(num_detectors));
  for (size_t ei = 0; ei < num_errors; ++ei) {
    for (int d : edets[ei]) {
      edets_bitsets[ei][d] = 1;
    }
  }

  for (size_t ei = 0; ei < num_errors; ++ei) {
    boost::dynamic_bitset<> neighbor_set(num_detectors, false);
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        // Unify detectors from neighboring errors
        neighbor_set |= edets_bitsets[oei];
      }
    }
    // Remove detectors from error's own set
    neighbor_set &= ~edets_bitsets[ei];

    for (size_t d = neighbor_set.find_first(); d != boost::dynamic_bitset<>::npos;
         d = neighbor_set.find_next(d)) {
      eneighbors[ei].push_back(d);
    }
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  if (config.det_orders.empty()) {
    throw std::runtime_error("Detector orders list must not be empty before decoding.");
  }

  if (config.beam_climbing) {
    int beam = 0;
    int detector_order = 0;
    for (int trial = 0; trial < std::max(config.det_beam + 1, int(config.det_orders.size()));
         ++trial) {
      decode_to_errors(detections, detector_order, beam);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
      beam += 1;
      detector_order += 1;
      beam %= (config.det_beam + 1);
      detector_order %= config.det_orders.size();
    }
  } else {
    for (size_t detector_order = 0; detector_order < config.det_orders.size(); ++detector_order) {
      decode_to_errors(detections, detector_order, config.det_beam);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << config.det_beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = best_cost == std::numeric_limits<double>::max();
}

size_t TesseractDecoder::get_min_det(size_t detector_order, const boost::dynamic_bitset<>& dets,
                                     const boost::dynamic_bitset<>& initial_dets,
                                     const std::vector<uint64_t>& seed_dets) const {
  // This must only return dets in the seed dets or fresh dets
  // Now look for seed dets
  for (uint64_t d : seed_dets) {
    if (dets[d]) {
      return d;
    }
  }
  // Look for fresh dets
  for (size_t d = 0; d < num_detectors; ++d) {
    size_t dod = config.det_orders[detector_order][d];
    if (dets[dod] and !initial_dets[dod]) {
      // If this is a fresh det
      return dod;
    }
  }
  return std::numeric_limits<size_t>::max();
}

void TesseractDecoder::flip_detectors_and_block_errors(
    size_t detector_order, const std::vector<size_t>& errors, boost::dynamic_bitset<>& dets,
    const boost::dynamic_bitset<>& initial_dets, const std::vector<uint64_t>& seed_dets,
    std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  for (size_t ei : errors) {
    size_t min_det = get_min_det(detector_order, dets, initial_dets, seed_dets);

    for (size_t oei : d2e[min_det]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (oei == ei) break;
    }

    for (int d : edets[ei]) {
      dets[d] = !dets[d];
    }
  }
}

static size_t counter = 0;

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                        size_t detector_order, size_t detector_beam) {
  std::vector<std::vector<uint64_t>> seeds;
  for (uint64_t d : detections) {
    seeds.push_back({d});
  }
  // seeds.push_back(detections);

  struct SeedDecodeResult {
    std::vector<size_t> predicted_errors;
    std::set<uint64_t> shell_errors;
    std::set<uint64_t> shell_dets;
    bool needs_recomputing = true;
  };
  std::vector<SeedDecodeResult> seed_results(seeds.size());

  while (true) {
    if (config.verbose) {
      std::cout << "Starting clustering iteration with " << seeds.size() << " seeds." << std::endl;
    }
    // Used to find collisions between shells
    std::vector<std::vector<uint64_t>> error_to_seeds(num_errors);
    std::vector<std::vector<uint64_t>> det_to_seeds(num_detectors);
    // Find the optimal resolutions for each seed
    std::vector<size_t> predicted_errors_concat;
    std::vector<size_t> concat_seeds;
    size_t seeds_to_redecode = 0;
    for (size_t si = 0; si < seeds.size(); ++si) {
      if (seed_results[si].needs_recomputing) {
        seeds_to_redecode++;
      }
    }
    if (config.verbose) {
      std::cout << "Number of seeds to re-decode: " << seeds_to_redecode << " / " << seeds.size()
                << std::endl;
    }
    for (size_t si = 0; si < seeds.size(); ++si) {
      const auto& seed = seeds[si];
      for (size_t d : seed) {
        concat_seeds.push_back(d);
      }
      // The seed dets are the detection event indices for which we want to search for the optimal
      // resolution node The shell errors (like shell area in Blossom) are the errors that we
      // considered using (i.e., for any node visited during the search)
      if (seed_results[si].needs_recomputing) {
        // std::cout << "decoding for seed " << si <<" with " << seed.size() <<" detection
        // events"<<std::endl;
        ++counter;
        // std::cout<<"counter = "<<counter<<std::endl;
        // if (counter == 8) {
        //   config.verbose=true;
        // } else {
        //   config.verbose=false;
        // }
        predicted_errors_buffer.clear();
        seed_results[si].shell_errors.clear();
        seed_results[si].shell_dets.clear();
        auto start_time = std::chrono::steady_clock::now();
        decode_to_errors_helper(detections, detector_order, detector_beam, seed,
                                seed_results[si].shell_errors, seed_results[si].shell_dets);
        auto end_time = std::chrono::steady_clock::now();
        double num_milliseconds =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        if (config.verbose) {
          std::cout << "Decoding seed " << si << " with " << seed.size()
                    << " detection events took " << num_milliseconds << " ms." << std::endl;
        }
        assert(!low_confidence_flag);
        seed_results[si].predicted_errors = predicted_errors_buffer;
        seed_results[si].needs_recomputing = false;
      }
      predicted_errors_concat.insert(predicted_errors_concat.end(),
                                     seed_results[si].predicted_errors.begin(),
                                     seed_results[si].predicted_errors.end());
      // std::cout<<"got shell_errors.size() = " << shell_errors.size()
      //           <<" shell_dets.size() = "<<shell_dets.size()
      //           << " predicted_errors_buffer.size() = " << predicted_errors_buffer.size()
      //           << std::endl;
      for (uint64_t ei : seed_results[si].shell_errors) {
        error_to_seeds[ei].push_back(si);
      }
      for (uint64_t d : seed_results[si].shell_dets) {
        det_to_seeds[d].push_back(si);
      }
    }

    std::sort(concat_seeds.begin(), concat_seeds.end());
    // std::cout<<"concat_seeds = ";
    // for(size_t d:concat_seeds) {
    //   std::cout<<d<<", ";
    // }
    // std::cout<<std::endl;
    for (size_t i = 0; i + 1 < concat_seeds.size(); ++i) {
      assert(concat_seeds[i + 1] != concat_seeds[i]);
    }
    assert(concat_seeds == detections);
    // Simple union-find data structure for the merging of seeds
    std::vector<size_t> parents(seeds.size());
    // Parents begins with the
    std::iota(parents.begin(), parents.end(), 0);
    // Find, but without any path compression for simplicity (ok if clusters are small)
    auto find = [&parents](size_t u) -> size_t {
      size_t v;
      while (true) {
        v = parents[u];
        if (u == v) return v;
        u = v;
      }
    };
    auto do_union = [&parents, &find](size_t u, size_t v) {
      u = find(u);
      v = find(v);
      parents[v] = u;
    };

    bool collision = false;
    for (size_t ei = 0; ei < num_errors; ++ei) {
      if (error_to_seeds[ei].size() > 1) {
        // std::cout << "shell collision on error " << ei << " with " << error_to_seeds[ei].size()
        //           << " seeds" << std::endl;
        collision = true;
      }
      // Merge the seeds that collided
      for (size_t i = 1; i < error_to_seeds[ei].size(); ++i) {
        do_union(error_to_seeds[ei][0], error_to_seeds[ei][i]);
      }
    }
    for (size_t d = 0; d < num_detectors; ++d) {
      if (det_to_seeds[d].size() > 1) {
        collision = true;
      }
      // Merge the seeds that collided
      for (size_t i = 1; i < det_to_seeds[d].size(); ++i) {
        do_union(det_to_seeds[d][0], det_to_seeds[d][i]);
      }
    }
    // // Hack test
    // for (size_t si=1; si<seeds.size(); ++si) {
    //   do_union(0, si);
    //   collision = true;
    // }

    if (!collision) {
      std::sort(predicted_errors_concat.begin(), predicted_errors_concat.end());
      // std::cout<<"predicted_errors_concat = ";
      // for (size_t ei: predicted_errors_concat) {
      //   std::cout<<ei<<", ";
      // }
      // std::cout<<std::endl;
      for (size_t i = 0; i + 1 < predicted_errors_concat.size(); ++i) {
        assert(predicted_errors_concat[i] != predicted_errors_concat[i + 1]);
      }
      predicted_errors_buffer = predicted_errors_concat;
      boost::dynamic_bitset<> predicted_dets(num_detectors, false);
      for (size_t ei : predicted_errors_buffer) {
        for (size_t d : edets[ei]) {
          predicted_dets[d] ^= true;
        }
      }
      boost::dynamic_bitset<> original_dets(num_detectors, false);
      for (size_t d : detections) {
        original_dets[d] = true;
      }
      // std::cout<<"original_dets =  ";
      // for (size_t d=0; d<num_detectors; ++d) {
      //   if (original_dets[d]) {
      //     std::cout<<d<<", ";
      //   }
      // }
      // std::cout<<std::endl;
      // std::cout<<"predicted_dets = ";
      // for (size_t d=0; d<num_detectors; ++d) {
      //   if (predicted_dets[d]) {
      //     std::cout<<d<<", ";
      //   }
      // }
      // std::cout<<std::endl;

      assert(predicted_dets == original_dets);
      return;
    }
    // At this point we need to extract the new seeds. We assign each one an index.
    std::map<size_t, size_t> root_indices;
    for (size_t si = 0; si < parents.size(); ++si) {
      if (parents[si] == si) {
        root_indices[si] = root_indices.size();
      }
    }
    if (config.verbose) {
      std::cout << "Number of seeds after merging: " << root_indices.size() << std::endl;
    }
    // std::cout<<"root_indices = ";
    // for (auto & [r, i] : root_indices) {
    //   std::cout<<"("<<r<<","<<i<<") ";
    // }
    // std::cout<<std::endl;
    std::vector<std::vector<uint64_t>> next_seeds(root_indices.size());
    std::vector<SeedDecodeResult> next_seed_results(root_indices.size());
    std::vector<size_t> root_to_child_count(seeds.size(), 0);
    for (size_t si = 0; si < seeds.size(); ++si) {
      root_to_child_count[find(si)]++;
    }

    for (size_t si = 0; si < seeds.size(); ++si) {
      size_t root = find(si);
      size_t root_idx = root_indices[root];
      std::vector<uint64_t>& seed = next_seeds[root_idx];
      for (size_t d : seeds[si]) {
        seed.push_back(d);
      }
      if (root_to_child_count[root] > 1) {
        next_seed_results[root_idx].needs_recomputing = true;
      } else {
        next_seed_results[root_idx] = seed_results[si];
      }
    }

    for (size_t si = 0; si < next_seeds.size(); ++si) {
      std::sort(next_seeds[si].begin(), next_seeds[si].end());
    }
    std::swap(next_seeds, seeds);
    std::swap(next_seed_results, seed_results);
    next_seeds.clear();
  }
}

void TesseractDecoder::decode_to_errors_helper(const std::vector<uint64_t>& detections,
                                               size_t detector_order, size_t detector_beam,
                                               const std::vector<uint64_t>& seed_dets,
                                               std::set<uint64_t>& shell_errors,
                                               std::set<uint64_t>& shell_dets) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_dets;

  boost::dynamic_bitset<> initial_dets(num_detectors, false);
  std::vector<DetectorCostTuple> initial_detector_cost_tuples(num_errors);

  for (size_t d : detections) {
    initial_dets[d] = true;
    //   for (int ei : d2e[d]) {
    //     ++initial_detector_cost_tuples[ei].num_dets;
    //   }
  }
  for (size_t d : seed_dets) {
    for (int ei : d2e[d]) {
      ++initial_detector_cost_tuples[ei].num_dets;
    }
  }

  double initial_cost = 0;
  // for (size_t d : detections) {
  for (size_t d : seed_dets) {
    assert(initial_dets[d]);
    initial_cost += get_detcost(d, initial_detector_cost_tuples);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  size_t min_num_dets = detections.size();
  size_t max_num_dets = min_num_dets + detector_beam;

  std::vector<size_t> next_errors;
  boost::dynamic_bitset<> next_dets;
  // std::vector<DetectorCostTuple> next_detector_cost_tuples;

  pq.push({initial_cost, min_num_dets, /*num_fresh_dets=*/0, std::vector<size_t>()});
  size_t num_pq_pushed = 1;

  while (!pq.empty()) {
    const Node node = pq.top();
    pq.pop();
    for (size_t ei : node.errors) {
      shell_errors.insert(ei);
    }

    if (node.num_dets > max_num_dets) continue;

    boost::dynamic_bitset<> dets = initial_dets;
    std::vector<DetectorCostTuple> detector_cost_tuples(num_errors);
    flip_detectors_and_block_errors(detector_order, node.errors, dets, initial_dets, seed_dets,
                                    detector_cost_tuples);

    // Can obviously fold this in
    if (node.num_fresh_dets == 0) {
      bool resolution = true;
      for (uint64_t sd : seed_dets) {
        if (dets[sd]) {
          resolution = false;
          break;
        }
      }
      if (resolution) {
        auto this_dets = initial_dets;
        for (size_t ei : node.errors) {
          for (size_t d : edets[ei]) {
            this_dets[d] ^= 1;
          }
        }
        // Double checks:
        for (size_t d : seed_dets) {
          assert(!this_dets[d]);
        }
        for (size_t d = 0; d < num_detectors; ++d) {
          if (this_dets[d]) {
            assert(initial_dets[d]);
          } else {
            if (initial_dets[d]) {
              shell_dets.insert(d);
            }
          }
        }

        predicted_errors_buffer = node.errors;
        return;
      }
    }

    if (node.num_dets == 0) {
      if (config.create_visualization) {
        visualizer.add_activated_errors(node.errors);
        visualizer.add_activated_detectors(dets, num_detectors);
      }
      if (config.verbose) {
        std::cout << "activated_errors = ";
        for (size_t oei : node.errors) {
          std::cout << oei << ", ";
        }
        std::cout << std::endl;
        std::cout << "activated_detectors = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (dets[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      predicted_errors_buffer = node.errors;
      return;
    }

    if (config.no_revisit_dets && !visited_dets[node.num_dets].insert(dets).second) continue;

    if (config.create_visualization) {
      visualizer.add_activated_errors(node.errors);
      visualizer.add_activated_detectors(dets, num_detectors);
    }
    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq.size() << " num_pq_pushed = " << num_pq_pushed << std::endl;
      std::cout << "num_dets = " << node.num_dets << " num_fresh_dets = " << node.num_fresh_dets
                << " max_num_dets = " << max_num_dets << " cost = " << node.cost << std::endl;
      std::cout << "activated_errors = ";
      for (size_t oei : node.errors) {
        std::cout << oei << ", ";
      }
      std::cout << std::endl;
      std::cout << "activated_detectors = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (dets[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_dets + detector_beam + 1; i <= max_num_dets; ++i) {
          visited_dets[i].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, min_num_dets + detector_beam);
    }

    // for (size_t d = 0; d < num_detectors; ++d) {
    //   if (!dets[d]) continue;
    //   for (int ei : d2e[d]) {
    //     ++detector_cost_tuples[ei].num_dets;
    //   }
    // }

    // next_detector_cost_tuples = detector_cost_tuples;

    size_t min_det = get_min_det(detector_order, dets, initial_dets, seed_dets);

    // size_t prev_ei = std::numeric_limits<size_t>::max();
    // std::vector<double> detector_cost_cache(num_detectors, -1);

    for (int ei : d2e[min_det]) {
      if (detector_cost_tuples[ei].error_blocked) continue;

      // if (prev_ei != std::numeric_limits<size_t>::max()) {
      //   for (int d : edets[prev_ei]) {
      //     int fired = dets[d] ? 1 : -1;
      //     for (int oei : d2e[d]) {
      //       next_detector_cost_tuples[oei].num_dets += fired;
      //     }
      //   }
      // }
      // prev_ei = ei;

      next_errors = node.errors;
      next_errors.push_back(ei);
      next_dets = dets;

      std::vector<DetectorCostTuple> next_detector_cost_tuples(num_errors);
      for (size_t oei = 0; oei < num_errors; ++oei) {
        next_detector_cost_tuples[oei].error_blocked = detector_cost_tuples[oei].error_blocked;
      }
      for (int prev_ei : d2e[min_det]) {
        if (prev_ei == ei) break;
        next_detector_cost_tuples[prev_ei].error_blocked = true;
      }
      next_detector_cost_tuples[ei].error_blocked = 1;

      // double next_cost = node.cost + errors[ei].likelihood_cost;
      double next_cost = 0;
      size_t next_num_dets = node.num_dets;
      size_t next_num_fresh_dets = node.num_fresh_dets;
      for (int d : edets[ei]) {
        next_dets[d] = !next_dets[d];
        int fired = next_dets[d] ? 1 : -1;
        next_num_dets += fired;
        // for (int oei : d2e[d]) {
        //   next_detector_cost_tuples[oei].num_dets += fired;
        // }
        // Update the number of fresh dets
        if (!initial_dets[d]) {
          next_num_fresh_dets += fired;
        }
      }

      if (next_num_dets > max_num_dets) continue;

      if (config.no_revisit_dets &&
          visited_dets[next_num_dets].find(next_dets) != visited_dets[next_num_dets].end())
        continue;

      // for (int d : edets[ei]) {
      //   if (dets[d]) {
      //     if (detector_cost_cache[d] == -1) {
      //       detector_cost_cache[d] = get_detcost(d, detector_cost_tuples);
      //     }
      //     next_cost -= detector_cost_cache[d];
      //   } else {
      //     next_cost += get_detcost(d, next_detector_cost_tuples);
      //   }
      // }

      // for (int od : eneighbors[ei]) {
      //   if (!dets[od] || !next_dets[od]) continue;
      //   if (detector_cost_cache[od] == -1) {
      //     detector_cost_cache[od] = get_detcost(od, detector_cost_tuples);
      //   }
      //   next_cost -= detector_cost_cache[od];
      //   next_cost += get_detcost(od, next_detector_cost_tuples);
      // }

      // Only count up fresh + seed dets
      std::vector<char> next_fresh_and_seed_dets(num_detectors);
      for (size_t d : seed_dets) {
        if (next_dets[d]) {
          // seed det
          next_fresh_and_seed_dets[d] = true;
        }
      }
      for (size_t d = 0; d < num_detectors; ++d) {
        if (next_dets[d] and !initial_dets[d]) {
          // fresh det
          next_fresh_and_seed_dets[d] = true;
        }
      }
      for (size_t d = 0; d < num_detectors; ++d) {
        if (!next_fresh_and_seed_dets[d]) continue;
        for (int oei : d2e[d]) {
          ++next_detector_cost_tuples[oei].num_dets;
        }
      }
      next_cost = cost_from_errors(next_errors);
      for (size_t d = 0; d < num_detectors; ++d) {
        if (next_fresh_and_seed_dets[d]) {
          next_cost += get_detcost(d, next_detector_cost_tuples);
        }
      }
      if (next_cost == INF) continue;

      pq.push({next_cost, next_num_dets, next_num_fresh_dets, next_errors});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        low_confidence_flag = true;
        return;
      }
    }
  }

  if (!pq.empty()) {
    throw std::runtime_error("Priority queue should be empty after decoding failure.");
  }
  if (config.verbose) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = true;
}

double TesseractDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) {
  double total_cost = 0;
  // Iterate over all errors and compute the cost
  for (size_t ei : predicted_errors) {
    total_cost += errors[ei].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> TesseractDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) {
  std::unordered_set<int> flipped_observables_set;

  // Iterate over all errors and compute the mask.
  // We use a set to perform an XOR-like sum.
  // If an observable is already in the set, we remove it (XORing with itself).
  // If it's not, we add it.
  for (size_t ei : predicted_errors) {
    for (int obs_index : errors[ei].symptom.observables) {
      if (flipped_observables_set.count(obs_index)) {
        flipped_observables_set.erase(obs_index);
      } else {
        flipped_observables_set.insert(obs_index);
      }
    }
  }

  // Convert the set to a vector and return it.
  std::vector<int> flipped_observables(flipped_observables_set.begin(),
                                       flipped_observables_set.end());
  // Sort observables
  std::sort(flipped_observables.begin(), flipped_observables.end());
  return flipped_observables;
}

std::vector<int> TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

void TesseractDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                    std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}
