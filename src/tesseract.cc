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
#include <chrono>
#include <functional>  // For std::hash (though not strictly necessary here, but good practice)
#include <iostream>

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
  // Primary sort key: cost (ascending)
  if (cost != other.cost) {
    return cost > other.cost;
  }
  // Secondary sort key: num_dets (descending)
  if (num_dets != other.num_dets) {
    return num_dets < other.num_dets;
  }
  // Tertiary, deterministic tie-breaker: the error path
  return errors > other.errors;
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
      // Primary comparison on cost
      if (error_costs[idx_a].min_cost != error_costs[idx_b].min_cost) {
        return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost;
      }
      // Deterministic tie-breaker on the index itself
      return idx_a < idx_b;
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

size_t find(const std::vector<size_t>& parents, size_t u) {
  size_t v;
  while (true) {
    v = parents[u];
    if (u == v) return v;
    u = v;
  }
}

void do_union(std::vector<size_t>& parents, size_t u, size_t v) {
  u = find(parents, u);
  v = find(parents, v);
  parents[v] = u;
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  // Simple union-find data structure for the merging of seeds.
  std::vector<size_t> parents(detections.size());
  std::iota(parents.begin(), parents.end(), 0);

  // These vectors track the "owner" (which is a root of a seed) of each error and detector
  // footprint. When a search from one seed touches a footprint owned by another, a
  // collision is detected and the seeds are merged.
  std::vector<size_t> error_owner(num_errors, SIZE_MAX);
  std::vector<size_t> det_owner(num_detectors, SIZE_MAX);
  for (size_t si = 0; si < detections.size(); ++si) {
    // Each detection event creates a seed
    det_owner[detections[si]] = si;
  }

  struct SeedDecodeResult {
    std::vector<size_t> predicted_errors;
    bool needs_recomputing = true;
    bool low_confidence_flag = false;
  };
  std::map<size_t, SeedDecodeResult> seed_results;

  while (true) {
    std::map<size_t, std::vector<uint64_t>> root_seeds;
    for (size_t si = 0; si < detections.size(); ++si) {
      size_t root_idx = find(parents, si);
      root_seeds[root_idx].push_back(detections[si]);
    }

    if (config.verbose) {
      std::cout << "CLUSTER: Starting clustering iteration with " << root_seeds.size() << " seeds."
                << std::endl;
    }

    bool collision_in_iteration = false;
    size_t seed_counter = 0;
    for (auto& [si, seed] : root_seeds) {
      if (!seed_results[si].needs_recomputing) {
        ++seed_counter;
        continue;
      }
      if (config.verbose) {
        std::cout << "CLUSTER: Decoding seed " << seed_counter++ << " / " << root_seeds.size()
                  << " (root = " << si
                  << ")"
                     " with "
                  << seed.size() << " detection events." << std::endl;
      }
      predicted_errors_buffer.clear();

      // This is the core of the incremental clustering.
      // `resolve_to_errors_ensemble` will return `true` if its search
      // collided with the footprint of a previously decoded seed.
      bool collision_detected =
          resolve_to_errors_ensemble(detections, seed, si, parents, error_owner, det_owner);

      if (collision_detected) {
        if (config.verbose) {
          std::cout << "CLUSTER: Collision detected for seed " << si << ". Restarting iteration."
                    << std::endl;
        }
        collision_in_iteration = true;
        // A merge occurred inside the resolver. We must break and
        // reconstruct the seeds based on the updated `parents` array.
        break;
      }

      seed_results[si].predicted_errors = predicted_errors_buffer;
      seed_results[si].needs_recomputing = false;
      seed_results[si].low_confidence_flag = low_confidence_flag;
    }

    if (!collision_in_iteration) {
      // No collisions occurred in a full pass over all seeds.
      // The clustering is stable.
      if (config.verbose) {
        std::cout << "CLUSTER: Finished clustering with " << root_seeds.size() << " stable seeds."
                  << std::endl;
      }
      std::vector<size_t> final_errors;
      for (auto& [si, seed] : root_seeds) {
        if (seed_results[si].low_confidence_flag) {
          low_confidence_flag = true;
          predicted_errors_buffer.clear();
          return;
        }
        final_errors.insert(final_errors.end(), seed_results[si].predicted_errors.begin(),
                            seed_results[si].predicted_errors.end());
      }
      predicted_errors_buffer = final_errors;
      return;
    }

    // A collision occurred, try again
  }
}

bool TesseractDecoder::resolve_to_errors_ensemble(const std::vector<uint64_t>& detections,
                                                  const std::vector<uint64_t>& seed_dets,
                                                  size_t seed_id, std::vector<size_t>& parents,
                                                  std::vector<size_t>& error_owner,
                                                  std::vector<size_t>& det_owner) {
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
      if (resolve_to_errors(detections, detector_order, beam, seed_dets, seed_id, parents,
                            error_owner, det_owner)) {
        return true;  // Collision detected, propagate signal up.
      }

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
      if (resolve_to_errors(detections, detector_order, config.det_beam, seed_dets, seed_id,
                            parents, error_owner, det_owner)) {
        return true;  // Collision detected, propagate signal up.
      }
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
  return false;  // No collision detected in any trial.
}

size_t TesseractDecoder::get_min_det(size_t detector_order, const boost::dynamic_bitset<>& dets,
                                     const boost::dynamic_bitset<>& initial_dets,
                                     const std::vector<uint64_t>& seed_dets) const {
  // This must only return dets in the seed dets or fresh dets
  for (size_t d = 0; d < num_detectors; ++d) {
    size_t dod = config.det_orders[detector_order][d];
    if (dets[dod] and !initial_dets[dod]) {
      // If this is a fresh det
      return dod;
    }
    if (dets[dod] and std::find(seed_dets.begin(), seed_dets.end(), dod) != seed_dets.end()) {
      // If this is a seed det
      return dod;
    }
  }
  return std::numeric_limits<size_t>::max();
}

void TesseractDecoder::flip_detectors_and_block_errors(
    size_t detector_order, const std::vector<size_t>& errors, boost::dynamic_bitset<>& dets,
    const boost::dynamic_bitset<>& initial_dets, const std::vector<uint64_t>& seed_dets,
    const std::vector<bool>& seed_dets_bools,
    std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  for (size_t ei : errors) {
    size_t min_det = get_min_det(detector_order, dets, initial_dets, seed_dets);

    for (size_t oei : d2e[min_det]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (oei == ei) break;
    }

    for (int d : edets[ei]) {
      dets[d] = !dets[d];
      int fired = dets[d] ? 1 : -1;
      // Incrementally update the detector cost tuples
      if (!initial_dets[d] or seed_dets_bools[d]) {
        for (size_t oei : d2e[d]) {
          detector_cost_tuples[oei].num_dets += fired;
        }
      }
    }
  }
}

bool TesseractDecoder::resolve_to_errors(const std::vector<uint64_t>& detections,
                                         size_t detector_order, size_t detector_beam,
                                         const std::vector<uint64_t>& seed_dets, size_t seed_id,
                                         std::vector<size_t>& parents,
                                         std::vector<size_t>& error_owner,
                                         std::vector<size_t>& det_owner) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_dets;

  boost::dynamic_bitset<> initial_dets(num_detectors, false);
  std::vector<DetectorCostTuple> initial_detector_cost_tuples(num_errors);

  for (size_t d : detections) {
    initial_dets[d] = true;
  }
  std::vector<bool> seed_dets_bools(num_detectors);
  for (size_t d : seed_dets) {
    for (int ei : d2e[d]) {
      ++initial_detector_cost_tuples[ei].num_dets;
    }
    seed_dets_bools[d] = true;
  }

  double initial_cost = 0;
  for (size_t d : seed_dets) {
    assert(initial_dets[d]);
    initial_cost += get_detcost(d, initial_detector_cost_tuples);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return false;
  }

  size_t min_num_dets = detections.size();
  size_t max_num_dets = min_num_dets + detector_beam;

  std::vector<size_t> next_errors;
  boost::dynamic_bitset<> next_dets;

  pq.push({initial_cost, min_num_dets, /*num_fresh_dets=*/0, std::vector<size_t>()});
  size_t num_pq_pushed = 1;

  while (!pq.empty()) {
    const Node node = pq.top();
    pq.pop();

    // Incremental clustering: check for error footprint collision.
    // As the search expands, each error in the current path is checked.
    // If the error is already "owned" by a different seed, we have a
    // collision. We merge the two seeds and return true to signal that the
    // clustering state has changed.
    size_t current_root = find(parents, seed_id);
    for (size_t ei : node.errors) {
      size_t owner_root = error_owner[ei];
      if (owner_root == SIZE_MAX) {
        error_owner[ei] = current_root;
      } else if (find(parents, owner_root) != current_root) {
        // TODO: simplify by making do_union return a bool
        do_union(parents, current_root, owner_root);
        return true;  // Collision detected and merged.
      }
    }

    if (node.num_dets > max_num_dets) continue;

    boost::dynamic_bitset<> dets = initial_dets;
    std::vector<DetectorCostTuple> detector_cost_tuples = initial_detector_cost_tuples;
    flip_detectors_and_block_errors(detector_order, node.errors, dets, initial_dets, seed_dets,
                                    seed_dets_bools, detector_cost_tuples);

    if (node.num_fresh_dets == 0) {
      bool resolution = true;
      for (uint64_t sd : seed_dets) {
        if (dets[sd]) {
          resolution = false;
          break;
        }
      }
      if (resolution) {
        // Incremental clustering: check for detector footprint collision.
        // When a valid resolution is found, we check the detectors that
        // were part of the search shell (i.e., originally on but now off).
        // If any of these detectors are owned by another seed, we merge
        // and signal a collision.
        auto this_dets = initial_dets;
        for (size_t ei : node.errors) {
          for (size_t d : edets[ei]) {
            this_dets[d] ^= 1;
          }
        }
        for (size_t d = 0; d < num_detectors; ++d) {
          if (!this_dets[d] && initial_dets[d]) {
            size_t owner_root = det_owner[d];
            if (owner_root == SIZE_MAX) {
              det_owner[d] = current_root;
              assert(false && "unreachable");
            } else if (find(parents, owner_root) != current_root) {
              do_union(parents, current_root, owner_root);
              return true;  // Collision detected and merged.
            }
          }
        }

        predicted_errors_buffer = node.errors;
        return false;  // Successful resolution, no collision.
      }
    }

    if (node.num_dets == 0) {
      predicted_errors_buffer = node.errors;
      return false;  // Successful resolution, no collision.
    }

    if (config.no_revisit_dets && !visited_dets[node.num_dets].insert(dets).second) continue;

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_dets + detector_beam + 1; i <= max_num_dets; ++i) {
          visited_dets[i].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, min_num_dets + detector_beam);
    }

    size_t min_det = get_min_det(detector_order, dets, initial_dets, seed_dets);

    size_t prev_ei = std::numeric_limits<size_t>::max();
    std::vector<double> detcost_cache(num_detectors, -1);

    // We incrementally maintain the correct detector cost tuples
    std::vector<DetectorCostTuple> next_detector_cost_tuples = detector_cost_tuples;
    for (int ei : d2e[min_det]) {
      if (detector_cost_tuples[ei].error_blocked) continue;

      // Undo previous updates to next_detector_cost_tuples
      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          if (!initial_dets[d] or seed_dets_bools[d]) {
            int fired = dets[d] ? 1 : -1;
            for (size_t oei : d2e[d]) {
              next_detector_cost_tuples[oei].num_dets += fired;
            }
          }
        }
      }
      prev_ei = ei;

      next_errors = node.errors;
      next_errors.push_back(ei);
      next_dets = dets;

      next_detector_cost_tuples[ei].error_blocked = 1;

      double next_cost = 0;
      size_t next_num_dets = node.num_dets;
      size_t next_num_fresh_dets = node.num_fresh_dets;
      for (int d : edets[ei]) {
        next_dets[d] = !next_dets[d];
        int fired = next_dets[d] ? 1 : -1;
        next_num_dets += fired;
        if (!initial_dets[d]) {
          // TODO: make num fresh dets include the seed dets to simplify the logic
          next_num_fresh_dets += fired;
        }
        // Incrementally update the detector cost tuples
        if (!initial_dets[d] or seed_dets_bools[d]) {
          for (size_t oei : d2e[d]) {
            next_detector_cost_tuples[oei].num_dets += fired;
          }
        }
      }

      if (next_num_dets > max_num_dets) continue;

      if (config.no_revisit_dets &&
          visited_dets[next_num_dets].find(next_dets) != visited_dets[next_num_dets].end())
        continue;

      // next_cost = cost_from_errors(next_errors);
      next_cost = node.cost + errors[ei].likelihood_cost;

      for (size_t d : edets[ei]) {
        if (!initial_dets[d] or seed_dets_bools[d]) {
          // This detector must eventually be turned off as it is a fresh or seed det, so we include
          // the cost lower bound for doing so in the total detcost (A* penalty)
          if (dets[d]) {
            // This detector was on and is now turning off, so we need to subtract its old detcost
            if (detcost_cache[d] == -1) {
              detcost_cache[d] = get_detcost(d, detector_cost_tuples);
            }
            next_cost -= detcost_cache[d];
          } else {
            // This detector was off and is now turning on, so we need to add the new detcost
            next_cost += get_detcost(d, next_detector_cost_tuples);
          }
        }
      }

      for (int od : eneighbors[ei]) {
        // We have to update the detcost contribution from extant fresh and seed dets in the
        // neighborhood If this detector is off or flipped, it is irrelevant if (!dets[od] ||
        // !next_dets[od]) continue;
        if (!dets[od]) continue;
        assert(dets[od] == next_dets[od]);  // edets[ei] is supposedly removed from eneighbors[ei]
        if (!initial_dets[od] or seed_dets_bools[od]) {
          // This is an extant fresh or seed det
          if (detcost_cache[od] == -1) {
            detcost_cache[od] = get_detcost(od, detector_cost_tuples);
          }
          next_cost -= detcost_cache[od];
          next_cost += get_detcost(od, next_detector_cost_tuples);
        }
      }

      // for (size_t d = 0; d < num_detectors; ++d) {
      //   if (next_dets[d] and (!initial_dets[d] or seed_dets_bools[d])) {
      //     next_cost += get_detcost(d, next_detector_cost_tuples);
      //   }
      // }

      if (next_cost == INF) continue;

      pq.push({next_cost, next_num_dets, next_num_fresh_dets, next_errors});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        if (config.verbose) {
          std::cout << "setting low confidence flag" << std::endl;
        }
        low_confidence_flag = true;
        return false;
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
  return false;  // Decoding failed, but no collision to report.
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
