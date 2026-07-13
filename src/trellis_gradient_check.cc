#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tesseract_trellis.h"

namespace {

struct SyndromeCounts {
  uint64_t syndrome;
  uint64_t label_zero_count;
  uint64_t label_one_count;
};

struct CoordinateCheck {
  size_t error_index;
  double analytic;
  double finite_difference;
};

enum class CheckObjective {
  Observable,
  SyndromeCensored,
};

uint64_t parse_uint64_strict(const std::string& token, const std::string& context) {
  if (token.empty() ||
      std::any_of(token.begin(), token.end(), [](char c) { return c < '0' || c > '9'; })) {
    throw std::invalid_argument("invalid unsigned integer " + token + " in " + context);
  }
  size_t parsed = 0;
  try {
    const uint64_t value = std::stoull(token, &parsed);
    if (parsed != token.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid unsigned integer " + token + " in " + context);
  }
}

double parse_double_strict(const std::string& token, const std::string& context) {
  size_t parsed = 0;
  try {
    const double value = std::stod(token, &parsed);
    if (parsed != token.size() || !std::isfinite(value)) {
      throw std::invalid_argument("invalid floating-point value");
    }
    return value;
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid floating-point value " + token + " in " + context);
  }
}

TesseractTrellisRankingMode parse_ranking_mode(const std::string& value) {
  if (value == "mass") {
    return TesseractTrellisRankingMode::MassOnly;
  }
  if (value == "future-detcost") {
    return TesseractTrellisRankingMode::FutureDetcostRanked;
  }
  if (value == "future-active-detcost") {
    return TesseractTrellisRankingMode::FutureActiveDetcostRanked;
  }
  throw std::invalid_argument("invalid trellis ranking mode: " + value);
}

stim::DetectorErrorModel read_dem(const std::string& path) {
  FILE* file = fopen(path.c_str(), "r");
  if (file == nullptr) {
    throw std::invalid_argument("could not open DEM: " + path);
  }
  stim::DetectorErrorModel dem = stim::DetectorErrorModel::from_file(file);
  fclose(file);
  return dem;
}

std::vector<SyndromeCounts> read_counts(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("could not open syndrome counts: " + path);
  }
  std::vector<SyndromeCounts> rows;
  std::unordered_set<uint64_t> seen_syndromes;
  std::string syndrome_token;
  while (input >> syndrome_token) {
    std::string zero_token;
    std::string one_token;
    if (!(input >> zero_token >> one_token)) {
      throw std::invalid_argument("malformed syndrome-count row in: " + path);
    }
    const uint64_t syndrome = parse_uint64_strict(syndrome_token, "syndrome-count row");
    const uint64_t zero = parse_uint64_strict(zero_token, "syndrome-count row");
    const uint64_t one = parse_uint64_strict(one_token, "syndrome-count row");
    if (!seen_syndromes.insert(syndrome).second) {
      throw std::invalid_argument("duplicate syndrome-count row: " + syndrome_token);
    }
    rows.push_back({syndrome, zero, one});
  }
  if (!input.eof()) {
    throw std::invalid_argument("malformed syndrome-count row in: " + path);
  }
  if (rows.empty()) {
    throw std::invalid_argument("no syndrome-count rows in: " + path);
  }
  return rows;
}

std::vector<uint64_t> syndrome_hits(uint64_t syndrome) {
  std::vector<uint64_t> hits;
  while (syndrome != 0) {
    const uint64_t bit = static_cast<uint64_t>(__builtin_ctzll(syndrome));
    hits.push_back(bit);
    syndrome &= syndrome - 1;
  }
  return hits;
}

uint64_t checked_count_sum(uint64_t a, uint64_t b, const std::string& context) {
  if (a > std::numeric_limits<uint64_t>::max() - b) {
    throw std::invalid_argument("syndrome counts overflow uint64 in " + context);
  }
  return a + b;
}

uint64_t validate_rows_against_dem(const std::vector<SyndromeCounts>& rows,
                                   const stim::DetectorErrorModel& dem) {
  const size_t num_detectors = dem.count_detectors();
  uint64_t total_shots = 0;
  for (const auto& row : rows) {
    for (uint64_t detector : syndrome_hits(row.syndrome)) {
      if (detector >= num_detectors) {
        throw std::invalid_argument("syndrome detector D" + std::to_string(detector) +
                                    " is outside the DEM detector range");
      }
    }
    const uint64_t row_shots =
        checked_count_sum(row.label_zero_count, row.label_one_count, "syndrome row");
    total_shots = checked_count_sum(total_shots, row_shots, "selected population");
  }
  return total_shots;
}

double shifted_probability(double probability, double logit_shift) {
  const double logit = std::log(probability / (1.0 - probability));
  return 1.0 / (1.0 + std::exp(-(logit + logit_shift)));
}

stim::DetectorErrorModel perturb_error_logit(const stim::DetectorErrorModel& dem,
                                             size_t target_error_index, double logit_shift) {
  stim::DetectorErrorModel result;
  size_t error_index = 0;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      double probability = instruction.arg_data[0];
      if (error_index == target_error_index) {
        probability = shifted_probability(probability, logit_shift);
      }
      result.append_error_instruction(probability, instruction.target_data, instruction.tag);
      ++error_index;
    } else {
      result.append_dem_instruction(instruction);
    }
  }
  if (target_error_index >= error_index) {
    throw std::out_of_range("error index is outside the preprocessed DEM");
  }
  return result;
}

double objective_nll(const stim::DetectorErrorModel& dem, const std::vector<SyndromeCounts>& rows,
                     size_t beam_width, TesseractTrellisRankingMode ranking_mode,
                     double future_detcost_scale, CheckObjective objective,
                     uint64_t population_shots) {
  TesseractTrellisConfig config;
  config.dem = dem;
  config.beam_width = beam_width;
  config.merge_errors = false;
  config.ranking_mode = ranking_mode;
  config.future_detcost_scale = future_detcost_scale;
  TesseractTrellisDecoder decoder(config);
  long double loss = 0.0;
  long double selected_mass = 0.0;
  uint64_t selected_shots = 0;
  for (const auto& row : rows) {
    decoder.decode_shot(syndrome_hits(row.syndrome));
    const uint64_t row_shots = row.label_zero_count + row.label_one_count;
    if (objective == CheckObjective::Observable) {
      const double probability = decoder.observable_probability();
      if (decoder.low_confidence_flag || !std::isfinite(probability) || probability <= 0.0 ||
          probability >= 1.0) {
        throw std::runtime_error("non-finite observable probability in finite-difference pass");
      }
      loss -= row.label_one_count * std::log(probability);
      loss -= row.label_zero_count * std::log1p(-probability);
    } else {
      const double log_probability = decoder.syndrome_log_probability;
      if (decoder.low_confidence_flag || !std::isfinite(log_probability) || log_probability > 0.0) {
        throw std::runtime_error("non-finite syndrome probability in finite-difference pass");
      }
      loss -= row_shots * log_probability;
      selected_mass += std::exp(log_probability);
    }
    selected_shots += row_shots;
  }
  if (objective == CheckObjective::SyndromeCensored) {
    if (population_shots < selected_shots || selected_mass <= 0.0 || selected_mass >= 1.0) {
      throw std::runtime_error("invalid censored population or selected syndrome mass");
    }
    loss -= (population_shots - selected_shots) * std::log1pl(-selected_mass);
  } else {
    population_shots = selected_shots;
  }
  if (population_shots == 0) {
    throw std::runtime_error("finite-difference objective contains no shots");
  }
  const double result = static_cast<double>(loss / population_shots);
  if (!std::isfinite(result)) {
    throw std::runtime_error("non-finite finite-difference objective");
  }
  return result;
}

void print_json(size_t beam_width, size_t error_count, uint64_t selected_shots,
                uint64_t population_shots, double baseline_nll, double epsilon,
                const std::string& objective, const std::string& ranking_mode,
                double future_detcost_scale, const std::vector<CoordinateCheck>& checks) {
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  for (const auto& check : checks) {
    const double absolute_error = std::abs(check.analytic - check.finite_difference);
    const double scale = std::max({std::abs(check.analytic), std::abs(check.finite_difference),
                                   std::numeric_limits<double>::epsilon()});
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, absolute_error / scale);
  }

  std::cout << std::setprecision(17);
  std::cout << "{\n"
            << "  \"schema_version\": 3,\n"
            << "  \"objective\": \"" << objective << "\",\n"
            << "  \"beam_width\": " << beam_width << ",\n"
            << "  \"preprocessed_errors\": " << error_count << ",\n"
            << "  \"selected_shots\": " << selected_shots << ",\n"
            << "  \"population_shots\": " << population_shots << ",\n"
            << "  \"baseline_nll\": " << baseline_nll << ",\n"
            << "  \"finite_difference_epsilon\": " << epsilon << ",\n"
            << "  \"ranking_mode\": \"" << ranking_mode << "\",\n"
            << "  \"future_detcost_scale\": " << future_detcost_scale << ",\n"
            << "  \"max_absolute_error\": " << max_absolute_error << ",\n"
            << "  \"max_relative_error\": " << max_relative_error << ",\n"
            << "  \"checks\": [\n";
  for (size_t i = 0; i < checks.size(); ++i) {
    const auto& check = checks[i];
    const double absolute_error = std::abs(check.analytic - check.finite_difference);
    const double scale = std::max({std::abs(check.analytic), std::abs(check.finite_difference),
                                   std::numeric_limits<double>::epsilon()});
    std::cout << "    {\"error_index\": " << check.error_index
              << ", \"analytic\": " << check.analytic
              << ", \"finite_difference\": " << check.finite_difference
              << ", \"absolute_error\": " << absolute_error
              << ", \"relative_error\": " << absolute_error / scale << "}";
    std::cout << (i + 1 == checks.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 8 && argc != 10) {
    std::cerr << "usage: trellis_gradient_check DEM COUNTS BEAM CHECKS EPSILON "
                 "[RANKING_MODE FUTURE_DETCOST_SCALE "
                 "[syndrome_censored POPULATION_SHOTS]]\n";
    return EXIT_FAILURE;
  }
  try {
    const std::string dem_path = argv[1];
    const std::string counts_path = argv[2];
    const size_t beam_width = parse_uint64_strict(argv[3], "BEAM");
    const size_t check_count = parse_uint64_strict(argv[4], "CHECKS");
    const double epsilon = parse_double_strict(argv[5], "EPSILON");
    const bool has_ranking = argc >= 8;
    const std::string ranking_mode_name = has_ranking ? argv[6] : "mass";
    const double future_detcost_scale =
        has_ranking ? parse_double_strict(argv[7], "FUTURE_DETCOST_SCALE") : 2.0;
    const std::string objective_name = argc == 10 ? argv[8] : "observable";
    const CheckObjective objective =
        objective_name == "observable" ? CheckObjective::Observable
        : objective_name == "syndrome_censored"
            ? CheckObjective::SyndromeCensored
            : throw std::invalid_argument(
                  "gradient-check objective must be observable or syndrome_censored");
    uint64_t population_shots = argc == 10 ? parse_uint64_strict(argv[9], "POPULATION_SHOTS") : 0;
    const auto ranking_mode = parse_ranking_mode(ranking_mode_name);
    if (beam_width == 0 || check_count == 0 || !std::isfinite(epsilon) || epsilon <= 0.0 ||
        !std::isfinite(future_detcost_scale) || future_detcost_scale < 0.0 ||
        (objective == CheckObjective::Observable && argc == 10) ||
        (objective == CheckObjective::SyndromeCensored && population_shots == 0)) {
      throw std::invalid_argument("BEAM, CHECKS, and EPSILON must be positive");
    }

    const auto rows = read_counts(counts_path);
    const stim::DetectorErrorModel source_dem = read_dem(dem_path);
    if (source_dem.count_observables() != 1) {
      throw std::invalid_argument("gradient checker requires a DEM with exactly one observable");
    }
    if (source_dem.count_errors() == 0) {
      throw std::invalid_argument("gradient checker requires at least one DEM error mechanism");
    }
    if (source_dem.count_detectors() > 64) {
      throw std::invalid_argument(
          "gradient checker syndrome integers support at most 64 detectors");
    }
    for (const auto& instruction : source_dem.flattened().instructions) {
      if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
        const double probability = instruction.arg_data[0];
        if (!std::isfinite(probability) || probability <= 0.0 || probability >= 1.0) {
          throw std::invalid_argument(
              "gradient checker requires error probabilities strictly between 0 and 1");
        }
      }
    }
    const uint64_t expected_selected_shots = validate_rows_against_dem(rows, source_dem);
    TesseractTrellisConfig config;
    config.dem = source_dem;
    config.beam_width = beam_width;
    config.merge_errors = false;
    config.ranking_mode = ranking_mode;
    config.future_detcost_scale = future_detcost_scale;
    TesseractTrellisDecoder decoder(config);
    const stim::DetectorErrorModel preprocessed_dem = decoder.config.dem;
    std::vector<double> gradient(decoder.errors.size(), 0.0);
    std::vector<double> mass_gradient(decoder.errors.size(), 0.0);
    long double loss = 0.0;
    long double selected_mass = 0.0;
    uint64_t selected_shots = 0;

    for (const auto& row : rows) {
      const uint64_t row_shots = row.label_zero_count + row.label_one_count;
      if (objective == CheckObjective::Observable) {
        const auto logit_gradient =
            decoder.decode_shot_with_observable_logit_gradient(syndrome_hits(row.syndrome));
        const double probability = decoder.observable_probability();
        if (decoder.low_confidence_flag || !std::isfinite(probability) || probability <= 0.0 ||
            probability >= 1.0 || logit_gradient.size() != gradient.size()) {
          throw std::runtime_error("invalid analytic observable-gradient result");
        }
        loss -= row.label_one_count * std::log(probability);
        loss -= row.label_zero_count * std::log1p(-probability);
        const double logit_loss_gradient =
            row_shots * probability - static_cast<double>(row.label_one_count);
        for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
          gradient[error_index] += logit_loss_gradient * logit_gradient[error_index];
        }
      } else {
        const auto log_probability_gradient =
            decoder.decode_shot_with_syndrome_log_probability_gradient(syndrome_hits(row.syndrome));
        const double log_probability = decoder.syndrome_log_probability;
        if (decoder.low_confidence_flag || !std::isfinite(log_probability) ||
            log_probability > 0.0 || log_probability_gradient.size() != gradient.size()) {
          throw std::runtime_error("invalid analytic syndrome-gradient result");
        }
        const double probability = std::exp(log_probability);
        loss -= row_shots * log_probability;
        selected_mass += probability;
        for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
          gradient[error_index] -= row_shots * log_probability_gradient[error_index];
          mass_gradient[error_index] += probability * log_probability_gradient[error_index];
        }
      }
      selected_shots += row_shots;
    }
    if (selected_shots != expected_selected_shots) {
      throw std::runtime_error("internal selected-shot accounting mismatch");
    }
    if (objective == CheckObjective::SyndromeCensored) {
      if (population_shots < selected_shots || selected_mass <= 0.0 || selected_mass >= 1.0) {
        throw std::runtime_error("invalid censored population or selected syndrome mass");
      }
      const uint64_t other_shots = population_shots - selected_shots;
      loss -= other_shots * std::log1pl(-selected_mass);
      const double other_gradient_scale = other_shots / (1.0 - selected_mass);
      for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
        gradient[error_index] += other_gradient_scale * mass_gradient[error_index];
      }
    } else {
      population_shots = selected_shots;
    }
    if (population_shots == 0) {
      throw std::runtime_error("analytic objective contains no shots");
    }
    for (double& value : gradient) {
      value /= population_shots;
      if (!std::isfinite(value)) {
        throw std::runtime_error("non-finite analytic gradient");
      }
    }
    const double baseline_nll = static_cast<double>(loss / population_shots);
    if (!std::isfinite(baseline_nll)) {
      throw std::runtime_error("non-finite analytic objective");
    }

    std::vector<size_t> ranked_indices(gradient.size());
    for (size_t i = 0; i < ranked_indices.size(); ++i) {
      ranked_indices[i] = i;
    }
    std::sort(ranked_indices.begin(), ranked_indices.end(),
              [&](size_t a, size_t b) { return std::abs(gradient[a]) > std::abs(gradient[b]); });
    ranked_indices.resize(std::min(check_count, ranked_indices.size()));

    std::vector<CoordinateCheck> checks;
    checks.reserve(ranked_indices.size());
    for (size_t error_index : ranked_indices) {
      const auto plus_dem = perturb_error_logit(preprocessed_dem, error_index, epsilon);
      const auto minus_dem = perturb_error_logit(preprocessed_dem, error_index, -epsilon);
      const double finite_difference =
          (objective_nll(plus_dem, rows, beam_width, ranking_mode, future_detcost_scale, objective,
                         population_shots) -
           objective_nll(minus_dem, rows, beam_width, ranking_mode, future_detcost_scale, objective,
                         population_shots)) /
          (2.0 * epsilon);
      if (!std::isfinite(finite_difference)) {
        throw std::runtime_error("non-finite finite-difference gradient");
      }
      checks.push_back({error_index, gradient[error_index], finite_difference});
    }
    print_json(beam_width, gradient.size(), selected_shots, population_shots, baseline_nll, epsilon,
               objective_name, ranking_mode_name, future_detcost_scale, checks);
  } catch (const std::exception& ex) {
    std::cerr << "trellis gradient validation failed: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
