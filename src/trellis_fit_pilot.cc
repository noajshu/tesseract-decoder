#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tesseract_trellis.h"

namespace {

constexpr double kProbabilityFloor = 1e-6;

struct SyndromeCounts {
  std::string syndrome;
  std::vector<uint64_t> detections;
  uint64_t train_zero_count;
  uint64_t train_one_count;
  uint64_t test_zero_count;
  uint64_t test_one_count;
};

enum class FitObjective {
  Observable,
  Syndrome,
  SyndromeCensored,
};

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

struct ObjectiveResult {
  double data_nll;
  std::vector<double> gradient;
  uint64_t shots;
  double elapsed_seconds;
};

struct HistoryEntry {
  size_t step;
  double data_nll;
  double regularization;
  double objective;
  double gradient_norm;
  double accepted_step_size;
  double elapsed_seconds;
};

struct CountTotals {
  uint64_t train;
  uint64_t test;
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
  std::unordered_set<std::string> seen_syndromes;
  std::string syndrome;
  while (input >> syndrome) {
    std::string train_zero_token;
    std::string train_one_token;
    std::string test_zero_token;
    std::string test_one_token;
    if (!(input >> train_zero_token >> train_one_token >> test_zero_token >> test_one_token)) {
      throw std::invalid_argument("malformed syndrome-count row in: " + path);
    }
    const uint64_t train_zero_count = parse_uint64_strict(train_zero_token, "syndrome-count row");
    const uint64_t train_one_count = parse_uint64_strict(train_one_token, "syndrome-count row");
    const uint64_t test_zero_count = parse_uint64_strict(test_zero_token, "syndrome-count row");
    const uint64_t test_one_count = parse_uint64_strict(test_one_token, "syndrome-count row");
    std::vector<uint64_t> detections;
    if (syndrome.rfind("dets:", 0) == 0) {
      const std::string encoded = syndrome.substr(5);
      if (encoded.empty() || encoded.back() == ',') {
        throw std::invalid_argument("invalid detector list in syndrome-count row: " + syndrome);
      }
      if (encoded != "-") {
        size_t begin = 0;
        while (begin < encoded.size()) {
          const size_t end = encoded.find(',', begin);
          const std::string token = encoded.substr(begin, end - begin);
          const uint64_t detector = parse_uint64_strict(token, "detector list");
          if (!detections.empty() && detector <= detections.back()) {
            throw std::invalid_argument("invalid detector list in syndrome-count row: " + syndrome);
          }
          detections.push_back(detector);
          if (end == std::string::npos) {
            break;
          }
          begin = end + 1;
        }
      }
    } else {
      uint64_t mask = parse_uint64_strict(syndrome, "syndrome-count row");
      while (mask != 0) {
        const uint64_t bit = static_cast<uint64_t>(__builtin_ctzll(mask));
        detections.push_back(bit);
        mask &= mask - 1;
      }
    }
    std::string canonical_syndrome = "dets:";
    if (detections.empty()) {
      canonical_syndrome += '-';
    } else {
      for (size_t k = 0; k < detections.size(); ++k) {
        if (k != 0) {
          canonical_syndrome += ',';
        }
        canonical_syndrome += std::to_string(detections[k]);
      }
    }
    if (!seen_syndromes.insert(canonical_syndrome).second) {
      throw std::invalid_argument("duplicate syndrome-count row: " + syndrome);
    }
    rows.push_back({syndrome, std::move(detections), train_zero_count, train_one_count,
                    test_zero_count, test_one_count});
  }
  if (!input.eof()) {
    throw std::invalid_argument("malformed syndrome-count row in: " + path);
  }
  if (rows.empty()) {
    throw std::invalid_argument("no syndrome-count rows in: " + path);
  }
  return rows;
}

uint64_t checked_count_sum(uint64_t a, uint64_t b, const std::string& context) {
  if (a > std::numeric_limits<uint64_t>::max() - b) {
    throw std::invalid_argument("syndrome counts overflow uint64 in " + context);
  }
  return a + b;
}

CountTotals validate_detections_against_dem(const std::vector<SyndromeCounts>& rows,
                                            const stim::DetectorErrorModel& dem) {
  const size_t num_detectors = dem.count_detectors();
  CountTotals totals{0, 0};
  for (const auto& row : rows) {
    for (uint64_t detector : row.detections) {
      if (detector >= num_detectors) {
        throw std::invalid_argument("detector D" + std::to_string(detector) +
                                    " is outside the DEM detector range in row " + row.syndrome);
      }
    }
    const uint64_t train =
        checked_count_sum(row.train_zero_count, row.train_one_count, "row " + row.syndrome);
    const uint64_t test =
        checked_count_sum(row.test_zero_count, row.test_one_count, "row " + row.syndrome);
    totals.train = checked_count_sum(totals.train, train, "training population");
    totals.test = checked_count_sum(totals.test, test, "held-out population");
  }
  return totals;
}

double probability_from_logit(double logit) {
  if (logit >= 0.0) {
    return 1.0 / (1.0 + std::exp(-logit));
  }
  const double exp_logit = std::exp(logit);
  return exp_logit / (1.0 + exp_logit);
}

std::vector<double> error_logits(const stim::DetectorErrorModel& dem) {
  std::vector<double> logits;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      continue;
    }
    const double probability = instruction.arg_data[0];
    if (!std::isfinite(probability) || probability <= 0.0 || probability >= 1.0) {
      throw std::invalid_argument("all fitted DEM probabilities must lie strictly between 0 and 1");
    }
    logits.push_back(std::log(probability / (1.0 - probability)));
  }
  return logits;
}

stim::DetectorErrorModel dem_with_probability_floor(const stim::DetectorErrorModel& dem) {
  stim::DetectorErrorModel result;
  std::unordered_set<std::string> seen_symptoms;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      const double probability = instruction.arg_data[0];
      if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("all fitted DEM probabilities must lie between 0 and 1");
      }
      common::Error error(instruction);
      const std::string symptom = error.symptom.str();
      if (!seen_symptoms.insert(symptom).second) {
        throw std::invalid_argument(
            "fitted DEM contains indistinguishable error mechanisms; merge them before fitting");
      }
      result.append_error_instruction(
          std::clamp(probability, kProbabilityFloor, 1.0 - kProbabilityFloor),
          instruction.target_data, instruction.tag);
    } else {
      result.append_dem_instruction(instruction);
    }
  }
  return result;
}

stim::DetectorErrorModel dem_with_logits(const stim::DetectorErrorModel& dem,
                                         const std::vector<double>& logits) {
  stim::DetectorErrorModel result;
  size_t error_index = 0;
  for (const auto& instruction : dem.flattened().instructions) {
    if (instruction.type == stim::DemInstructionType::DEM_ERROR) {
      if (error_index >= logits.size()) {
        throw std::invalid_argument("too few fitted logits for DEM");
      }
      result.append_error_instruction(probability_from_logit(logits[error_index]),
                                      instruction.target_data, instruction.tag);
      ++error_index;
    } else {
      result.append_dem_instruction(instruction);
    }
  }
  if (error_index != logits.size()) {
    throw std::invalid_argument("too many fitted logits for DEM");
  }
  return result;
}

ObjectiveResult objective_and_gradient(const stim::DetectorErrorModel& dem,
                                       const std::vector<SyndromeCounts>& rows, size_t beam_width,
                                       size_t num_threads, FitObjective fit_objective,
                                       uint64_t censored_total_shots,
                                       TesseractTrellisRankingMode ranking_mode,
                                       double future_detcost_scale) {
  const auto started = std::chrono::steady_clock::now();
  const size_t num_errors = dem.count_errors();
  std::vector<std::vector<double>> partial_gradients(num_threads,
                                                     std::vector<double>(num_errors, 0.0));
  std::vector<long double> partial_losses(num_threads, 0.0);
  std::vector<long double> partial_selected_masses(num_threads, 0.0);
  std::vector<uint64_t> partial_shots(num_threads, 0);
  std::vector<std::vector<double>> partial_mass_gradients(num_threads,
                                                          std::vector<double>(num_errors, 0.0));
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    workers.emplace_back([&, thread_index]() {
      try {
        TesseractTrellisConfig config;
        config.dem = dem;
        config.beam_width = beam_width;
        config.merge_errors = false;
        config.ranking_mode = ranking_mode;
        config.future_detcost_scale = future_detcost_scale;
        TesseractTrellisDecoder decoder(config);
        auto& gradient = partial_gradients[thread_index];
        for (size_t row_index = thread_index; row_index < rows.size(); row_index += num_threads) {
          const auto& row = rows[row_index];
          const uint64_t row_shots = row.train_zero_count + row.train_one_count;
          if (row_shots == 0 && fit_objective != FitObjective::SyndromeCensored) {
            continue;
          }
          if (fit_objective == FitObjective::Observable) {
            const auto logit_gradient =
                decoder.decode_shot_with_observable_logit_gradient(row.detections);
            const double probability = decoder.observable_probability();
            if (decoder.low_confidence_flag || !std::isfinite(probability) || probability <= 0.0 ||
                probability >= 1.0 || logit_gradient.size() != gradient.size()) {
              throw std::runtime_error("invalid observable gradient while fitting");
            }
            partial_losses[thread_index] -= row.train_one_count * std::log(probability);
            partial_losses[thread_index] -= row.train_zero_count * std::log1p(-probability);
            const double logit_loss_gradient =
                row_shots * probability - static_cast<double>(row.train_one_count);
            for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
              gradient[error_index] += logit_loss_gradient * logit_gradient[error_index];
            }
          } else {
            const auto log_probability_gradient =
                decoder.decode_shot_with_syndrome_log_probability_gradient(row.detections);
            if (decoder.low_confidence_flag || !std::isfinite(decoder.syndrome_log_probability) ||
                decoder.syndrome_log_probability > 0.0 ||
                log_probability_gradient.size() != gradient.size()) {
              throw std::runtime_error("invalid syndrome gradient while fitting");
            }
            partial_losses[thread_index] -= row_shots * decoder.syndrome_log_probability;
            for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
              gradient[error_index] -= row_shots * log_probability_gradient[error_index];
            }
            if (fit_objective == FitObjective::SyndromeCensored) {
              const double probability = std::exp(decoder.syndrome_log_probability);
              partial_selected_masses[thread_index] += probability;
              for (size_t error_index = 0; error_index < gradient.size(); ++error_index) {
                partial_mass_gradients[thread_index][error_index] +=
                    probability * log_probability_gradient[error_index];
              }
            }
          }
          partial_shots[thread_index] += row_shots;
        }
      } catch (...) {
        errors[thread_index] = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  ObjectiveResult result{0.0, std::vector<double>(num_errors, 0.0), 0, 0.0};
  long double total_loss = 0.0;
  long double selected_mass = 0.0;
  std::vector<double> mass_gradient(num_errors, 0.0);
  for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    total_loss += partial_losses[thread_index];
    selected_mass += partial_selected_masses[thread_index];
    result.shots += partial_shots[thread_index];
    for (size_t error_index = 0; error_index < num_errors; ++error_index) {
      result.gradient[error_index] += partial_gradients[thread_index][error_index];
      mass_gradient[error_index] += partial_mass_gradients[thread_index][error_index];
    }
  }
  if (result.shots == 0 && fit_objective != FitObjective::SyndromeCensored) {
    throw std::runtime_error("selected syndromes contain no training shots");
  }
  if (fit_objective == FitObjective::SyndromeCensored) {
    if (censored_total_shots < result.shots || selected_mass <= 0.0 || selected_mass >= 1.0) {
      throw std::runtime_error("invalid censored syndrome mass or total shot count");
    }
    const uint64_t other_shots = censored_total_shots - result.shots;
    total_loss -= other_shots * std::log1pl(-selected_mass);
    const double other_gradient_scale = other_shots / (1.0 - selected_mass);
    for (size_t error_index = 0; error_index < num_errors; ++error_index) {
      result.gradient[error_index] += other_gradient_scale * mass_gradient[error_index];
    }
    result.shots = censored_total_shots;
  }
  result.data_nll = static_cast<double>(total_loss / result.shots);
  if (!std::isfinite(result.data_nll)) {
    throw std::runtime_error("non-finite objective value while fitting");
  }
  for (double& value : result.gradient) {
    value /= result.shots;
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite objective gradient while fitting");
    }
  }
  result.elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

std::vector<double> decode_values(const stim::DetectorErrorModel& dem,
                                  const std::vector<SyndromeCounts>& rows, size_t beam_width,
                                  size_t num_threads, FitObjective fit_objective,
                                  TesseractTrellisRankingMode ranking_mode,
                                  double future_detcost_scale) {
  std::vector<double> values(rows.size(), std::numeric_limits<double>::quiet_NaN());
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    workers.emplace_back([&, thread_index]() {
      try {
        TesseractTrellisConfig config;
        config.dem = dem;
        config.beam_width = beam_width;
        config.merge_errors = false;
        config.ranking_mode = ranking_mode;
        config.future_detcost_scale = future_detcost_scale;
        TesseractTrellisDecoder decoder(config);
        for (size_t row_index = thread_index; row_index < rows.size(); row_index += num_threads) {
          decoder.decode_shot(rows[row_index].detections);
          values[row_index] = fit_objective == FitObjective::Observable
                                  ? decoder.observable_probability()
                                  : decoder.syndrome_log_probability;
          if (decoder.low_confidence_flag || !std::isfinite(values[row_index]) ||
              (fit_objective == FitObjective::Observable &&
               (values[row_index] < 0.0 || values[row_index] > 1.0)) ||
              (fit_objective != FitObjective::Observable && values[row_index] > 0.0)) {
            throw std::runtime_error("invalid trellis value while scoring");
          }
        }
      } catch (...) {
        errors[thread_index] = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return values;
}

double nll_from_values(const std::vector<double>& values, const std::vector<SyndromeCounts>& rows,
                       bool test, FitObjective fit_objective, uint64_t censored_total_shots) {
  long double loss = 0.0;
  long double selected_mass = 0.0;
  uint64_t shots = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const uint64_t zeros = test ? rows[i].test_zero_count : rows[i].train_zero_count;
    const uint64_t ones = test ? rows[i].test_one_count : rows[i].train_one_count;
    if (fit_objective == FitObjective::Observable) {
      const double probability = values[i];
      if (probability <= 0.0 || probability >= 1.0) {
        throw std::runtime_error("conditional NLL requires probabilities strictly between 0 and 1");
      }
      loss -= ones * std::log(probability);
      loss -= zeros * std::log1p(-probability);
    } else {
      if (values[i] > 0.0) {
        throw std::runtime_error("syndrome log probability cannot be positive");
      }
      loss -= (zeros + ones) * values[i];
      if (fit_objective == FitObjective::SyndromeCensored) {
        selected_mass += std::exp(values[i]);
      }
    }
    shots += zeros + ones;
  }
  if (fit_objective == FitObjective::SyndromeCensored) {
    if (censored_total_shots < shots || selected_mass <= 0.0 || selected_mass >= 1.0) {
      throw std::runtime_error("invalid censored syndrome mass or total shot count while scoring");
    }
    loss -= (censored_total_shots - shots) * std::log1pl(-selected_mass);
    shots = censored_total_shots;
  }
  if (shots == 0) {
    throw std::runtime_error("selected syndromes contain no shots while scoring");
  }
  const double result = static_cast<double>(loss / shots);
  if (!std::isfinite(result)) {
    throw std::runtime_error("non-finite NLL while scoring");
  }
  return result;
}

void write_scores(const std::string& path, const std::vector<SyndromeCounts>& rows,
                  const std::vector<double>& baseline_probabilities,
                  const std::vector<double>& fitted_probabilities) {
  std::ofstream output(path);
  if (!output) {
    throw std::invalid_argument("could not write scores: " + path);
  }
  output << "syndrome train_zero train_one test_zero test_one baseline_q fitted_q\n";
  output << std::setprecision(17);
  for (size_t i = 0; i < rows.size(); ++i) {
    output << rows[i].syndrome << ' ' << rows[i].train_zero_count << ' ' << rows[i].train_one_count
           << ' ' << rows[i].test_zero_count << ' ' << rows[i].test_one_count << ' '
           << baseline_probabilities[i] << ' ' << fitted_probabilities[i] << '\n';
  }
  output.flush();
  if (!output) {
    throw std::runtime_error("failed while writing scores: " + path);
  }
}

std::vector<SyndromeCounts> rows_for_population(const std::vector<SyndromeCounts>& rows,
                                                bool test) {
  std::vector<SyndromeCounts> active;
  active.reserve(rows.size());
  for (const auto& row : rows) {
    const uint64_t zeros = test ? row.test_zero_count : row.train_zero_count;
    const uint64_t ones = test ? row.test_one_count : row.train_one_count;
    if (zeros != 0 || ones != 0) {
      active.push_back(row);
    }
  }
  if (active.empty()) {
    throw std::invalid_argument("requested population contains no shots");
  }
  return active;
}

void write_partial_summary(const std::string& path, const std::string& mode,
                           const std::string& objective, const std::string& population,
                           size_t beam_width, size_t num_threads, size_t num_errors,
                           size_t input_rows, size_t active_rows, uint64_t shots, double data_nll,
                           double elapsed_seconds, const std::string& ranking_mode,
                           double future_detcost_scale, const std::vector<double>* gradient,
                           const std::vector<SyndromeCounts>* value_rows = nullptr,
                           const std::vector<double>* values = nullptr) {
  const std::string temporary_path = path + ".tmp";
  std::ofstream output(temporary_path);
  if (!output) {
    throw std::invalid_argument("could not write partial summary: " + temporary_path);
  }
  output << std::setprecision(17);
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"objective\": \"" << objective << "\",\n"
         << "  \"population\": \"" << population << "\",\n"
         << "  \"beam_width\": " << beam_width << ",\n"
         << "  \"threads\": " << num_threads << ",\n"
         << "  \"model_errors\": " << num_errors << ",\n"
         << "  \"input_rows\": " << input_rows << ",\n"
         << "  \"active_rows\": " << active_rows << ",\n"
         << "  \"shots\": " << shots << ",\n"
         << "  \"data_nll\": " << data_nll << ",\n"
         << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n"
         << "  \"ranking_mode\": \"" << ranking_mode << "\",\n"
         << "  \"future_detcost_scale\": " << future_detcost_scale << ",\n"
         << "  \"gradient\": ";
  if (gradient == nullptr) {
    output << "null,\n";
  } else {
    output << '[';
    for (size_t i = 0; i < gradient->size(); ++i) {
      if (i != 0) {
        output << ',';
      }
      output << (*gradient)[i];
    }
    output << "],\n";
  }
  output << "  \"syndromes\": ";
  if (value_rows == nullptr || values == nullptr) {
    output << "null,\n"
           << "  \"zero_counts\": null,\n"
           << "  \"one_counts\": null,\n"
           << "  \"values\": null\n";
  } else {
    if (value_rows->size() != values->size()) {
      throw std::invalid_argument("partial value rows and values have different sizes");
    }
    output << '[';
    for (size_t i = 0; i < value_rows->size(); ++i) {
      if (i != 0) {
        output << ',';
      }
      output << '\"' << (*value_rows)[i].syndrome << '\"';
    }
    output << "],\n  \"zero_counts\": [";
    for (size_t i = 0; i < value_rows->size(); ++i) {
      if (i != 0) {
        output << ',';
      }
      const auto& row = (*value_rows)[i];
      output << (population == "test" ? row.test_zero_count : row.train_zero_count);
    }
    output << "],\n  \"one_counts\": [";
    for (size_t i = 0; i < value_rows->size(); ++i) {
      if (i != 0) {
        output << ',';
      }
      const auto& row = (*value_rows)[i];
      output << (population == "test" ? row.test_one_count : row.train_one_count);
    }
    output << "],\n  \"values\": [";
    for (size_t i = 0; i < values->size(); ++i) {
      if (i != 0) {
        output << ',';
      }
      output << (*values)[i];
    }
    output << "]\n";
  }
  output << "}\n";
  output.flush();
  if (!output) {
    std::remove(temporary_path.c_str());
    throw std::runtime_error("failed while writing partial summary: " + temporary_path);
  }
  output.close();
  if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
    std::remove(temporary_path.c_str());
    throw std::runtime_error("could not atomically publish partial summary: " + path);
  }
}

void run_partial(int argc, char** argv) {
  if (argc != 12) {
    throw std::invalid_argument(
        "usage: trellis_fit_pilot partial DEM COUNTS OUT_JSON BEAM THREADS "
        "[gradient|score|predict] [observable|syndrome] [train|test] RANKING_MODE "
        "FUTURE_DETCOST_SCALE");
  }
  const std::string dem_path = argv[2];
  const std::string counts_path = argv[3];
  const std::string output_path = argv[4];
  const size_t beam_width = parse_uint64_strict(argv[5], "BEAM");
  const size_t num_threads = parse_uint64_strict(argv[6], "THREADS");
  const std::string mode = argv[7];
  const std::string objective_name = argv[8];
  const std::string population = argv[9];
  const std::string ranking_mode_name = argv[10];
  const double future_detcost_scale =
      parse_double_strict(argv[11], "FUTURE_DETCOST_SCALE");
  if (mode != "gradient" && mode != "score" && mode != "predict") {
    throw std::invalid_argument("partial mode must be gradient, score, or predict");
  }
  const FitObjective objective =
      objective_name == "observable"
          ? FitObjective::Observable
          : objective_name == "syndrome"
                ? FitObjective::Syndrome
                : throw std::invalid_argument(
                      "distributed partial objective must be observable or syndrome");
  if (population != "train" && population != "test") {
    throw std::invalid_argument("partial population must be train or test");
  }
  if (mode == "gradient" && population != "train") {
    throw std::invalid_argument("partial gradients are defined for the training population");
  }
  if (mode == "predict" && objective != FitObjective::Observable) {
    throw std::invalid_argument("partial prediction requires the observable objective");
  }
  if (beam_width == 0 || num_threads == 0 || !std::isfinite(future_detcost_scale) ||
      future_detcost_scale < 0.0) {
    throw std::invalid_argument("invalid partial fit parameter");
  }

  const auto ranking_mode = parse_ranking_mode(ranking_mode_name);
  const auto rows = read_counts(counts_path);
  const stim::DetectorErrorModel dem = read_dem(dem_path);
  // Distributed controllers have already constructed the effective starting
  // model. Validate its topology and strict probabilities without reapplying
  // the initialization floor to later optimizer candidates.
  (void)dem_with_probability_floor(dem);
  (void)error_logits(dem);
  if (dem.count_observables() != 1 || dem.count_errors() == 0) {
    throw std::invalid_argument(
        "partial fitter requires a DEM with errors and exactly one observable");
  }
  validate_detections_against_dem(rows, dem);
  const bool test = population == "test";
  const auto active = rows_for_population(rows, test);

  if (mode == "gradient") {
    const auto result = objective_and_gradient(dem, active, beam_width, num_threads, objective, 0,
                                               ranking_mode, future_detcost_scale);
    write_partial_summary(output_path, mode, objective_name, population, beam_width, num_threads,
                          dem.count_errors(), rows.size(), active.size(), result.shots,
                          result.data_nll, result.elapsed_seconds, ranking_mode_name,
                          future_detcost_scale, &result.gradient);
    return;
  }

  const auto started = std::chrono::steady_clock::now();
  const auto values = decode_values(dem, active, beam_width, num_threads, objective, ranking_mode,
                                    future_detcost_scale);
  const double data_nll = nll_from_values(values, active, test, objective, 0);
  const auto totals = validate_detections_against_dem(active, dem);
  const uint64_t shots = test ? totals.test : totals.train;
  const double elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  write_partial_summary(output_path, mode, objective_name, population, beam_width, num_threads,
                        dem.count_errors(), rows.size(), active.size(), shots, data_nll,
                        elapsed_seconds, ranking_mode_name, future_detcost_scale, nullptr,
                        mode == "predict" ? &active : nullptr,
                        mode == "predict" ? &values : nullptr);
}

void write_summary(const std::string& path, size_t beam_width, size_t num_threads,
                   size_t num_errors, size_t num_syndromes, uint64_t train_shots,
                   uint64_t test_shots, size_t requested_steps, size_t optimizer_updates,
                   double learning_rate, double l2, double max_shift, double baseline_test_nll,
                   double fitted_test_nll, double final_shift_l2, double final_shift_max,
                   size_t shifts_at_bound, double total_seconds, FitObjective fit_objective,
                   const std::string& ranking_mode, double future_detcost_scale,
                   const std::vector<HistoryEntry>& history) {
  std::ofstream output(path);
  if (!output) {
    throw std::invalid_argument("could not write summary: " + path);
  }
  output << std::setprecision(17);
  output << "{\n"
         << "  \"schema_version\": 4,\n"
         << "  \"fit_objective\": \""
         << (fit_objective == FitObjective::Observable         ? "observable"
             : fit_objective == FitObjective::SyndromeCensored ? "syndrome_censored"
                                                               : "syndrome")
         << "\",\n"
         << "  \"beam_width\": " << beam_width << ",\n"
         << "  \"threads\": " << num_threads << ",\n"
         << "  \"model_errors\": " << num_errors << ",\n"
         << "  \"syndromes\": " << num_syndromes << ",\n"
         << "  \"train_shots\": " << train_shots << ",\n"
         << "  \"test_shots\": " << test_shots << ",\n"
         << "  \"requested_steps\": " << requested_steps << ",\n"
         << "  \"optimizer_updates\": " << optimizer_updates << ",\n"
         << "  \"learning_rate\": " << learning_rate << ",\n"
         << "  \"l2\": " << l2 << ",\n"
         << "  \"probability_floor\": " << kProbabilityFloor << ",\n"
         << "  \"max_logit_shift\": " << max_shift << ",\n"
         << "  \"ranking_mode\": \"" << ranking_mode << "\",\n"
         << "  \"future_detcost_scale\": " << future_detcost_scale << ",\n"
         << "  \"baseline_test_nll\": " << baseline_test_nll << ",\n"
         << "  \"fitted_test_nll\": " << fitted_test_nll << ",\n"
         << "  \"final_shift_l2\": " << final_shift_l2 << ",\n"
         << "  \"final_shift_max\": " << final_shift_max << ",\n"
         << "  \"shifts_at_bound\": " << shifts_at_bound << ",\n"
         << "  \"total_seconds\": " << total_seconds << ",\n"
         << "  \"history\": [\n";
  for (size_t i = 0; i < history.size(); ++i) {
    const auto& entry = history[i];
    output << "    {\"step\": " << entry.step << ", \"data_nll\": " << entry.data_nll
           << ", \"regularization\": " << entry.regularization
           << ", \"objective\": " << entry.objective
           << ", \"gradient_norm\": " << entry.gradient_norm
           << ", \"accepted_step_size\": " << entry.accepted_step_size
           << ", \"elapsed_seconds\": " << entry.elapsed_seconds << "}";
    output << (i + 1 == history.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  output.flush();
  if (!output) {
    throw std::runtime_error("failed while writing summary: " + path);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "partial") {
    try {
      run_partial(argc, argv);
    } catch (const std::exception& ex) {
      std::cerr << "trellis partial fit failed: " << ex.what() << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
  if (argc != 12 && argc != 13 && argc != 15 && argc != 17) {
    std::cerr << "usage: trellis_fit_pilot DEM COUNTS OUT_DEM OUT_SCORES OUT_SUMMARY BEAM "
                 "THREADS STEPS LEARNING_RATE L2 MAX_SHIFT "
                 "[observable|syndrome|syndrome_censored [TRAIN_TOTAL TEST_TOTAL] "
                 "[RANKING_MODE FUTURE_DETCOST_SCALE]]\n";
    return EXIT_FAILURE;
  }
  try {
    const std::string dem_path = argv[1];
    const std::string counts_path = argv[2];
    const std::string out_dem_path = argv[3];
    const std::string out_scores_path = argv[4];
    const std::string out_summary_path = argv[5];
    const size_t beam_width = parse_uint64_strict(argv[6], "BEAM");
    const size_t num_threads = parse_uint64_strict(argv[7], "THREADS");
    const size_t steps = parse_uint64_strict(argv[8], "STEPS");
    const double learning_rate = parse_double_strict(argv[9], "LEARNING_RATE");
    const double l2 = parse_double_strict(argv[10], "L2");
    const double max_shift = parse_double_strict(argv[11], "MAX_SHIFT");
    const std::string fit_objective_name = argc >= 13 ? argv[12] : "observable";
    const FitObjective fit_objective =
        fit_objective_name == "observable" ? FitObjective::Observable
        : fit_objective_name == "syndrome" ? FitObjective::Syndrome
        : fit_objective_name == "syndrome_censored"
            ? FitObjective::SyndromeCensored
            : throw std::invalid_argument(
                  "fit objective must be observable, syndrome, or syndrome_censored");
    const bool has_censored_totals =
        fit_objective == FitObjective::SyndromeCensored && (argc == 15 || argc == 17);
    const bool has_ranking = (fit_objective == FitObjective::SyndromeCensored && argc == 17) ||
                             (fit_objective != FitObjective::SyndromeCensored && argc == 15);
    const uint64_t train_total_shots =
        has_censored_totals ? parse_uint64_strict(argv[13], "TRAIN_TOTAL") : 0;
    const uint64_t test_total_shots =
        has_censored_totals ? parse_uint64_strict(argv[14], "TEST_TOTAL") : 0;
    const size_t ranking_offset = fit_objective == FitObjective::SyndromeCensored ? 15 : 13;
    const std::string ranking_mode_name = has_ranking ? argv[ranking_offset] : "mass";
    const double future_detcost_scale =
        has_ranking ? parse_double_strict(argv[ranking_offset + 1], "FUTURE_DETCOST_SCALE") : 2.0;
    const auto ranking_mode = parse_ranking_mode(ranking_mode_name);
    if (beam_width == 0 || num_threads == 0 || steps == 0 || !std::isfinite(learning_rate) ||
        learning_rate <= 0.0 || !std::isfinite(l2) || l2 < 0.0 || !std::isfinite(max_shift) ||
        max_shift <= 0.0 || !std::isfinite(future_detcost_scale) || future_detcost_scale < 0.0 ||
        ((fit_objective == FitObjective::SyndromeCensored) != has_censored_totals) ||
        (fit_objective != FitObjective::SyndromeCensored && argc == 17) ||
        (fit_objective == FitObjective::SyndromeCensored &&
         (train_total_shots == 0 || test_total_shots == 0))) {
      throw std::invalid_argument("invalid fit parameter");
    }

    const auto total_started = std::chrono::steady_clock::now();
    const auto rows = read_counts(counts_path);
    // Preserve the supplied, already merged effective topology. Flooring keeps
    // zero-frequency symptoms trainable without allowing preprocessing to
    // remove them.
    const stim::DetectorErrorModel baseline_dem = dem_with_probability_floor(read_dem(dem_path));
    if (baseline_dem.count_observables() != 1) {
      throw std::invalid_argument("fitter requires a DEM with exactly one observable");
    }
    if (baseline_dem.count_errors() == 0) {
      throw std::invalid_argument("fitter requires at least one DEM error mechanism");
    }
    const CountTotals selected_count_totals = validate_detections_against_dem(rows, baseline_dem);
    if (fit_objective == FitObjective::SyndromeCensored &&
        (selected_count_totals.train > train_total_shots ||
         selected_count_totals.test > test_total_shots)) {
      throw std::invalid_argument("selected syndrome counts exceed the censored populations");
    }
    const std::vector<double> baseline_logits = error_logits(baseline_dem);
    std::vector<double> logits = baseline_logits;
    std::vector<HistoryEntry> history;
    history.reserve(steps + 1);
    size_t optimizer_updates = 0;

    for (size_t step = 0; step < steps; ++step) {
      const auto current_dem = dem_with_logits(baseline_dem, logits);
      auto objective =
          objective_and_gradient(current_dem, rows, beam_width, num_threads, fit_objective,
                                 train_total_shots, ranking_mode, future_detcost_scale);
      double regularization = 0.0;
      for (size_t i = 0; i < logits.size(); ++i) {
        const double shift = logits[i] - baseline_logits[i];
        regularization += 0.5 * l2 * shift * shift;
        objective.gradient[i] += l2 * shift;
      }
      double gradient_norm_squared = 0.0;
      for (double value : objective.gradient) {
        gradient_norm_squared += value * value;
      }
      const double gradient_norm = std::sqrt(gradient_norm_squared);
      const double current_objective = objective.data_nll + regularization;
      if (!std::isfinite(gradient_norm) || !std::isfinite(current_objective)) {
        throw std::runtime_error("non-finite optimizer state");
      }
      double accepted_step_size = 0.0;
      double candidate_step_size = learning_rate;
      std::vector<double> candidate_logits(logits.size());
      for (size_t attempt = 0; attempt < 12 && gradient_norm > 0.0; ++attempt) {
        for (size_t i = 0; i < logits.size(); ++i) {
          const double updated =
              logits[i] - candidate_step_size * objective.gradient[i] / gradient_norm;
          candidate_logits[i] =
              std::clamp(updated, baseline_logits[i] - max_shift, baseline_logits[i] + max_shift);
        }
        const auto candidate_dem = dem_with_logits(baseline_dem, candidate_logits);
        const auto candidate_values =
            decode_values(candidate_dem, rows, beam_width, num_threads, fit_objective, ranking_mode,
                          future_detcost_scale);
        const double candidate_data_nll =
            nll_from_values(candidate_values, rows, false, fit_objective, train_total_shots);
        double candidate_regularization = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
          const double shift = candidate_logits[i] - baseline_logits[i];
          candidate_regularization += 0.5 * l2 * shift * shift;
        }
        if (candidate_data_nll + candidate_regularization < current_objective) {
          logits = candidate_logits;
          accepted_step_size = candidate_step_size;
          break;
        }
        candidate_step_size *= 0.5;
      }
      history.push_back({optimizer_updates, objective.data_nll, regularization, current_objective,
                         gradient_norm, accepted_step_size, objective.elapsed_seconds});
      std::cerr << "step " << step << '/' << steps << " data_nll=" << std::setprecision(10)
                << objective.data_nll << " objective=" << current_objective
                << " grad_norm=" << gradient_norm << " accepted_step=" << accepted_step_size
                << " seconds=" << objective.elapsed_seconds << std::endl;
      if (accepted_step_size == 0.0) {
        std::cerr << "stopping because backtracking found no improving step" << std::endl;
        break;
      }
      ++optimizer_updates;
    }

    const auto fitted_dem = dem_with_logits(baseline_dem, logits);
    auto final_objective =
        objective_and_gradient(fitted_dem, rows, beam_width, num_threads, fit_objective,
                               train_total_shots, ranking_mode, future_detcost_scale);
    double final_regularization = 0.0;
    double final_gradient_norm_squared = 0.0;
    double final_shift_l2_squared = 0.0;
    double final_shift_max = 0.0;
    size_t shifts_at_bound = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
      const double shift = logits[i] - baseline_logits[i];
      final_regularization += 0.5 * l2 * shift * shift;
      final_objective.gradient[i] += l2 * shift;
      final_shift_l2_squared += shift * shift;
      final_shift_max = std::max(final_shift_max, std::abs(shift));
      if (std::abs(shift) >= max_shift - 1e-12) {
        ++shifts_at_bound;
      }
    }
    for (double value : final_objective.gradient) {
      final_gradient_norm_squared += value * value;
    }
    if (!std::isfinite(final_regularization) || !std::isfinite(final_gradient_norm_squared) ||
        !std::isfinite(final_shift_l2_squared) || !std::isfinite(final_shift_max)) {
      throw std::runtime_error("non-finite final optimizer state");
    }
    if (history.empty() || history.back().step != optimizer_updates) {
      history.push_back({optimizer_updates, final_objective.data_nll, final_regularization,
                         final_objective.data_nll + final_regularization,
                         std::sqrt(final_gradient_norm_squared), 0.0,
                         final_objective.elapsed_seconds});
    }

    const auto baseline_probabilities =
        decode_values(baseline_dem, rows, beam_width, num_threads, FitObjective::Observable,
                      ranking_mode, future_detcost_scale);
    const auto fitted_probabilities =
        decode_values(fitted_dem, rows, beam_width, num_threads, FitObjective::Observable,
                      ranking_mode, future_detcost_scale);
    const auto baseline_test_values =
        fit_objective == FitObjective::Observable
            ? baseline_probabilities
            : decode_values(baseline_dem, rows, beam_width, num_threads, fit_objective,
                            ranking_mode, future_detcost_scale);
    const auto fitted_test_values =
        fit_objective == FitObjective::Observable
            ? fitted_probabilities
            : decode_values(fitted_dem, rows, beam_width, num_threads, fit_objective, ranking_mode,
                            future_detcost_scale);
    const double baseline_test_nll =
        nll_from_values(baseline_test_values, rows, true, fit_objective, test_total_shots);
    const double fitted_test_nll =
        nll_from_values(fitted_test_values, rows, true, fit_objective, test_total_shots);
    uint64_t train_shots = selected_count_totals.train;
    uint64_t test_shots = selected_count_totals.test;
    if (fit_objective == FitObjective::SyndromeCensored) {
      train_shots = train_total_shots;
      test_shots = test_total_shots;
    }

    std::ofstream dem_output(out_dem_path);
    if (!dem_output) {
      throw std::invalid_argument("could not write fitted DEM: " + out_dem_path);
    }
    dem_output << fitted_dem.str();
    dem_output.flush();
    if (!dem_output) {
      throw std::runtime_error("failed while writing fitted DEM: " + out_dem_path);
    }
    write_scores(out_scores_path, rows, baseline_probabilities, fitted_probabilities);
    const double total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_started).count();
    write_summary(out_summary_path, beam_width, num_threads, logits.size(), rows.size(),
                  train_shots, test_shots, steps, optimizer_updates, learning_rate, l2, max_shift,
                  baseline_test_nll, fitted_test_nll, std::sqrt(final_shift_l2_squared),
                  final_shift_max, shifts_at_bound, total_seconds, fit_objective, ranking_mode_name,
                  future_detcost_scale, history);
    std::cerr << "finished fit in " << total_seconds << " seconds; held-out NLL "
              << baseline_test_nll << " -> " << fitted_test_nll << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "trellis pilot fit failed: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
