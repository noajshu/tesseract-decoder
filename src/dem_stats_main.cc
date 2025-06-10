#include <argparse/argparse.hpp>
#include <fstream>
#include <iostream>

#include "common.h"
#include "stim.h"

int main(int argc, char **argv) {
  argparse::ArgumentParser program("dem_stats");
  std::string dem_path;
  std::string circuit_path;
  bool no_merge_errors = false;
  program.add_argument("--dem")
      .help("Path to detector error model")
      .store_into(dem_path);
  program.add_argument("--circuit")
      .help("Path to stim circuit file")
      .store_into(circuit_path);
  program.add_argument("--no-merge-errors")
      .help("Do not merge identical error mechanisms before analysis")
      .flag()
      .store_into(no_merge_errors);
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  if (dem_path.empty() && circuit_path.empty()) {
    std::cerr << "Must specify either --dem or --circuit" << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  stim::DetectorErrorModel dem;
  if (!dem_path.empty()) {
    FILE *f = fopen(dem_path.c_str(), "r");
    if (!f) {
      std::cerr << "Failed to open " << dem_path << std::endl;
      return EXIT_FAILURE;
    }
    dem = stim::DetectorErrorModel::from_file(f);
    fclose(f);
  } else {
    FILE *f = fopen(circuit_path.c_str(), "r");
    if (!f) {
      std::cerr << "Failed to open " << circuit_path << std::endl;
      return EXIT_FAILURE;
    }
    stim::Circuit circuit = stim::Circuit::from_file(f);
    fclose(f);
    dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
        circuit,
        /*decompose_errors=*/false,
        /*fold_loops=*/true,
        /*allow_gauge_detectors=*/true,
        /*approximate_disjoint_errors_threshold=*/1,
        /*ignore_decomposition_failures=*/false,
        /*block_decomposition_from_introducing_remnant_edges=*/false);
  }

  if (!no_merge_errors) {
    dem = common::merge_identical_errors(dem);
  }
  dem = common::remove_zero_probability_errors(dem);
  auto redund = common::find_redundant_errors(dem);

  std::cout << redund.size() << " of " << dem.count_errors()
            << " errors are redundant" << std::endl;
  return 0;
}
