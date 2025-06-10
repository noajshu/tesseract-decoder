#include <argparse/argparse.hpp>
#include <fstream>
#include <iostream>

#include "stim.h"
#include "common.h"

int main(int argc, char **argv) {
    argparse::ArgumentParser program("dem_stats");
    std::string dem_path;
    bool no_merge_errors = false;
    program.add_argument("--dem")
        .required()
        .help("Path to detector error model")
        .store_into(dem_path);
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

    FILE *f = fopen(dem_path.c_str(), "r");
    if (!f) {
        std::cerr << "Failed to open " << dem_path << std::endl;
        return EXIT_FAILURE;
    }
    stim::DetectorErrorModel dem = stim::DetectorErrorModel::from_file(f);
    fclose(f);

    if (!no_merge_errors) {
        dem = common::merge_identical_errors(dem);
    }
    dem = common::remove_zero_probability_errors(dem);
    auto redund = common::find_redundant_errors(dem);

    std::cout << redund.size() << " of " << dem.count_errors()
              << " errors are redundant" << std::endl;
    return 0;
}
