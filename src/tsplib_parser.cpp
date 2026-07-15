#include "tsp/tsplib_parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tsp {
namespace {

std::string trim(const std::string& value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }
    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
        --last;
    }
    return value.substr(first, last - first);
}

std::string uppercase(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::pair<std::string, std::string> parse_header_line(const std::string& line) {
    const size_t colon = line.find(':');
    if (colon != std::string::npos) {
        return {uppercase(trim(line.substr(0, colon))), trim(line.substr(colon + 1))};
    }

    std::istringstream iss(line);
    std::string key;
    iss >> key;
    std::string value;
    std::getline(iss, value);
    return {uppercase(trim(key)), trim(value)};
}

void assign_coord(Instance& instance,
                  std::vector<bool>& coordinate_ids_seen,
                  int id,
                  double x,
                  double y) {
    if (id <= 0) {
        throw std::runtime_error("NODE_COORD_SECTION uses non-positive node id");
    }
    if (instance.dimension <= 0) {
        throw std::runtime_error("NODE_COORD_SECTION requires DIMENSION before the section");
    }
    if (id > instance.dimension) {
        throw std::runtime_error("NODE_COORD_SECTION node id exceeds DIMENSION");
    }
    if (instance.coords.empty()) {
        instance.coords.assign(static_cast<size_t>(instance.dimension), Coord{});
    }
    if (coordinate_ids_seen.empty()) {
        coordinate_ids_seen.assign(static_cast<size_t>(instance.dimension), false);
    }

    const size_t index = static_cast<size_t>(id - 1);
    if (coordinate_ids_seen[index]) {
        throw std::runtime_error("NODE_COORD_SECTION contains duplicate node id " +
                                 std::to_string(id));
    }
    coordinate_ids_seen[index] = true;
    instance.coords[index] = Coord{x, y};
}

void validate_coordinate_ids(const Instance& instance,
                             const std::vector<bool>& coordinate_ids_seen,
                             bool saw_node_coord_section) {
    if (!saw_node_coord_section) {
        return;
    }
    for (int id = 1; id <= instance.dimension; ++id) {
        const size_t index = static_cast<size_t>(id - 1);
        if (index >= coordinate_ids_seen.size() || !coordinate_ids_seen[index]) {
            throw std::runtime_error(
                "NODE_COORD_SECTION is missing node id " + std::to_string(id) +
                "; expected each id from 1 through DIMENSION exactly once");
        }
    }
}

void validate_basic_fields(const Instance& instance) {
    if (instance.dimension <= 0) {
        throw std::runtime_error("missing or invalid DIMENSION in TSPLIB file");
    }
    if (instance.edge_weight_type.empty()) {
        throw std::runtime_error("missing EDGE_WEIGHT_TYPE in TSPLIB file");
    }
}

void reject_unsupported_problem_type(const Instance& instance) {
    if (uppercase(instance.type) == "ATSP") {
        throw std::runtime_error(
            "unsupported TSPLIB TYPE ATSP: 2-opt delta evaluation requires a symmetric TSP distance matrix");
    }
}

void reject_asymmetric_explicit_full_matrix(const Instance& instance) {
    if (uppercase(instance.edge_weight_type) != "EXPLICIT") {
        return;
    }

    const std::string format = instance.edge_weight_format.empty()
                                   ? "FULL_MATRIX"
                                   : uppercase(instance.edge_weight_format);
    if (format != "FULL_MATRIX") {
        return;
    }

    const size_t n = static_cast<size_t>(instance.dimension);
    const size_t expected_weight_count = n * n;
    // Keep malformed-section diagnostics in DistanceMatrix, which already owns
    // the exact count validation for every EXPLICIT format.
    if (instance.raw_weights.size() != expected_weight_count) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const int forward = instance.raw_weights[i * n + j];
            const int reverse = instance.raw_weights[j * n + i];
            if (forward != reverse) {
                std::ostringstream message;
                message << "unsupported asymmetric EXPLICIT FULL_MATRIX: 2-opt delta evaluation "
                           "requires symmetric distances ("
                        << (i + 1) << "->" << (j + 1) << "=" << forward << ", "
                        << (j + 1) << "->" << (i + 1) << "=" << reverse << ")";
                throw std::runtime_error(message.str());
            }
        }
    }
}

}  // namespace

Instance load_tsplib(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open TSPLIB file: " + path);
    }

    enum class Section {
        Header,
        NodeCoord,
        EdgeWeight,
    };

    Instance instance;
    Section section = Section::Header;
    bool saw_node_coord_section = false;
    std::vector<bool> coordinate_ids_seen;
    std::string line;

    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const std::string upper_line = uppercase(line);
        if (upper_line == "EOF") {
            break;
        }

        if (upper_line == "NODE_COORD_SECTION") {
            if (instance.dimension <= 0) {
                throw std::runtime_error(
                    "NODE_COORD_SECTION requires a positive DIMENSION before the section");
            }
            if (!saw_node_coord_section) {
                coordinate_ids_seen.assign(static_cast<size_t>(instance.dimension), false);
                saw_node_coord_section = true;
            }
            section = Section::NodeCoord;
            continue;
        }
        if (upper_line == "EDGE_WEIGHT_SECTION") {
            section = Section::EdgeWeight;
            continue;
        }
        if (upper_line == "DISPLAY_DATA_SECTION" || upper_line == "TOUR_SECTION") {
            break;
        }

        if (section == Section::Header) {
            const auto [key, value] = parse_header_line(line);
            if (key == "NAME") {
                instance.name = value;
            } else if (key == "TYPE") {
                instance.type = value;
            } else if (key == "COMMENT") {
                if (!instance.comment.empty()) {
                    instance.comment += "\n";
                }
                instance.comment += value;
            } else if (key == "DIMENSION") {
                instance.dimension = std::stoi(value);
            } else if (key == "EDGE_WEIGHT_TYPE") {
                instance.edge_weight_type = value;
            } else if (key == "EDGE_WEIGHT_FORMAT") {
                instance.edge_weight_format = value;
            }
            continue;
        }

        if (section == Section::NodeCoord) {
            std::istringstream iss(line);
            int id = 0;
            double x = 0.0;
            double y = 0.0;
            if (!(iss >> id >> x >> y)) {
                throw std::runtime_error("invalid NODE_COORD_SECTION line: " + line);
            }
            assign_coord(instance, coordinate_ids_seen, id, x, y);
            continue;
        }

        if (section == Section::EdgeWeight) {
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                if (uppercase(token) == "EOF") {
                    break;
                }
                instance.raw_weights.push_back(std::stoi(token));
            }
        }
    }

    validate_basic_fields(instance);
    validate_coordinate_ids(instance, coordinate_ids_seen, saw_node_coord_section);
    reject_unsupported_problem_type(instance);
    reject_asymmetric_explicit_full_matrix(instance);
    if (instance.name.empty()) {
        instance.name = path;
    }
    if (static_cast<int>(instance.coords.size()) > instance.dimension) {
        throw std::runtime_error("too many coordinates in NODE_COORD_SECTION");
    }
    return instance;
}

}  // namespace tsp
