#pragma once

#include <string>
#include <vector>

namespace tsp {

struct Coord {
    double x = 0.0;
    double y = 0.0;
};

struct Instance {
    std::string name;
    std::string type;
    int dimension = 0;
    std::string edge_weight_type;
    std::string edge_weight_format;
    std::vector<Coord> coords;
    std::vector<int> raw_weights;
    std::string comment;
};

}  // namespace tsp
