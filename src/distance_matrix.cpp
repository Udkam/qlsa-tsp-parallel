#include "tsp/distance_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>
#include <string>

namespace tsp {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kGeoRadius = 6378.388;

std::string uppercase(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

double geo_to_radians(double x) {
    const int deg = static_cast<int>(x);
    const double min = x - static_cast<double>(deg);
    return kPi * (static_cast<double>(deg) + 5.0 * min / 3.0) / 180.0;
}

int coord_distance(const Coord& a, const Coord& b, const std::string& type) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;

    if (type == "EUC_2D") {
        return static_cast<int>(std::sqrt(dx * dx + dy * dy) + 0.5);
    }
    if (type == "CEIL_2D") {
        return static_cast<int>(std::ceil(std::sqrt(dx * dx + dy * dy)));
    }
    if (type == "ATT") {
        const double rij = std::sqrt((dx * dx + dy * dy) / 10.0);
        const int tij = static_cast<int>(std::round(rij));
        return (static_cast<double>(tij) < rij) ? tij + 1 : tij;
    }
    if (type == "GEO") {
        const double lat_i = geo_to_radians(a.x);
        const double lon_i = geo_to_radians(a.y);
        const double lat_j = geo_to_radians(b.x);
        const double lon_j = geo_to_radians(b.y);
        const double q1 = std::cos(lon_i - lon_j);
        const double q2 = std::cos(lat_i - lat_j);
        const double q3 = std::cos(lat_i + lat_j);
        double arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3);
        arg = std::clamp(arg, -1.0, 1.0);
        return static_cast<int>(kGeoRadius * std::acos(arg) + 1.0);
    }

    throw std::invalid_argument("unsupported EDGE_WEIGHT_TYPE: " + type);
}

void require_weight_count(const std::vector<int>& weights, size_t expected, const std::string& format) {
    if (weights.size() < expected) {
        throw std::runtime_error("EDGE_WEIGHT_SECTION for " + format + " has too few values");
    }
    if (weights.size() > expected) {
        throw std::runtime_error("EDGE_WEIGHT_SECTION for " + format + " has too many values");
    }
}

void set_symmetric(std::vector<int>& data, int n, int i, int j, int value) {
    data[static_cast<size_t>(i) * n + j] = value;
    data[static_cast<size_t>(j) * n + i] = value;
}

std::vector<int> build_explicit_matrix(const Instance& instance, const std::string& format) {
    const int n = instance.dimension;
    std::vector<int> data(static_cast<size_t>(n) * n, 0);
    const auto& weights = instance.raw_weights;

    size_t pos = 0;
    if (format == "FULL_MATRIX") {
        require_weight_count(weights, static_cast<size_t>(n) * n, format);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                data[static_cast<size_t>(i) * n + j] = weights[pos++];
            }
        }
        return data;
    }

    if (format == "UPPER_ROW") {
        require_weight_count(weights, static_cast<size_t>(n) * (n - 1) / 2, format);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                set_symmetric(data, n, i, j, weights[pos++]);
            }
        }
        return data;
    }

    if (format == "LOWER_ROW") {
        require_weight_count(weights, static_cast<size_t>(n) * (n - 1) / 2, format);
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                set_symmetric(data, n, i, j, weights[pos++]);
            }
        }
        return data;
    }

    if (format == "UPPER_DIAG_ROW") {
        require_weight_count(weights, static_cast<size_t>(n) * (n + 1) / 2, format);
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                set_symmetric(data, n, i, j, weights[pos++]);
            }
        }
        return data;
    }

    if (format == "LOWER_DIAG_ROW") {
        require_weight_count(weights, static_cast<size_t>(n) * (n + 1) / 2, format);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                set_symmetric(data, n, i, j, weights[pos++]);
            }
        }
        return data;
    }

    throw std::invalid_argument("unsupported EDGE_WEIGHT_FORMAT: " + format);
}

}  // namespace

DistanceMatrix::DistanceMatrix(const Instance& instance) : n_(instance.dimension) {
    if (n_ <= 0) {
        throw std::invalid_argument("TSPLIB instance dimension must be positive");
    }

    const std::string type = uppercase(instance.edge_weight_type);
    std::string format = uppercase(instance.edge_weight_format);

    if (type == "EXPLICIT") {
        if (format.empty()) {
            format = "FULL_MATRIX";
        }
        data_ = build_explicit_matrix(instance, format);
    } else {
        if (static_cast<int>(instance.coords.size()) != n_) {
            throw std::runtime_error("coordinate instance requires exactly DIMENSION coordinates");
        }
        data_.assign(static_cast<size_t>(n_) * n_, 0);
        for (int i = 0; i < n_; ++i) {
            for (int j = i + 1; j < n_; ++j) {
                const int d = coord_distance(instance.coords[static_cast<size_t>(i)],
                                             instance.coords[static_cast<size_t>(j)],
                                             type);
                set_symmetric(data_, n_, i, j, d);
            }
        }
    }

    for (int i = 0; i < n_; ++i) {
        data_[static_cast<size_t>(i) * n_ + i] = 0;
    }
}

}  // namespace tsp
