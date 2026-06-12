#pragma once

#include <string>

#include "tsp/instance.hpp"

namespace tsp {

Instance load_tsplib(const std::string& path);

}  // namespace tsp
