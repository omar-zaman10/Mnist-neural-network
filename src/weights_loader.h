#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <unordered_map>

#include "xtensor/xarray.hpp"

std::unordered_map<int, xt::xarray<double> > biases_loader();

std::unordered_map<int, xt::xarray<double>>  weights_loader();

#endif // WEIGHTS_LOADER_H