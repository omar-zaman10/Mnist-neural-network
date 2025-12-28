#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <tuple>

#include <xtensor/xarray.hpp>

std::tuple<xt::xarray<double>,xt::xarray<double>> training_data_loader();

#endif // DATA_LOADER_H