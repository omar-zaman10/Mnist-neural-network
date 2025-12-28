#include "data_loader.h"

#include <iostream>
#include <fstream>
#include <cstdio>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/istreamwrapper.h>

std::tuple<xt::xarray<double>,xt::xarray<double>> training_data_loader() {
    std::ifstream stream {"data/training_data.json"};
    if (stream.is_open()) {
        std::cout << "Opened json training_data file for reading\n";
    } else {
        throw std::runtime_error{
            "Could not open file for reading!\n\
            Run gzip data/training_data.json.gz"
        };
    }

    auto stream_wrapper = rapidjson::IStreamWrapper{stream};
    auto doc = rapidjson::Document{};
    doc.ParseStream(stream_wrapper);
    
    const auto& input_data = doc["input_data"];
    const auto& labels = doc["labels"];

    xt::xarray<double> x = xt::zeros<double>({input_data.Size(),input_data[0].Size()});
    xt::xarray<double> y = xt::zeros<double>({labels.Size()});

    for (int i = 0; i < labels.Size(); i++){
        y[i] = labels[i].GetDouble();
    }

    for (int i = 0; i < input_data.Size(); i++){
        for(int j = 0; j< input_data[0].Size(); j++){
            x[i * input_data[0].Size() + j] = input_data[i][j].GetDouble();
        }
    }

    return {x, y};
}
