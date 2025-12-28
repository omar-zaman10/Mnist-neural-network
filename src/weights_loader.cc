#include "weights_loader.h"

#include <iostream>
#include <cstdio>
#include <fstream>
#include <unordered_map>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/istreamwrapper.h>

std::unordered_map<int, xt::xarray<double> > biases_loader() {
    auto stream = std::ifstream{"data/biases.json"};
    if (stream.is_open()) {
        std::cout << "Opened json biases data file for reading\n";
    } else {  
        throw std::runtime_error{"Could not open file for reading!\n"};
    }

    auto stream_wrapper = rapidjson::IStreamWrapper{stream};
    auto doc = rapidjson::Document{};
    doc.ParseStream(stream_wrapper);

    auto biases_map = std::unordered_map<int, xt::xarray<double>>{};
    auto biases = xt::xarray<double>{};

    for (auto itr = doc.MemberBegin(); itr != doc.MemberEnd(); itr++) {
        biases = xt::zeros<double>({itr->value.Size()});
        
        for (int i = 0; i < itr->value.Size(); i++) { 
            biases[i] = itr->value[i].GetDouble();
        }        
        
        biases_map[std::stoi(itr->name.GetString())] = biases;
    }

    return biases_map;
}

std::unordered_map<int, xt::xarray<double>>  weights_loader() {
    auto stream = std::ifstream{"data/weights.json"};
    if (stream.is_open()) {
        std::cout << "Opened json weights data file for reading\n";
    } else {  
        throw std::runtime_error{"Could not open file for reading!\n"};
    }

    auto stream_wrapper = rapidjson::IStreamWrapper{stream};
    auto doc = rapidjson::Document{};
    doc.ParseStream(stream_wrapper);

    auto weights_map = std::unordered_map<int, xt::xarray<double>>{};
    auto weights = xt::xarray<double>{};

    for (auto itr = doc.MemberBegin(); itr != doc.MemberEnd(); itr++) {
        weights = xt::zeros<double>({itr->value.Size(), itr->value[0].Size()});        

        for (int i = 0; i < itr->value.Size(); i++){ 
            for(int j = 0; j < itr->value[0].Size(); j++){
                weights[i*itr->value[0].Size() + j] = itr->value[i][j].GetDouble();
            }
        }        
        
        weights_map[std::stoi(itr->name.GetString())] = weights;   
    }

    return weights_map;
}

