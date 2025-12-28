#include "data_loader.h"
#include "weights_loader.h"
#include "network.h"

int main() {    
    Network nn;
    auto [input_data, labels] = training_data_loader();
    nn.train(input_data,labels);

    // std::unordered_map<int, xt::xarray<double> > biases = biases_loader();
    // std::unordered_map<int, xt::xarray<double>> weights =  weights_loader();
    // nn.set_weights_and_biases(biases, weights);
    // nn.evaluate();

    return 0;
}