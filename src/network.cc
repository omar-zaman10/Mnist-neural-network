#include "network.h"

#include <cmath>
#include <vector>
#include <unordered_map>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>

#include <xtensor-blas/xblas.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-blas/xlapack.hpp>

void Network::set_weights_and_biases(
    std::unordered_map<int, xt::xarray<double>> biases,
    std::unordered_map<int, xt::xarray<double>> weights
)
{
    B = biases;
    W = weights;
}

void Network::initialise_weights() {
    for (int l = 1; l < layers; l++){
        W[l] = xt::random::randn<double>({nn_structure[l], nn_structure[l-1]});
        B[l] = xt::random::randn<double>({nn_structure[l]});
    }
}

void Network::initialise_deltas() {
    for(int l =1; l < layers; l++){
        delta_W[l] = xt::zeros<double>({W[l].shape(0), W[l].shape(1)});
        delta_B[l] = xt::zeros<double>({B[l].shape(0)});
    }
}

void Network::initialise_data(xt::xarray<double> x, xt::xarray<double> y) {
    input_data = x;
    labels = y;
}

double Network::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double Network::sigmoid_gradient(double z) {
    auto f = sigmoid(z);
    return f*(1-f) ;
}

void Network::feedforward(xt::xarray<double> x) {
    H[1] = x;
    xt::xarray<double>  z_l;
    auto vecf = xt::vectorize(&Network::sigmoid);

    for (int l =1; l < layers; l++) {
        z_l = xt::linalg::dot(W[l], H[l]) + B[l];
        H[l+1] = vecf(z_l);            
        Z[l+1] = z_l;
    }

    H[layers] = H[layers] / xt::sum(H[layers]);
}

xt::xarray<double> Network::cost_derivative(xt::xarray<double> y) {
    xt::xarray<double> deriv = H[layers] - y;
    return deriv;
}

void Network::backprop(double y) {
    xt::xarray<double> vect_label = vectorised_label(y);
    xt::xarray<double> cost_grad = cost_derivative(vect_label);
    auto vecf = xt::vectorize(&Network::sigmoid_gradient);

    xt::xarray<double> output_deltas = cost_grad * vecf(Z[layers]);
    std::unordered_map<int, xt::xarray<double> > deltas;
    deltas[layers] = output_deltas;

    for (int l= layers-1; l >0; l--) {
        delta_B[l] += deltas[l+1];
        delta_W[l] +=  xt::linalg::dot(xt::view(deltas[l+1], xt::all(), xt::newaxis()),xt::transpose(xt::view(H[l], xt::all(), xt::newaxis())));

        if(l > 1){
            deltas[l] = xt::linalg::dot(xt::transpose(W[l]),deltas[l+1]) * vecf(Z[l]);
        }
    }
}

void Network::gradient_descent(){
    for (int l =1; l < layers; l++){
        W[l] +=  -(alpha * 1.0 / batch_size ) * delta_W[l];
        B[l] +=  -(alpha * 1.0 / batch_size ) * delta_B[l];
    }
}

xt::xarray<double> Network::vectorised_label(int y){
    xt::xarray<double> lab = xt::zeros<double>({10});
    lab[y] = 1.0;
    return lab;
}

std::tuple<xt::xarray<double>,xt::xarray<double>> Network::batch(int k){
    
    xt::xarray<double> batch_labels = xt::zeros<double>({batch_size});
    xt::xarray<double> batch_input = xt::zeros<double>({batch_size,784});
    
    for(int i=0; i < batch_size; i++){
        batch_labels[i] = labels[k + i];
        for (int j=0; j < 784; j++) {
            batch_input[i * 784 + j]  = input_data[(i + k) * 784 + j];
        }
    }            
    return {batch_input, batch_labels};
}

double Network::batch_accuracy(xt::xarray<double> batch_inputs, xt::xarray<double> batch_labels){
    double accuracy = 0.0;
    xt::xarray<double> image;

    for (int i=0; i < batch_size; i++) {
        image = xt::row(batch_inputs,i);
        feedforward(image);

        int predicted = xt::argmax(H[layers])[0];
        if (predicted == batch_labels[i]) {
            accuracy += 1.0;
        }
    }

    accuracy /= batch_size;
    accuracy *= 100.0;
    return accuracy;
}

void Network::train(xt::xarray<double> x, xt::xarray<double> y) {
    initialise_data(x,y);
    initialise_weights();

    int k = 0;
    xt::xarray<double> image;

    for (int i=0; i< max_iter; i++) {
        initialise_deltas();
        k = (i*batch_size) % labels.size();
        auto [input_batch, labels_batch] = batch(k);

        for (int j=0; j < batch_size; j++) {
            image = xt::row(input_batch,j);
            feedforward(image);
            backprop(labels_batch[j]);
        }
                
        gradient_descent();

        if (i % 10 == 0){
            auto accuracy = batch_accuracy(input_batch,labels_batch);
            std::cout << "Iteration " << i << " accuracy of " << accuracy << "%\n";
        }
    }

    evaluate();
}

void Network::evaluate() {
    double accuracy = 0.0;
    xt::xarray<double> image;

    for (int i=0; i < labels.size(); i++) {
        image = xt::row(input_data,i);
        feedforward(image);

        int predicted = xt::argmax(H[layers])[0];            
        if (predicted == labels[i]) {
            accuracy += 1.0;
        }
    }

    accuracy /= labels.size();
    accuracy *= 100.0;
    std::cout << "Accuracy of " << accuracy << "%\n";
}
