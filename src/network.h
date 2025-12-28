#include "weights_loader.h"
#include "data_loader.h"

#include <vector>
#include <unordered_map>

#include <xtensor/xarray.hpp>

class Network {
  public:
    Network() = default;

    void train(xt::xarray<double> x, xt::xarray<double> y);
    void evaluate();

  private:
    std::vector<int> nn_structure = {784,16,16,10};

    std::size_t layers = nn_structure.size();

    std::unordered_map<int, xt::xarray<double> > W;
    std::unordered_map<int, xt::xarray<double> > B;

    std::unordered_map<int, xt::xarray<double> > H;
    std::unordered_map<int, xt::xarray<double> > Z;

    std::unordered_map<int, xt::xarray<double> > delta_W;
    std::unordered_map<int, xt::xarray<double> > delta_B;

    xt::xarray<double> training_data;
    xt::xarray<double> input_data;
    xt::xarray<double> labels;

    int batch_size = 500;
    int max_iter = 2000;
    double alpha = 1.0;

    void set_weights_and_biases(
        std::unordered_map<int, xt::xarray<double>> biases,
        std::unordered_map<int, xt::xarray<double>> weights
    );

    void initialise_weights();
    void initialise_deltas();
    void initialise_data(xt::xarray<double> x, xt::xarray<double> y);

    static double sigmoid(double z);
    static double sigmoid_gradient(double z);
    xt::xarray<double> cost_derivative(xt::xarray<double> y);

    void feedforward(xt::xarray<double> x);
    void backprop(double y);
    void gradient_descent();

    xt::xarray<double> vectorised_label(int y);
    std::tuple<xt::xarray<double>,xt::xarray<double>> batch(int k);
    double batch_accuracy(xt::xarray<double> batch_inputs, xt::xarray<double> batch_labels);
};