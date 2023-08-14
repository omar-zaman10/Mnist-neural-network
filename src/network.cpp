#include "weights_loader.cpp"
#include <list>
#include <cmath>
#include <xtensor/xvectorize.hpp>
#include <vector>
#include <xtensor/xrandom.hpp>
#include "data_loader.cpp"
#include <typeinfo>
#include <xtensor/xsort.hpp>


class network{
    private:

        vector<int> nn_structure = {784,16,16,10};

        int layers ;

        unordered_map<int, xt::xarray<double> > W;
        unordered_map<int, xt::xarray<double> > B;

        unordered_map<int, xt::xarray<double> > H;
        unordered_map<int, xt::xarray<double> > Z;

        unordered_map<int, xt::xarray<double> > delta_W;
        unordered_map<int, xt::xarray<double> > delta_B;

        xt::xarray<double> training_data;
        xt::xarray<double> input_data;
        xt::xarray<double> labels;

        int batch_size = 500;
        int max_iter = 2000;
        double alpha = 1;


    public:
        network(){
            layers = size(nn_structure);
            
        }

        void set_weights_and_biases(unordered_map<int, xt::xarray<double> > biases,unordered_map<int, xt::xarray<double> > weights){
            B = biases;
            W = weights;
        
        }

        void initialise_weights(){
            for(int l =1; l < layers; l++){



                W[l] = xt::random::randn<double>({nn_structure[l], nn_structure[l-1]});
                B[l] = xt::random::randn<double>({nn_structure[l]});


            }
        
            

        }
    
        void initialise_deltas(){
            for(int l =1; l < layers; l++){
                delta_W[l] = xt::zeros<double>({W[l].shape(0), W[l].shape(1)});
                delta_B[l] = xt::zeros<double>({B[l].shape(0)});


            }
            

        }

        void initialise_data(xt::xarray<double> x, xt::xarray<double> y){

            input_data = x;
            labels = y;

        }
    
        static double sigmoid(double z){
            return 1.0 / (1.0 + exp(-z));}

        static double sigmoid_gradient(double z){
            double f = sigmoid(z);
            return f*(1-f) ;
        }

        void feedforward(xt::xarray<double> x){

            H[1] = x;

            xt::xarray<double>  z_l;
            auto vecf = xt::vectorize(&network::sigmoid);

            for(int l =1; l < layers; l++){

                z_l = xt::linalg::dot(W[l], H[l]) + B[l];

                H[l+1] = vecf(z_l);            

                Z[l+1] = z_l;

            }
            H[layers] = H[layers] / xt::sum(H[layers]);


        }

        xt::xarray<double> cost_derivative(xt::xarray<double> y){
            xt::xarray<double> deriv = H[layers] - y;
        
            return deriv;

        }

        void backprop(double y){

            xt::xarray<double> vect_label = vectorised_label(y);

            xt::xarray<double> cost_grad = cost_derivative(vect_label);

            auto vecf = xt::vectorize(&network::sigmoid_gradient);
            
            xt::xarray<double> output_deltas = cost_grad * vecf(Z[layers]);

            unordered_map<int, xt::xarray<double> > deltas;

            deltas[layers] = output_deltas;

            for(int l= layers-1; l >0; l-- ){

                delta_B[l] += deltas[l+1];

                delta_W[l] +=  xt::linalg::dot(xt::view(deltas[l+1], xt::all(), xt::newaxis()),xt::transpose(xt::view(H[l], xt::all(), xt::newaxis())));

                if(l > 1){

                    deltas[l] = xt::linalg::dot(xt::transpose(W[l]),deltas[l+1]) * vecf(Z[l]);
                }

            }

            

        }

        void gradient_descent(){

            for(int l =1; l < layers; l++){

                W[l] +=  -(alpha * 1.0 / batch_size ) * delta_W[l];
                B[l] +=  -(alpha * 1.0 / batch_size ) * delta_B[l];


            }


        }
 
        void train(xt::xarray<double> x, xt::xarray<double> y){

            initialise_data(x,y);
            initialise_weights();
            
            int k;
            xt::xarray<double> image;

            for(int i=0; i< max_iter; i++){

                initialise_deltas();
                
                k = (i*batch_size) % labels.size();
                auto [input_batch, labels_batch] = batch(k);

                for(int j=0; j < batch_size; j++){
                    
                    image = xt::row(input_batch,j);
                    feedforward(image);
                    backprop(labels_batch[j]);

                }
                        
            gradient_descent();

            if( i %10 == 0){
                cout << "Iteration " << i << " ";
                batch_accuracy(input_batch,labels_batch);
            }
            

            }

            evaluate();


        }








        xt::xarray<double> vectorised_label(int y){

            xt::xarray<double> lab = xt::zeros<double>({10});
            lab[y] = 1.0;

            return lab;
        }

        tuple<xt::xarray<double>,xt::xarray<double>>  batch(int k){
            
            xt::xarray<double> batch_labels = xt::zeros<double>({batch_size});
            xt::xarray<double> batch_input = xt::zeros<double>({batch_size,784});
            
            for(int i=0; i < batch_size; i++){

                batch_labels[i] = labels[k+i];

                for(int j=0; j <784; j++){
                    batch_input[i*784 + j]  = input_data[(i+k)*784 + j];

                }

                
            }            


             return {batch_input, batch_labels};
        }

        void batch_accuracy(xt::xarray<double> batch_inputs, xt::xarray<double> batch_labels){

            double accuracy = 0.0;
            xt::xarray<double> image;

            for(int i=0; i < batch_size; i++){
                
                image = xt::row(batch_inputs,i);

                feedforward(image);

                int predicted = xt::argmax(H[layers])[0];

                
                if(predicted == batch_labels[i]){
                    accuracy += 1.0;}

            }

            accuracy /= batch_size;
            accuracy *= 100.0;

            cout << "Accuracy of " << accuracy << "%" << endl;

        }


        void evaluate(){

            double accuracy = 0.0;
            xt::xarray<double> image;

            for(int i=0; i < labels.size(); i++){
                
                image = xt::row(input_data,i);

                feedforward(image);

                int predicted = xt::argmax(H[layers])[0];

                
                if(predicted == labels[i]){
                    accuracy += 1.0;}

            }

            accuracy /= labels.size();
            accuracy *= 100.0;

            cout << "Accuracy of " << accuracy << "%" << endl;

        }





};





int main(){

    unordered_map<int, xt::xarray<double> > biases = biases_loader();

    /*
    unordered_map<int, xt::xarray<double>>::iterator itr;
    for (itr = biases.begin(); 
       itr != biases.end(); itr++) {
    // itr works as a pointer to 
    // pair<string, double> type 
    // itr->first stores the key part and
    // itr->second stores the value part
    cout << itr->first << "  " << 
            itr->second << endl;
    }
    */

    unordered_map<int, xt::xarray<double>> weights =  weights_loader();
    
    network nn;

    auto [input_data, labels]  = JsonLoader();

    //nn.set_weights_and_biases(biases, weights); //comment out initialise_weights in train
    //nn.evaluate();

    nn.train(input_data,labels);

    return 0;
}