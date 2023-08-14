import numpy as np
from  mnist_loader import *
import json

class NeuralNet:

    def __init__(self) :

        self.nn_structure = [784,16,16,10]

        self.layers = len(self.nn_structure)

        self.W = {} # Weights set up as layer: weight matrix,   wji  
        self.b = {} # Biases set up as layer: bias vector

        self.h = {} # Activations
        self.z = {} 

        self.delta_W = {} # dictionary which has delta wji matrix for each layer
        self.delta_b = {} # dictionary which has delta b vectors for each layer

        self.batch_size = 1000
        self.training_data = None
        self.input_data = None
        self.labels = None

        self.max_iter = 2000
        self.alpha = 2.5

    def initialise_data(self,training_data):

        self.training_data = training_data
        self.input_data = training_data[0]
        self.labels = training_data[1]


    def initialise_weights(self):

        for l in range(1,self.layers):
            # randn used to generate numbers between -0.5 and 0.5 for the weights and biases
            self.W[l] = np.random.randn(self.nn_structure[l], self.nn_structure[l-1] ) #Creates weight matrix in each layer
            self.b[l] = np.random.randn(self.nn_structure[l]) #Creates bias vector in each layer

            
    def initialise_deltas(self):

        for l in range(1,self.layers):
            self.delta_W[l] = np.zeros((self.nn_structure[l], self.nn_structure[l-1] ))
            self.delta_b[l] = np.zeros(self.nn_structure[l])


    def give_vectorised_label(self,y):
        """Give a vectorised label"""

        lab = np.zeros(10)
        lab[y] = 1.0
        return lab

    def give_batch(self,k):
        
        return self.input_data[k:k+self.batch_size] , self.labels[k:k+self.batch_size]

    def reshuffle(self):

        randomize = np.arange(len(self.input_data))
        np.random.shuffle(randomize)
        self.input_data = self.input_data[randomize]
        self.labels = self.labels[randomize]


            
    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))


    def sigmoid_gradient(self,x):
        f = self.sigmoid(x)
        return  (1.0 - f)*f


    def feed_forward(self,x) :

        self.h[1] = x

        for l in range(1,len(self.nn_structure)):
            z_l =  np.matmul(self.W[l],self.h[l]) + self.b[l] # W h + b = z

            
            self.h[l+1] = self.sigmoid(z_l)  # f(z) = h
        
            self.z[l+1] = z_l

        #self.h[self.layers] = z_l
        #self.softmax_output_layer()
        
        self.normalise_output_layer()



    def softmax_output_layer(self):
        '''Use instead of sigmoid activation layer and normalisation'''
        self.h[self.layers] = np.exp(self.h[self.layers]) 

        factor = sum(self.h[self.layers])
        self.h[self.layers] /= factor      

    def normalise_output_layer(self):
        factor = sum(self.h[self.layers])
        self.h[self.layers] /= factor   


    def cost_function(self,x,y):
        """Cost for a single image"""

        self.feed_forward(x)
        return np.linalg.norm(y-self.h[self.layers])

    def batch_cost(self,input_batch,labels_batch):
        """Cost of a batch of images"""

        c = 0
        for x,y in zip(input_batch,labels_batch):

            y = self.give_vectorised_label(y)
            c += self.cost_function(x,y)
            
        c /= len(labels_batch)
        return c

    def entire_cost(self):
        c = 0
        for x,y in zip(self.input_data,self.labels):

            y = self.give_vectorised_label(y)
            c += self.cost_function(x,y)
            
        c /= len(self.labels)
        return c


    def cost_derivative(self,y) :

        return self.h[self.layers] - y



    def backprop(self,y) :

        y = self.give_vectorised_label(y)

        cost_grad = self.cost_derivative(y)

        output_deltas = cost_grad * self.sigmoid_gradient(self.z[self.layers]) # delta = dC/dh_l * dh_l/dz_l  

        deltas = {self.layers:output_deltas} 

        
        for l in range(self.layers-1, 0, -1):
            
            # Chain rule derivative dC/db_l = delta
            self.delta_b[l] += deltas[l+1]  

            # Chain rule derivative dC/dW = delta * h_l
            self.delta_W[l] += np.dot(deltas[l+1][:,np.newaxis],np.transpose(self.h[l][:,np.newaxis])) 
            
            #Backpropagation for deltas across different layers
            if l > 1: deltas[l] = np.dot(np.transpose(self.W[l]),deltas[l+1]) * self.sigmoid_gradient(self.z[l]) 
                


    def gradient_descent(self):
        
        m = self.batch_size
        
        for l in range(1,self.layers):
            self.W[l] +=  -1.0/m * self.alpha * self.delta_W[l] 
            self.b[l] +=  -1.0/m * self.alpha * self.delta_b[l] 
        
            
    def train(self,training_data):

        self.initialise_data(training_data)
        self.initialise_weights()
        #self.load_from_json()

        num_batches = len(input_data) / self.batch_size

        for i in range(self.max_iter):

            if i % num_batches == 0: self.reshuffle()

            self.initialise_deltas()


            k = (i*self.batch_size) % len(input_data)

            input_batch,labels_batch = self.give_batch(k)
        
            for x,y in zip(input_batch,labels_batch):

                self.feed_forward(x)
                self.backprop(y)            
    
            

            self.gradient_descent()

            c = self.batch_cost(input_batch,labels_batch)
        
            if i%10 == 0: print(f'Cost {c.round(8)} for {i}')


    def save_to_json(self):
        """Save weights and biases after training as json file"""

        weights = {key:value.tolist() for key,value in self.W.items()}
        biases = {key:value.tolist() for key,value in self.b.items()}

        with open('../data/weights.json', 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=4)
        
        with open('../data/biases.json', 'w', encoding='utf-8') as f:
            json.dump(biases, f, ensure_ascii=False, indent=4)

        print(f'Weights and biases saved as json files!')

    def load_from_json(self):
        """Load weights from json file"""

        with open('../data/weights.json') as w:
            weights = json.load(w)
        

        with open('../data/biases.json') as b:
            biases = json.load(b)

        self.W = {int(key):np.array(value) for key,value in weights.items()}
        self.b = {int(key):np.array(value) for key,value in biases.items()}


    def evaluate_training(self):
        accuracy = 0
        
        i = 0
        for x,y in zip(self.input_data,self.labels):
            self.feed_forward(x)
            soft_output = self.h[self.layers]
            prediction = list(soft_output).index(max(soft_output))

            if prediction == y : accuracy += 1
            i+=1
        print(f'accuracy was {100.0 * accuracy / len(self.input_data)} %')
        return accuracy / len(self.input_data)
          
    def evaluate(self,test_data):
        input_data , labels = test_data[0] , test_data[1]

        accuracy = 0
        i = 0
        for x,y in zip(input_data,labels):
            self.feed_forward(x)
            soft_output = self.h[self.layers]
            prediction = list(soft_output).index(max(soft_output))

            if prediction == y : accuracy += 1
            i+=1
            

        print(f'accuracy was {100.0 * accuracy / len(input_data)} %')
        return accuracy / len(input_data)

    

            

        



if __name__ == '__main__':
    training_data, validation_data, test_data = load_data()

    input_data = test_data[0]
    labels = test_data[1]
    

    neural_net = NeuralNet()


    neural_net.train(training_data)
    neural_net.evaluate_training()


    neural_net.save_to_json()

    
    
    
