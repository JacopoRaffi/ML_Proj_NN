from ABCNeuron import ABCNeuron
from InputNeuron import InputNeuron
from HiddenNeuron import HiddenNeuron
from OutputNeuron import OutputNeuron
#from ActivationFunctions import ActivationFunctions
import ActivationFunctions
import ErrorFunctions

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import json
import math


# TODO: vedere cosa trra predecessori e successori va rimosso, sia nel dizionario di input che nei neuroni per semplificare
# TODO: nesterov momentum

# TODO: nel notebook test_simple l'errore di validazione è minore di quello di training, ha senso?

# TODO: dalle slide
'''Note that often the bias w0
is omitted from the regularizer (because
its inclusion causes the results to be not independent from target
shift/scaling) or it may be included but with its own regularization
coefficient (see Bishop book, Hastie et al. book)
'''

# TODO: dalle slide
'''For on-line/mini-batch take care of possible effects over many
steps (patterns/examples): hence to compare w.r.t. batch version in a
fair way do not force equal lambda but it would be better to use
            λ x (mb /#total patterns l )
Of course if you choose lambda by model selection they will automatically select
different lambda for on-line and batch (or any mini-batch)'''


class NeuralNetwork:
    '''
    Implementation of the neural network that includes neurons and all method to use and train the NN
    
    Attributes
    ----------
    input_size: int
        the number of input units of the network
    output_size: int
        the number of output units of the network
    neurons: list of ABCNeuron
        the list of neurons, sorted in topological order, composing the NN
    
    '''

    def display_topology(topology):
        '''
        Function to vsualize a simple topology, if there are more then one hidden layer, the visualization is bad...
        '''
        G = {}
        layer = {}
        for key in topology.keys():
            if not topology[key][0] in layer:
                layer[topology[key][0]] = len(layer.keys())

        for key in topology:
            G[key] = []
            for i in topology[key][3]:
                G[key].append(i)
        
        G = nx.DiGraph(G)
        # Assegnazione delle parti ai nodi
        for node, data in G.nodes(data=True):
            data['subset'] = layer[topology[node][0]]

        # Calcolo del layout multipartito
        pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')

        # Creazione della visualizzazione del grafo multipartito
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_weight='bold', arrows=True)

        # Mostra il grafo
        plt.show()

    def toJSON(self):
        str_js = ''
        attributes = vars(self)
        for attr in attributes:
            str_js += json.dumps(attributes[attr])
        return str_js
    
    def __get_function_from_string(self, name:str):
        '''
        Map the function name to the corresponding callable variable
        
        param name: the name of the function to map

        return: the function corrisponding to the name as a callable
        '''
        '''
        if name == 'identity':
            fun = ActivationFunctions.identity

        elif name == 'sigmoid':
            fun = ActivationFunctions.sigmoid

        elif name == 'tanh':
            fun = ActivationFunctions.tanh

        elif name == 'softplus':
            fun = ActivationFunctions.softplus

        elif name == 'gaussian':
            fun = ActivationFunctions.gaussian

        elif name == 'gaussian':
            fun = ActivationFunctions.gaussian
        else:
            raise ValueError(f"Activation function {name} not found")'''

        return getattr(ActivationFunctions, name)

    def __init_weights(self, units:list):
        # All Neurons weights vectors are initialised
        for neuron in units:
            if neuron.type == 'output':
                #neuron.initialise_weights(rand_range_min, rand_range_max, False, random_generator)
                neuron.initialise_weights(self.rand_range_min, self.rand_range_max, False, self.random_generator)
            if neuron.type == 'hidden':
                neuron.initialise_weights(self.rand_range_min, self.rand_range_max, self.fan_in, self.random_generator)

    def __construct_from_dict(self, topology:dict):
        '''
        Builds a Neural Network of ABCNeuron's objects from the topology
        
        param topology: the graph structure described by a dictionary (see __init__ comments)
        param rand_range_min: minimum value for random weights initialisation range
        param rand_range_max: maximum value for random weights initialisation range
        param fan_in: if the weights initialisation should also consider the Neuron's fan-in

        return: the list of Neurons that compose the Neural Network
        '''
        units = []
        unit_type = ''
        unit_activation_function = ''
        unit_activation_function_args = []
        unit_successors = []
        unit_index = 0
        
        # All Neurons are initialised without synapses (successors/predecessors dependencies)
        for node in topology:
            unit_index = int(node) # TODO qua se i neuroni sono lettere scoppia? se si basta cambiarlo in modo che dia lui un numero ad ogni neurone e via
            # TODO non solo, devono partire da 0 ed essere gli indici dell'array... assunzioni un po traballanti...
            unit_type = topology[node][0]
            unit_activation_function = topology[node][1]
            unit_activation_function_args = [np.float64(a) for a in topology[node][2]]
            
            if unit_type.startswith('input'):
                self.input_size += 1
                units.append(InputNeuron(unit_index))
                
            elif unit_type.startswith('hidden'):
                units.append(HiddenNeuron(unit_index, self.__get_function_from_string(unit_activation_function), *unit_activation_function_args))
                
            elif unit_type.startswith('output'): # Fan-in is fixed as False for output units so to prevent Delta (Backpropagation Error Signal) to be a low value 
                self.output_size += 1
                units.append(OutputNeuron(unit_index, self.__get_function_from_string(unit_activation_function), *unit_activation_function_args))
            
        # All Neurons dependecies of successors and predecessors are filled inside the objects
        for node in topology:
            unit_type = topology[node][0]
            
            if not unit_type.startswith('output'): # Output units have no successors
                unit_successors = [units[u] for u in topology[node][3]]
                units[node].extend_successors(unit_successors)
        
        # All Neurons weights vectors are initialised
        self.__init_weights(units)

        return units
    
    def __topological_sort_util(self, index:int, visited:list, ordered:list):
        '''
        Recursive function that builds the topological(inverse) order updating ordered list 
        
        param: index: the index of the neuron to visit
        param: visited: the list of visited neurons
        param: ordered: the list of ordered neurons (inverse topological order)

        return: -
        '''
        visited[index] = True
        
        if self.neurons[index].type != 'output':
            for succ in self.neurons[index].successors:
                if not visited[succ.index]:
                    self.__topological_sort_util(succ.index, visited, ordered)
        
        ordered.append(self.neurons[index])
    
    def __topological_sort(self):
        '''
        Sort on a topological order the neurons of the Neural Network
        
        param: -

        return: -
        '''

        visited = [False]*self.n_neurons
        ordered = []
        
        for i in range(self.n_neurons):
            if not visited[i]:
                self.__topological_sort_util(i, visited, ordered)
        
        self.neurons = ordered[::-1]
                
    def __init__(self, topology:dict = {}, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = True, random_state:int=None):
        '''
        Neural Network inizialization
        
        param topology: the graph structure is described by a dictionary that has a key for each unit in the network, 
            and for each key contains a list of unit type (input, hidden, output), activation function, parameters of activation functions
            and list of nodes where an outgoing arc terminates.
            
            eg: {'0': ['input', 'None', [], ['2', '3', '4']], 
                 '1': ['input', 'None', [], ['2', '3', '4']],
                 '2': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '3': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '4': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '5': ['output', 'identity', [fun_args...], []],
                 '6': ['output', 'identity', [fun_args...], []]}

        param: rand_range_min: minimum value for random weights initialisation range
        param: rand_range_max: maximum value for random weights initialisation range
        param: fan_in: if the weights'initialisation should also consider the Neuron's fan-in

        return: -
        '''

        self.input_size = 0
        self.output_size = 0

        self.random_generator = np.random.default_rng(random_state)
        self.rand_range_min = rand_range_min
        self.rand_range_max = rand_range_max
        self.fan_in = self.fan_in
        self.neurons = self.__construct_from_dict(topology)
        self.n_neurons = len(self.neurons)
        self.__topological_sort() # The NN keeps its neurons in topological order
    
    def __str__(self):
        '''
        Return a string that describe the internal state of the neural network
        
        return: the description of the internal rapresentation
        '''
        attributes = vars(self)  # Get a dictionary of instance attributes
        attributes_str = "\n".join([f"{key}: {str(value)}" for key, value in attributes.items()])
        return f"Instance Attributes:\n{attributes_str}"

    def predict(self, input:np.array):
        '''
        Compute the output of the network given an input vector
        
        :param input: the input vector for the model
        :return: the network's output vector
        '''
        output_vector = np.zeros(self.output_size)
        
        # The input is forwarded towards the input units
        for feature_index, neuron in enumerate(self.neurons[:self.input_size]):
            neuron.forward(input[feature_index])
        
        # The hidden units will now take their predecessors results forwarding the signal throught the network
        for neuron in self.neurons[self.input_size:-self.output_size]:
            neuron.forward()
            
        # The output units will now take their predecessors results producing (returning) teh network's output
        for feature_index, neuron in enumerate(self.neurons[-self.output_size:]):
            output_vector[feature_index] = neuron.forward()
            
        return output_vector
    
    def predict_array(self, input:np.array):

        output = np.empty((len(input), self.output_size))
        for i, el in enumerate(input):
            output[i] = self.predict(el)
        return output
          
    def __backpropagation(self, target:np.array):
        '''
        Compute the Backpropagation training algorithm on the NN for a single training pattern
        
        :param target: the target vector for the Backprogation iteration
        :return: -
        '''

        # The error is calculated in the ouput units and propagated backwards
        for i, neuron in enumerate(reversed(self.neurons[-self.output_size:])):
            neuron.backward(target[-(i+1)])
        # The hidden units will now calculate their errors based on the signal propagated by their successors in the nn
        for neuron in reversed(self.neurons[self.input_size:-self.output_size]):
            neuron.backward()
        
        # The error is calculated in the ouput units and propagated backwards
        #target_index = len(target) - 1
        #while nn_neuron_index >= self.n_neurons-self.output_size:
            #print("Back Index:", self.neurons[nn_neuron_index].index)
            #self.neurons[nn_neuron_index].backward(target[target_index])
            #target_index -= 1
            #nn_neuron_index -= 1
        
        # The hidden units will now calculate their errors based on the signal propagated by their successors in the nn
        #while nn_neuron_index >= self.input_size:
            #print("Back Index:", self.neurons[nn_neuron_index].index)
            #self.neurons[nn_neuron_index].backward()
            #nn_neuron_index -= 1
    
    def ho_train(self, 
              training_set:np.ndarray, 
              validation_set:np.ndarray, 
              batch_size:int, 
              max_epochs:int = 1024, 
              error_decrease_tolerance:float = 0.0001, 
              patience: int = 8, 
              min_epochs: int = 0,
              learning_rate:float = 0.01, 
              lambda_tikhonov:float = 0.0, 
              alpha_momentum:float = 0.0, 
              metrics:list=[], 
              collect_data:bool=True, 
              collect_data_batch:bool=False, 
              verbose:bool=True):
        '''
        Compute the Backpropagation training algorithm on the NN for given training samples and hyperparameters
        
        param training_set: a set of samples (pattern-target pairs) for supervised learning
        param batch_size: parameter which determines the amount of training samples consumed in each iteration of the algorithm
            -> 1: Online
            -> 1 < batch_size < len(TR): Minibatch with minibatch size equals to batch_size
            -> len(TR): Batch
        param max_epochs: the maximum number of epochs (consumption of the whole training set) on which the algorithm will iterate
        param error_function: a string indicating the error function that the algorithm whould exploit when calculating the error distances between iterations
            -> "mee": Mean Euclidean Error
            -> "lms": Least Mean Square
        param error_decrease_tolerance: the errors difference (gain) value that the algorithm should consider as sufficiently low in order to stop training 
        param patience: the number of epochs to wait when a "no more significant error decrease" occurs
        param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        param lambda_tikhonov: Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        param alpha_momentum: Momentum Hyperparameter

        return: -
        '''

        def circular_index(a, start, stop): 
            '''
            start is included
            stop is NOT included
            '''
            # fn to convert your start stop to a wrapped range
            if stop<=start:
                stop += len(a)
            return np.arange(start, stop)%len(a)

        #TODO Controllare i valori e capire se sono permessi (ad esempio batch_size > 0 ecc...)

        # TODO: magari creare una nuova funzione train pubblica e rendere questa 'privata' per gestire meglio gli input
        # TODO: il learning rate deve dipendere dalla dimensione della batch!!!!
        epochs = 0
        exhausting_patience = patience
        last_error_decrease_percentage = 1
        last_error = 0
        new_error = 0
        training_set_length = len(training_set)
        batch_index = 0
        if batch_size > training_set_length: batch_size = training_set_length


        # initializing the dict where collect stats of the training
        if collect_data:
            stats = {
                # -- input stats --
                'training_set_len':training_set_length,
                'minibatch_size':batch_size,
                'max_epochs':max_epochs,
                'error_decrease_tolerance':error_decrease_tolerance,
                'patience':patience,
                'min_epochs':min_epochs,
                'learning_rate':learning_rate,
                'lambda_tikhonov':lambda_tikhonov,
                'alpha_momentum':alpha_momentum,

                # -- training stats --
                # epoch stats
                'epochs':0,
                'total_train_time': datetime.datetime.now() - datetime.datetime.now(),
                'mean_epoch_train_time':0,
                'units_weights' : {},
                
                # batch stats
                'units_weights_batch' : {}
            }
            for mes in metrics:
                # epoch stats
                stats['training_' + mes.__name__] = []
                stats['validation_' + mes.__name__] = []
                # batch stats
                stats['training_batch_' + mes.__name__] = []
                stats['validation_batch_' + mes.__name__] = []

            for unit in self.neurons[self.input_size:]:
                # epoch stats
                stats['units_weights'][unit.index] = []
                # batch stats
                stats['units_weights_batch'][unit.index] = []
        
        # print some information
        if verbose: print('starting values: ', stats)
        if collect_data: # take training time for the batch
            start_time = datetime.datetime.now()

        # start training cycle
        while epochs < max_epochs and exhausting_patience > 0:
            
            # batch
            for sample in training_set[circular_index(training_set, batch_index, (batch_index + batch_size) % training_set_length)]:
                self.predict(sample[:self.input_size])
                self.__backpropagation(sample[self.input_size:])

            for neuron in self.neurons[self.input_size:]:
                neuron.update_weights(learning_rate, lambda_tikhonov, alpha_momentum)

            # stats for every batch
            if collect_data and collect_data_batch and epochs > 0: # to avoid stupidly high starting values
                for mes in metrics:
                    stats['training_batch_' + mes.__name__].append(mes(self.predict_array(training_set[:,:self.input_size]), training_set[:,self.input_size:]))
                    stats['validation_batch_' + mes.__name__].append(mes(self.predict_array(validation_set[:,:self.input_size]), validation_set[:,self.input_size:]))
                for unit in self.neurons[self.input_size:]:
                    stats['units_weights_batch'][unit.index].append(unit.w.copy())

            #TODO: per implementare altri errori dobbiamo cambiare la back prop
            #TODO: stiamo facendo il controllo sull'errore dell'iterazione prima, non good
            batch_index += batch_size
            if batch_index >= training_set_length:
                if epochs > min_epochs and last_error_decrease_percentage <= error_decrease_tolerance:
                    exhausting_patience -= 1
                else:
                    exhausting_patience = patience

                epochs += 1
                batch_index = batch_index%training_set_length

                new_error = ErrorFunctions.mean_squared_error(self.predict_array(training_set[:,:self.input_size]), training_set[:,self.input_size:])
                if last_error != 0:
                    last_error_decrease_percentage = abs(last_error - new_error)/last_error    
                last_error = new_error

                # stats for every epoch
                if collect_data:
                    # take training time for the epoch
                    end_time = datetime.datetime.now()   

                    if collect_data:
                        stats['total_train_time'] += end_time-start_time

                    if verbose: metrics_to_print = ''
                    for mes in metrics:
                        
                        tr_err = mes(self.predict_array(training_set[:,:self.input_size]), training_set[:,self.input_size:])
                        stats['training_' + mes.__name__].append(tr_err)
                        val_err = mes(self.predict_array(validation_set[:,:self.input_size]), validation_set[:,self.input_size:])
                        stats['validation_' + mes.__name__].append(val_err)

                        if verbose: metrics_to_print += '| ' +mes.__name__ + ': tr=' + str(tr_err) + ' val=' + str(val_err) + ' | '
                    for unit in self.neurons[self.input_size:]:
                        stats['units_weights'][unit.index].append(unit.w.copy())  

                    if verbose: print('[' + str(epochs) + '/' + str(max_epochs) + '] tr time:', end_time-start_time, metrics_to_print)
                    # take training time for the batch
                    start_time = datetime.datetime.now()

        # final stats gathering
        if collect_data:
            stats['epochs'] = epochs
            stats['mean_epoch_train_time'] = stats['total_train_time']/stats['epochs']
            return stats

    def kf_train(self, 
              data_set:np.ndarray, 
              k:int,
              batch_size:int,
              max_epochs:int = 1024, 
              error_decrease_tolerance:float = 0.0001, 
              patience: int = 8, 
              min_epochs: int = 0,
              learning_rate:float = 0.01, 
              lambda_tikhonov:float = 0.0, 
              alpha_momentum:float = 0.0, 
              metrics:list=[], 
              collect_data:bool=True, 
              collect_data_batch:bool=False, 
              verbose:bool=False):
        '''
        Compute the Backpropagation training algorithm on the NN for given a data set, estimating the hyperparameter performances
        trough validation in k folds of the data
        
        param data_set: a set of samples (pattern-target pairs) for supervised learning
        param k: number of data folds (data splits)
        param batch_size: parameter which determines the amount of training samples consumed in each iteration of the algorithm
            -> 1: Online
            -> 1 < batch_size < len(TR): Minibatch with minibatch size equals to batch_size
            -> len(TR): Batch
        param max_epochs: the maximum number of epochs (consumption of the whole training set) on which the algorithm will iterate
        param error_function: a string indicating the error function that the algorithm whould exploit when calculating the error distances between iterations
            -> "mee": Mean Euclidean Error
            -> "lms": Least Mean Square
        param error_decrease_tolerance: the errors difference (gain) value that the algorithm should consider as sufficiently low in order to stop training 
        param patience: the number of epochs to wait when a "no more significant error decrease" occurs
        param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        param lambda_tikhonov: Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        param alpha_momentum: Momentum Hyperparameter

        return: mean and variance of the validation error
        '''
        
        # Computation of the size of each split
        data_len = len(data_set)
        split_size = math.floor(data_len/k)
        
        val_errors = np.empty(k)
        stats = {}
        split_index = 0
        training_set = np.append(data_set[:split_index], data_set[split_index + split_size:], axis=0)
        validation_set = data_set[split_index : split_index + split_size]
        
        
        # At each iteration only one of the K subsets of data is used as the validation set, 
        # while all others are used for training the model validated on it.
        for i in range(k):
        
            stats = self.ho_train(training_set, validation_set, batch_size, max_epochs, error_decrease_tolerance, patience, min_epochs, 
                   learning_rate, lambda_tikhonov, alpha_momentum, metrics, collect_data, collect_data_batch, verbose)
            
            val_errors[i] = stats['validation_mean_squared_error'][-1]
            
            if i != k-1:
                split_index += split_size
                self.__init_weights(self.neurons) # reset of the network to proceeds towards the next training (next fold)
                training_set = np.append(data_set[:split_index], data_set[split_index + split_size:], axis=0)
                validation_set = data_set[split_index : split_index + split_size]
                
        return np.mean(val_errors), np.var(val_errors)
        
        
if __name__ == '__main__':
    def create_dataset(n_items, n_input, input_range, output_functions, seed):
        random.seed(seed)

        n_output = len(output_functions)
        x = np.ndarray((n_items, n_input + n_output))

        for i in range(n_items):
            for l in range(n_input):
                x[i,l] = random.randrange(input_range[0], input_range[1], input_range[2])

            for l, fun in enumerate(output_functions):
                
                x[i, n_input + l] = fun(x[i][:n_input])
                #print(x[i][:n_input], fun(x[i][:n_input]), x[i, l])

        return pd.DataFrame(x, columns = ['input_' + str(i + 1) for i in range(n_input)] + ['output_' + str(i + 1) for i in range(n_output)])


    topology_2 = {'0': ['input', 'None', [], ['2', '3']],
                '1': ['input', 'None', [], ['2', '3']], 
                '2': ['output', 'identity', ['0', '1'], []],
                '3': ['output', 'identity', ['0', '1'], []]}

    RANDOM_STATE = 12345
    NN = NeuralNetwork(topology_2, -0.75, 0.75, True, RANDOM_STATE)

    tr_1_input = 2
    tr_1_output = 2
    len_dataset = 500
    tr_1 = create_dataset(len_dataset, tr_1_input, [0, 100, 1], [lambda x : 2*x[0] + x[1] + 4, lambda x : -2*x[0] + x[1] + 1], RANDOM_STATE)

    training_set = tr_1.values

    batch_size = 20
    max_epochs = 250
    error_decrease_tolerance = 0.0001
    patience = 20

    learning_rate = 0.0001/batch_size
    lambda_tikhonov = 0
    alpha_momentum = 0

    stats = NN.train_2(training_set, batch_size, max_epochs, error_decrease_tolerance, patience, 
                    learning_rate, lambda_tikhonov, alpha_momentum)
    predictions = NN.predict_array(training_set)