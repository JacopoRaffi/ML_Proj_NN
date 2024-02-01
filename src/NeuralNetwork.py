from InputNeuron import InputNeuron
from HiddenNeuron import HiddenNeuron
from OutputNeuron import OutputNeuron
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
    
    input_stats = {
        'training_set_len',
        'minibatch_size',
        'max_epochs',
        'retrainig_es_error',
        'error_increase_tolerance',
        'patience',
        'min_epochs',
        'learning_rate',
        'lr_decay_tau',
        'eta_tau',
        'lambda_tikhonov',
        'alpha_momentum',
        'nesterov',
        
        'adamax',
        'adamax_learning_rate',
        'exp_decay_rate_1',
        'exp_decay_rate_2',
        }
    train_stats = {
        'epochs',
        'total_train_time',
        'mean_epoch_train_time',
        'units_weights',
        'units_weights_batch',
    }
    train_input = ['training_set', 
              'validation_set', 
              
              'batch_size', 
              'max_epochs', 
              'min_epochs',
              'retrainig_es_error',
              'patience', 
              'error_increase_tolerance', 
             
              'lambda_tikhonov',
              
              'adamax',
              'adamax_learning_rate',
              'exp_decay_rate_1',
              'exp_decay_rate_2',
              
              'learning_rate',
              'lr_decay_tau',
              'eta_tau',         
              'alpha_momentum', 
              'nesterov',
              
              'metrics', 
              'collect_data', 
              'collect_data_batch', 
              'verbose']
    
    inzialization_input = ['topology', 'range_min', 'range_max', 'fan_in', 'random_state']

    def display_topology(topology):
        '''
        Simple function to vsualize a topology
        
        topology: dict
            the topology to be displayed
            
                eg: {0: ['input_0', None, [], [10, 11, 12]], 
                    1: ['input_0', None, [], [10, 11, 12]], 
                    2: ['input_0', None, [], [10, 11, 12]], 
                    3: ['input_0', None, [], [10, 11, 12]], 
                    4: ['input_0', None, [], [10, 11, 12]], 
                    5: ['input_0', None, [], [10, 11, 12]], 
                    6: ['input_0', None, [], [10, 11, 12]], 
                    7: ['input_0', None, [], [10, 11, 12]], 
                    8: ['input_0', None, [], [10, 11, 12]], 
                    9: ['input_0', None, [], [10, 11, 12]], 
                    10: ['hidden_1', 'ReLU', [1], [13, 14, 15]], 
                    11: ['hidden_1', 'ReLU', [1], [13, 14, 15]], 
                    12: ['hidden_1', 'ReLU', [1], [13, 14, 15]], 
                    13: ['hidden_2', 'ReLU', [1], [16, 17, 18]], 
                    14: ['hidden_2', 'ReLU', [1], [16, 17, 18]], 
                    15: ['hidden_2', 'ReLU', [1], [16, 17, 18]], 
                    16: ['hidden_3', 'ReLU', [1], [19, 20, 21]], 
                    17: ['hidden_3', 'ReLU', [1], [19, 20, 21]], 
                    18: ['hidden_3', 'ReLU', [1], [19, 20, 21]], 
                    19: ['output_4', 'identity', [], []], 
                    20: ['output_4', 'identity', [], []], 
                    21: ['output_4', 'identity', [], []]}
                
                Every different layer (the first element of each list), is displayed in the correct layer
                The neuron must be number
                
                For creating layered topology use the function in MyUtils create_stratified_topology
        
        return: -
        
        
        '''
        # fist a adjacency Graph is created, then the layers are divided and displayed with networkx
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

        # compute the multipartited layer
        pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')

        # displaying the graph
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_weight='bold', arrows=True)
        plt.show()

    def fromJSON(json_str):
        '''
        Function used to restore a NeuralNetwork state (the weights) from a json string
        
        json_str: str
            the rapresentation of the state
            
        return: NeuralNetwork
            the created NN
        '''
        
        # first use json to get the original dict from the str
        a = json.loads(json_str)

        # then create a new NN and use the saved information to restore the original weights
        nn = NeuralNetwork(a['topology'])
        if isinstance(list(a.keys())[0], str):
            for n in nn.neurons[nn.input_size:]:
                n.w = np.array(a[str(n.index)])
        else:
            for n in nn.neurons[nn.input_size:]:
                n.w = np.array(a[n.index])
        
        # return the created NN
        return nn
        
    def toJSON(self):
        '''
        Function used to store a NeuralNetwork state (the weights) in a json string
        
        param: -
            
        return: str
            the string that represents the NN
        '''
        
        # first create the dict containing all the useful information
        save = {}
        save['topology'] = self.topology
        for neuron in self.neurons[self.input_size:]:
            save[neuron.index] = list(neuron.w)
        
        # then use json to create a json rappresentation of those
        return json.dumps(save)
    
    def reset(self):
        '''
        Reset all the neurons weights in the Network
        
        return: -
        '''
        self.__init_weights()

    def __get_function_from_string(self, name:str):
        '''
        Map the function name to the corresponding callable variable
        
        name: str
            the name of the function to map

        return: callable
            the function corrisponding to the name as a callable
        '''

        return getattr(ActivationFunctions, name)

    def __init_weights(self):
        '''
        Inizialize all the newly created neurons with the correct parameters stored in istance variables
        
        param: -

        return: -
        '''
        # All Neurons weights vectors are initialised
        for neuron in self.neurons:
            # if the neuron is an output unit, we don't want to use the fan-in to reduce the weights
            # also input newron do not contain weights
            if neuron.type == 'output':
                # Fan-in is fixed as False for output units so to prevent Delta (Backpropagation Error Signal) to be a low value 
                neuron.initialise_weights(self.rand_range_min, self.rand_range_max, False, self.random_generator)
            if neuron.type == 'hidden':
                neuron.initialise_weights(self.rand_range_min, self.rand_range_max, self.fan_in, self.random_generator)
                
    def __construct_from_dict(self, topology:dict):
        '''
        Builds a Neural Network from the topology
        
        topology: dict
            the graph structure described by a dictionary

        return: -
        '''
        
        # inzialization of useful variables
        units = [] 
        unit_type = ''
        unit_activation_function = ''
        unit_activation_function_args = []
        unit_successors = []
        unit_index = 0
        
        # All Neurons are initialised without synapses (successors/predecessors dependencies)
        for node in topology:
            
            
            unit_index = int(node) # save the neuron's index
            unit_type = topology[node][0] # save the neuron's type (input, output, hidden)
            unit_activation_function = topology[node][1] # save the activation_function
            unit_activation_function_args = [np.float64(a) for a in topology[node][2]] # save, if present, act_fun args
            
            # now every neuron type is inizializated separately
            if unit_type.startswith('input'):
                # increase input_size and create the neuron
                self.input_size += 1
                units.append(InputNeuron(unit_index))
                
            elif unit_type.startswith('hidden'):
                # create the hidden neuron with the correct activation function
                units.append(HiddenNeuron(unit_index, self.__get_function_from_string(unit_activation_function), *unit_activation_function_args))
                
            elif unit_type.startswith('output'): 
                # create the hidden neuron with the correct activation function and increase the output size
                self.output_size += 1
                units.append(OutputNeuron(unit_index, self.__get_function_from_string(unit_activation_function), *unit_activation_function_args))
            
        # All Neurons dependecies of successors and predecessors are established inside the objects
        for node in topology:
            unit_index = int(node)
            unit_type = topology[node][0]
            
            if not unit_type.startswith('output'): # Output units have no successors
                unit_successors = [units[u] for u in topology[node][3]]
                units[unit_index].extend_successors(unit_successors)

        self.neurons = units
        self.n_neurons = len(self.neurons)
        # All Neurons weights vectors are initialised
        self.__init_weights()
    
    def __topological_sort_util(self, index:int, visited:list, ordered:list):
        '''
        Recursive function that builds the (inverse-)topological order and update the ordered list self.neurons
        
        index: int
            the index of the neuron to visit
        visited: list
            the list of visited neurons
        ordered: list
            the list of ordered neurons (inverse topological order)

        return: -
        '''
        # if the neuron is been already visited then is before in the list
        visited[index] = True
        
        if self.neurons[index].type != 'output':
            for succ in self.neurons[index].successors:
                if not visited[succ.index]:
                    self.__topological_sort_util(succ.index, visited, ordered)
        
        ordered.append(self.neurons[index])
    
    def __topological_sort(self):
        '''
        Sort on a topological order the neurons of the Neural Network, this order is needed to peerform the correct
        feed forwardng and the backpropagation (that use the inverse order)
        
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
        
        topology: dict
            the graph structure is described by a dictionary that has a key for each unit in the network, 
            and for each key contains a list of unit type (input, hidden, output), activation function, parameters of activation functions
            and list of nodes where an outgoing arc terminates.
            
            eg: {'0': ['input', 'None', [], ['2', '3', '4']], 
                 '1': ['input', 'None', [], ['2', '3', '4']],
                 '2': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '3': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '4': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '5': ['output', 'identity', [fun_args...], []],
                 '6': ['output', 'identity', [fun_args...], []]}

        rand_range_min: float
            minimum value for random weights initialisation range
        rand_range_max: float
            maximum value for random weights initialisation range
        fan_in: bool
            if the weights'initialisation should also consider the Neuron's fan-in
        random_state: int
            the seed to create the radom generator, use this to make replicable behavior
        
        return: -
        '''

        # all these variables are computed correctly later, here are only initialized
        self.input_size = 0
        self.output_size = 0
        
        self.topology = topology

        self.random_generator = np.random.default_rng(random_state)
        self.rand_range_min = rand_range_min
        self.rand_range_max = rand_range_max
        self.fan_in = fan_in
        
        self.__construct_from_dict(topology)
        self.__topological_sort() # The NN keeps its neurons in topological order
    
    def __str__(self):
        '''
        Return a string that describe the internal state of the neural network.
        For semplicity only the neuron's weights are showed.
        
        param: -
        
        return: str
            the description of the internal rapresentation
        '''
        # is showed for each newron:
        #   id: index
        #   w: the weights
        #   i/o: number of predecessors/number of successors
        ret = ''
        for i in self.neurons[self.input_size:-self.output_size]:
            ret += 'id: '+ str(i.index) +  ' w: ' + str(i.w) + ' ' + ' i/o: ' + str(i.n_predecessors) + '/' + str(i.n_successors) + '\n'
        for i in self.neurons[-self.output_size:]:
            ret += 'id: '+ str(i.index) +  ' w: ' + str(i.w) + ' ' + ' i/o: ' + str(i.n_predecessors) + '\n'
        return ret

    def predict(self, input:np.array):
        '''
        Compute the output of the network given an input vector
        
        input: np.array
            the input vector for the model
        
        return: np.array
            the network's output vector
        '''
        # first the output vector is created
        output_vector = np.empty(self.output_size)
        
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
        '''
        Compute the output of the network given multiples input vectors
        
        input: np.array
            the inputs vector for the model
        
        return: np.array
            the network's output vector
        '''
        # for each input pattern the function self.predict is used and the results is stored and then returned
        output = np.empty((len(input), self.output_size))
        for i, el in enumerate(input):
            output[i] = self.predict(el)
        return output
          
    def __backpropagation(self, target:np.array):
        '''
        Compute the Backpropagation training algorithm on the NN for a single training pattern.
        Is called strictly after the self.predict, when every neuron has stored the last predict correctly.
        
        target: np.array
            the target vector for the Backprogation iteration
        return: -
        '''

        # The error is calculated in the ouput units and propagated backwards
        for i, neuron in enumerate(reversed(self.neurons[-self.output_size:])):
            neuron.backward(target[-(i+1)])
        # The hidden units will now calculate their errors based on the signal propagated by their successors in the nn
        for neuron in reversed(self.neurons[self.input_size:-self.output_size]):
            neuron.backward()
                
    def train(self, 
              training_set:np.ndarray, 
              validation_set:np.ndarray = None, 
              
              batch_size:int = 1, 
              max_epochs:int = 512, 
              min_epochs: int = 0,
              retrainig_es_error: float = -1.0, # off
              patience: int = 5, 
              error_increase_tolerance:float = 0.0001, 
            
              lambda_tikhonov:float = 0.0, # off
              
              adamax:bool = False,
              adamax_learning_rate:float = 0.01,
              exp_decay_rate_1:float = 0.9,
              exp_decay_rate_2:float = 0.999,
              
              learning_rate:float = 0.01,
              lr_decay_tau:int = 0, # off
              eta_tau:float = 0.0, # off
              alpha_momentum:float = 0.0, # off
              nesterov:bool = False,
              
              metrics:list=[], 
              collect_data:bool=True, 
              collect_data_batch:bool=False, 
              verbose:bool=True,
              
              supp_dataset=None):
        '''
        Compute the Backpropagation training algorithm on the NN for given training samples and hyperparameters
        
        Parameters
        ----------
        training_set: np.ndarray
            set of samples (pattern-target) for supervised learning that is used for the training
        validation_set: np.ndarray
            set of samples (pattern-target) for supervised learning that is used for valdation purposes:
                -> computing validation error in the training process
                -> computing best_validation_training_error, the training error when the validation 
                    error in min during training
                -> stopping the training if the network reached convergence
        batch_size: int
            parameter which determines the amount of training samples consumed in each iteration of the algorithm
                -> 1: Online
                -> 1 < batch_size < len(TR): Minibatch with minibatch size equals to batch_size
                -> len(TR): Batch
        max_epochs: int
            the maximum number of epochs on which the algorithm will iterate
        min_epochs: int
            the minimum number of epochs on which the algorithm will iterate, 
            is used to decide when to start using the patience
        retrainig_es_error: float
            the network stops the training when the error on training set falls below this value, 
            used in retraining
        patience: int
            Number of successive iterations in which the validation error increases before to terminate the training
        error_increase_tollerance: float
            tollerance used to decide if the validation error is increasing between two iterations
        lambda_tikhonov: float
            Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        
        adamax: boool
            If True the adamax optimizer is used instead of the standard one so learnig_rate, lr_decay-tau, eta_tau, 
                alpha_momentum,
            nesterov_momentum are ignored.
            If False the standard optimized is used and adamax_learning_rate, exp_decay_rate_1, exp_decay_rate_2 is ignored.
        adamax_learning_rate: float
            The learning rate used in update_weights_adamax inside the neurons, the step multiplyed to the gradient 
            and added to the weights at each iteration.
        exp_decay_rate_1: float
            The first exponential decay factor used in Adamax optimizer, applied to the momentum and learning rate
        exp_decay_rate_2: float
            The second exponential decay factor used in Adamax optimizer, applied to the infinity norm
        
        learning_rate: float
            The learning rate used in update_weights inside the neurons, the step multiplyed to the gradient and 
            added to the weights at each iteration.
        lr_decay_tau: int
            Parameter that controls the decay of the learning rate in the standard optimizer, after the actual learning
            epoch is greater then this value the learning rate is fixed at eta_tau
        eta_tau: float 
            Parameter that controls the decay of the learning rate in the standard optimizer, after the actual learning
            epoch is greater then lr_decay_tau the learning rate is fixed at this value
        alpha_momentum: float
            The influence of the momentum (previous iteration weights update) in the weights update
        nesterov: bool
            If the nesterov momentum technique is used instead of the classical momentum
            
        metrics: list
            The list of callable function used to compute the errors during the training, related to training set and validation set
        collect_data: bool 
            If to collect datas at each epoch during training, if True the training is slightly slower
        collect_data_batch: bool
            If to collect datas at the end of each batch during traning, if True  the training is severely slowed
        verbose: bool
            If to print some information in the standard output during training
            
        supp_dataset: np.ndarray
            Used to gather information regarding the dataset usend in the ensimbling
            
        Returns
        -------
        return: dict
            All the stats gathered during training.
            If collect_data is set to False it contains only essential stats and inputs of the training method
            If collect_data is set to True it contains stats computed at every epochs such as errors, computational time ...
            If collect_data_batch is set to True it contains also stats computed at every epochs such as the neuron's weights,
                errors ...
                
        See Also
        --------
        The neuron's structure and hidden and output neuron's update_weights functions
        '''

        # function used to iterate over the training set and tanke batch_size patterns at each time
        def circular_index(a, start, stop): 
            '''
            start is included
            stop is NOT included
            '''
            # fn to convert your start stop to a wrapped range
            if stop<=start:
                stop += len(a)
            return np.arange(start, stop)%len(a)

        # initializing every variables with the correct value
        epochs = 0
        exhausting_patience = patience
        last_error_increase_percentage = -1
        training_err = np.inf
        last_error = np.inf
        new_error = np.inf
        tr_err = np.inf
        training_set_length = len(training_set)
        batch_index = 0
        # simple check to adjust bad input values
        if batch_size > training_set_length: batch_size = training_set_length
        
        # variables used in retrain to stop at the right training error
        retrainig_es = True
        retrainig_es_tollerance = 0.1
        retrainig_es_error = retrainig_es_error + retrainig_es_error*retrainig_es_tollerance
        # initializing the dict where collect stats of the training
        stats = {
            # -- input stats --
            'training_set_len':training_set_length,
            'minibatch_size':batch_size,
            'max_epochs':max_epochs,
            'retrainig_es_error':retrainig_es_error,
            'error_increase_tolerance':error_increase_tolerance,
            'patience':patience,
            'min_epochs':min_epochs,
            'learning_rate':learning_rate,
            'lr_decay_tau':lr_decay_tau,
            'eta_tau':eta_tau,
            'lambda_tikhonov':lambda_tikhonov,
            'alpha_momentum':alpha_momentum,
            
            'nesterov':nesterov,
            
            'adamax':adamax,
            'adamax_learning_rate':adamax_learning_rate,
            'exp_decay_rate_1':exp_decay_rate_1,
            'exp_decay_rate_2':exp_decay_rate_2,

            # --early stopping stats--
            'best_validation_training_error': np.Inf,

            # -- training stats --
            # epoch stats
            'epochs':0,      
        }
        
        if collect_data: 
            # take training time for the batch
            stats['total_train_time'] = datetime.datetime.now() - datetime.datetime.now()
            stats['mean_epoch_train_time'] = 0
            stats['units_weights'] = {}
            stats['units_weights_batch'] = {}
            
            stats['training_pred'] = []
            stats['validation_pred'] = []
            
            # initializing lists to collect data
            for mes in metrics:
            # epoch stats
                stats['training_' + mes.__name__] = []
                stats['validation_' + mes.__name__] = []
                
                
                if collect_data_batch:
                    # batch stats
                    stats['training_batch_' + mes.__name__] = []
                    stats['validation_batch_' + mes.__name__] = []
                    
            for unit in self.neurons[self.input_size:]:
                # epoch stats
                stats['units_weights'][unit.index] = []
                if collect_data_batch:
                    # batch stats
                    stats['units_weights_batch'][unit.index] = []
            
            if verbose: print('starting values: ', stats)
            start_time = datetime.datetime.now()
            
        # start training cycle
        while (epochs < max_epochs) and (exhausting_patience > 0) and retrainig_es:
            # batch iteration
            for sample in training_set[circular_index(training_set, batch_index, (batch_index + batch_size) % training_set_length)]:
                # for each pattern the prediction is computed, and the values is forwarded in the net
                self.predict(sample[:self.input_size])
                # then the backpropagation is done to backpropagate the error and store information for the weights update
                self.__backpropagation(sample[self.input_size:])

            # after every batch the weighs are update accordingly to the optimized chosen
            if adamax:
                # Adamax
                for neuron in self.neurons[self.input_size:]:
                    neuron.update_weights_adamax(adamax_learning_rate, exp_decay_rate_1, exp_decay_rate_2, lambda_tikhonov)
            else:
                # Standard
                for neuron in self.neurons[self.input_size:]:
                    neuron.update_weights(learning_rate, lr_decay_tau, eta_tau, lambda_tikhonov, alpha_momentum, nesterov)

            # stats for every batch
            # to avoid stupidly high starting values the first epochs is skipped
            if collect_data and collect_data_batch and epochs > 0: 
                # computing errors
                for mes in metrics:
                    stats['training_batch_' + mes.__name__].append(mes(self.predict_array(training_set[:,:self.input_size]), training_set[:,self.input_size:]))
                    if not(validation_set is None):
                        stats['validation_batch_' + mes.__name__].append(mes(self.predict_array(validation_set[:,:self.input_size]), validation_set[:,self.input_size:]))
                # storing unit's weights
                for unit in self.neurons[self.input_size:]:
                    stats['units_weights_batch'][unit.index].append(list(unit.w))

            batch_index += batch_size
            # after every batch is checked if an epoch is passed
            if batch_index >= training_set_length:
                # end of the epoch
                
                # step parameter in neuron are updated
                for neuron in self.neurons[self.input_size:]:
                    neuron.increase_steps()
                # patience related computation, usefoul to check if to stop the training
                if epochs > min_epochs and last_error_increase_percentage > error_increase_tolerance:
                    exhausting_patience -= 1
                else:
                    exhausting_patience = patience

                epochs += 1
                batch_index = batch_index%training_set_length

                # the training error is computed
                training_err = ErrorFunctions.mean_squared_error(self.predict_array(training_set[:,:self.input_size]), training_set[:,self.input_size:])
                # if validation_set is given we compute the error to check if we are ath the minimum and store the 
                # training error of this iteration
                if not(validation_set is None) and (error_increase_tolerance > 0): # if True compute Early Stopping
                    new_error = ErrorFunctions.mean_squared_error(self.predict_array(validation_set[:,:self.input_size]), validation_set[:,self.input_size:])
                    if new_error > last_error:
                        last_error_increase_percentage = (new_error - last_error)/last_error    
                    else:
                        last_error_increase_percentage = 0
                        
                        stats['best_validation_training_error'] = min(stats['best_validation_training_error'], training_err)
                    last_error = new_error
                # in retraining this stops the learning when the validation tends to be the minimum, is calculated correctly
                # in a previous training
                if training_err < retrainig_es_error:
                        retrainig_es = False
                
                # stats for every epoch
                if collect_data:
                    # take training time for the epoch
                    end_time = datetime.datetime.now()   
                    if collect_data:
                        stats['total_train_time'] += (end_time-start_time)

                    # computing every error and printing some information if verbose is True
                    if verbose: metrics_to_print = ''
                    
                    pred_tr = self.predict_array(training_set[:,:self.input_size])
                    if not(supp_dataset is None):
                        pred_dsa = self.predict_array(supp_dataset[:,:self.input_size])
                        stats['training_pred'].append(pred_dsa)
                    
                    val_err = -1
                    if not(validation_set is None):
                        pred_val = self.predict_array(validation_set[:,:self.input_size])
                        stats['validation_pred'].append(pred_val)
                        
                    for mes in metrics:     
                        tr_err = mes(pred_tr, training_set[:,self.input_size:])
                        stats['training_' + mes.__name__].append(tr_err)
                        if not(validation_set is None):
                            val_err = mes(pred_val, validation_set[:,self.input_size:])
                            stats['validation_' + mes.__name__].append(val_err)
                            
                        if verbose: 
                            metrics_to_print += '| ' +mes.__name__ + ': tr=' + str(tr_err) + ' val=' + str(val_err) + ' | '
                        
                    for unit in self.neurons[self.input_size:]:
                        stats['units_weights'][unit.index].append(list(unit.w))

                    if verbose: print('[' + str(epochs) + '/' + str(max_epochs) + '] tr time:', end_time-start_time, metrics_to_print)
                    # take training time for the batch
                    start_time = datetime.datetime.now()
        
        # if nesterov is exploited, the weight needs to be modified after the final training iteration, check the neuron's
        # update_weights method
        if nesterov:
            for neuron in self.neurons[self.input_size:]:
                neuron.w += alpha_momentum * neuron.old_weight_update * nesterov 
            
        # final stats gathering
        stats['epochs'] = epochs
        if collect_data:
            stats['mean_epoch_train_time'] = stats['total_train_time']/stats['epochs']
        return stats
    
