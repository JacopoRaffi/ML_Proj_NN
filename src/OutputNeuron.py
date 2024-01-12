import numpy as np
from ActivationFunctions import ActivationFunctions

from ABCNeuron import ABCNeuron

class OutputNeuron(ABCNeuron):
    '''
    Implementation of an output neuron composing the NN
    
    Attributes
    ----------
    index : int
        the index of the neuron in the NN
    type : str
        the type of the neuron
    predecessors : list of neurons
        list of neurons sending their outputs in input to this neuron
    n_predecessors: int
        number of units linked as predecessors to this neuron
    w : array of float
        weights vector
    f : callable
        activation function
    f_parameters : list of float
        the list for the additional (optional) parameters of the activation function
    net : float
        inner product between the weight vector and the unit's input at a given iteration
    last_predict : float
        output of the neuron (instance variable exploited for predictions out of training)
   delta_error : float
        the last error signal related to the unit's output
    partial_weight_update : array of float
        the partial sum (on the minibatch) that will compose the DeltaW weight update value
    old_weight_update : array of float
        the old weight update value
    '''

    def __init__(self, index:int, activation_fun:callable = ActivationFunctions.sigmoid,  *args):
        '''
        Neuron initialisation
        
        param index: the index of the neuron in the NN
        param n_input: the number of inputs receivable by the Neuron
        param activation_fun: the Neuron's actviation function
        param rand_range_min: minimum value for random weights initialisation range
        param rand_range_max: maximum value for random weights initialisation range
        param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        param args: additional (optional) parameters of the activation function

        return: -
        '''
        self.index = index
        self.type = 'output'
        self.predecessors = [] # list of neurons sending their outputs in input to this neuron
        self.n_predecessors = 0
        self.w = np.array([]) # weights vector (initialised later)
        self.f = activation_fun # activation function
        self.f_parameters = list(*args) # creates the list for the additional (optional) parameters of the activation function
        self.net = 0.0 # inner product between the weight vector and the unit's input at a given iteration

        self.last_predict = 0.0 # output of the neuron (instance variable exploited for predictions out of training)
        self.delta_error = 0.0 # the error signal calculated in the backpropagation
        self.partial_weight_update = np.array([]) # the partial sum (on the minibatch) that will compose the DeltaW weight update value
        self.old_weight_update = np.array([]) #the old weight update value DeltaW

        # the creation of the variable is not necessary because can be created in any moment, just having the istance of the object but
        # the None value can help in preventing error, also resetting the variable can help in this sense
        
    
    #TODO: rivedere
    #def add_nesterov_momentum(self, alpha_momentum:float = 0.0):
        '''
        Updates the weight vector (w) of the Neuron with Nesterov's Momentum
        this update should be done before the next minibatch learning iteration
        
        :param alpha_momentum: Nertov's Momentum Hyperparameter
        :return: -
        '''
        
        #self.w = self.w + alpha_momentum*self.old_weight_update
    
    def update_weights(self, learning_rate:float = 1, lambda_tikhonov:float = 0.0, alpha_momentum:float = 0.0):
        '''
        Updates the weight vector (w) of the Neuron
        
        param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        param lambda_tikhonov: Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        param alpha_momentum: Momentum Hyperparameter

        return: -
        '''

        # here we compute the weight update if the momentum is used or not
        weight_update = (learning_rate * self.partial_weight_update) + (alpha_momentum * self.old_weight_update)
        
        # the weight_update value is calculated separated from Tikhonov Regularization for code/concept cleanliness
        weight_update = weight_update - (lambda_tikhonov * self.w)

        self.w += weight_update
        self.old_weight_update = weight_update
        self.partial_weight_update = np.zeros(self.n_predecessors + 1)
        
    def initialise_weights(self, rand_range_min:float, rand_range_max:float, fan_in:bool, random_generator:np.random.Generator):
        '''
        Initialises the Neuron's weights vector (w)
        Updates the unit's numbers of predecessors and successors (the network has already been completely linked together)
        
        param rand_range_min: minimum value for random weights initialisation range
        param rand_range_max: maximum value for random weights initialisation range
        param fan_in: if the weights'initialisation should also consider the Neuron's fan-in

        return: -
        '''
        self.n_predecessors = len(self.predecessors)
        self.w = random_generator.uniform(rand_range_min, rand_range_max, self.n_predecessors + 1) # bias
        self.old_weight_update = np.zeros(self.n_predecessors + 1) # bias
        self.partial_weight_update = np.zeros(self.n_predecessors + 1) # bias
        if fan_in:
            self.w = self.w * 2/(self.n_predecessors + 1)
        
    def forward(self):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units
        
        param input: Neuron's input vector

        return: the Neuron's output
        '''
        input = np.empty(self.n_predecessors + 1) #np.zeros(self.n_predecessors + 1)
        input[0] = 1
        for index, p in enumerate(self.predecessors):
            input[index + 1] = p.last_predict
                
        self.net = np.inner(self.w, input)
        self.last_predict = self.f(self.net, *self.f_parameters)
        
        #print(self.index, self.last_predict)
        return self.last_predict
    
    def backward(self, target:float):
        '''
        Calculates the Neuron's error contribute for a given learning pattern
        Calculates a partial weight update for the Neuron (Partial Backpropagation)
        
        param target: the Output Unit's target value

        return: -
        '''
        
        predecessors_outputs = np.zeros(self.n_predecessors + 1) # bias
        
        self.delta_error = (target - self.last_predict) * ActivationFunctions.derivative(self.f, self.net, self.f_parameters)
        
        predecessors_outputs[0] = 1 # bias
        for index, p in enumerate(self.predecessors):
            predecessors_outputs[index + 1] = p.last_predict # bias
            if p.type != "input":
                p.accumulate_weighted_error(self.delta_error, self.w[index + 1]) # bias
        
        self.partial_weight_update += (self.delta_error * predecessors_outputs)
    
    def add_predecessor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's predecessors
        
        param neuron: the Neuron to add to the list of predecessors
        
        return: -
        '''
        self.predecessors.append(neuron)


    # TODO: eliminare?
    #def extend_predecessors(self, neurons:list):
        '''
        Extends the list of the Neuron's predecessors
        
        :param neurons: the list of Neurons to add to the list of predecessors
        :return: -
        '''
        #self.predecessors.extend(neurons)
