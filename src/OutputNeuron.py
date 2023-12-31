import numpy
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
    last_delta_error : float
        the last error signal related to the unit's output
    partial_weight_update : array of float
        the partial sum (on the minibatch) that will compose the DeltaW weight update value
    old_weight_update : array of float
        the old weight update value DeltaW
    '''

    def __init__(self, index:int, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = False, activation_fun:callable = ActivationFunctions.sigmoid,  *args):
        '''
        Neuron initialisation
        
        :param index: the index of the neuron in the NN
        :param n_input: the number of inputs receivable by the Neuron
        :param activation_fun: the Neuron's actviation function
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :param args: additional (optional) parameters of the activation function
        :return: -
        '''
        self.index = index
        self.type = 'output'
        self.predecessors = [] # list of neurons sending their outputs in input to this neuron
        self.n_predecessors = 0
        self.w = numpy.array([]) # weights vector (initialised later)
        self.f = activation_fun # activation function
        self.f_parameters = list(*args) # creates the list for the additional (optional) parameters of the activation function
        self.net = 0.0 # inner product between the weight vector and the unit's input at a given iteration

        self.last_predict = 0.0 # output of the neuron (instance variable exploited for predictions out of training)
        self.last_delta_error = 0.0 # the list of error signals calculated in the backpropagation
        self.partial_weight_update = numpy.array([]) # the partial sum (on the minibatch) that will compose the DeltaW weight update value
        self.old_weight_update = numpy.array([]) #the old weight update value DeltaW
        # TODO WTF??
        # the creation of the variable is not necessary because can be created in any moment, just having the istance of the object but
        # the None value can help in preventing error, also resetting the variable can help in this sense
        
    
    def add_nesterov_momentum(self, alpha_momentum:float = 0.0):
        '''
        Updates the weight vector (w) of the Neuron with Nesterov's Momentum
        this update should be done before the next minibatch learning iteration
        
        :param alpha_momentum: Nertov's Momentum Hyperparameter
        :return: -
        '''
        
        self.w = self.w + alpha_momentum*self.old_weight_update
    
    def update_weights(self, learning_rate:float = 1, lambda_tikhonov:float = 0.0, alpha_momentum:float = 0.0, nesterov_momentum:bool = False):
        '''
        Updates the weight vector (w) of the Neuron
        
        :param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        :param lambda_tikhonov: Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        :param alpha_momentum: Momentum Hyperparameter
        :return: -
        '''
        
        tmp_old_weight_update = self.partial_weight_update
        
        if nesterov_momentum: # if we exploited Nesterov's Momentum we need now to consider for the weight update the original w (without Nesterov)
            self.w = self.w - (alpha_momentum * self.old_weight_update)
           
        # the weight_update value is calculated separated from Tikhonov Regularization for code/concept cleanliness
        self.partial_weight_update = (learning_rate * self.partial_weight_update) + (alpha_momentum * self.old_weight_update)
        
        # if the algorithm is exploiting momentum the weight update value of the iteration corresponds to the previous formula
        if alpha_momentum != 0:
            self.old_weight_update = self.partial_weight_update
        else: # if not, it should not be influenced by learning rate
            self.old_weight_update = tmp_old_weight_update
    
        self.w = self.w + self.partial_weight_update - (lambda_tikhonov * self.w)
        
        
    def initialise_weights(self, rand_range_min:float, rand_range_max:float, fan_in:bool):
        '''
        Initialises the Neuron's weights vector (w)
        Updates the unit's numbers of predecessors and successors (the network has already been completely linked together)
        
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :return: -
        '''
        self.n_predecessors = len(self.predecessors)
        self.w = numpy.random.uniform(rand_range_min, rand_range_max, self.n_predecessors)
        self.old_weight_update = numpy.zeros(self.n_predecessors)
        self.partial_weight_update = numpy.zeros(self.n_predecessors)
        if fan_in:
            self.w = self.w * 2/self.n_predecessors
        

    def forward(self):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units, adding the output in the output_list
        
        :param input: Neuron's input vector
        :return: the Neuron's output
        '''
        input = numpy.zeros(self.n_predecessors)
        index = 0
        for p in self.predecessors:
            input[index] = p.last_predict
            index += 1
                
        self.net = numpy.inner(self.w, input)
        output_value = self.f(self.net, *self.f_parameters)

        self.last_predict = output_value
            
        return output_value
    
    def backward(self, target:float):
        '''
        Calculates the Neuron's error contribute for a given learning pattern
        Calculates a partial weight update for the Neuron (Partial Backpropagation)
        
        :param input: Neuron's input vector
        :return: the Neuron's output
        '''
        
        predecessors_outputs = numpy.zeros(self.n_predecessors)
        index = 0
        
        for p in self.predecessors:
            predecessors_outputs[index] = p.last_predict
        
        self.delta_error = (target - self.last_predict) * ActivationFunctions.derivative(self.f, self.net, self.f_parameters)
        self.partial_weight_update = self.partial_weight_update + self.delta_error * predecessors_outputs
        
    def add_predecessor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's predecessors
        
        :param neuron: the Neuron to add to the list of predecessors
        :return: -
        '''
        self.predecessors.append(neuron)

    def extend_predecessors(self, neurons:list):
        '''
        Extends the list of the Neuron's predecessors
        
        :param neurons: the list of Neurons to add to the list of predecessors
        :return: -
        '''
        self.predecessors.extend(neurons)

    def reset_neuron_history(self):
        '''
        Resets the history of the neuron by clearing the list of previous outputs and the delta error
        
        :return: -
        '''
        self.output_list = []
        self.delta_error = None 