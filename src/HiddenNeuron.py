import numpy
from ActivationFunctions import ActivationFunctions

from ABCNeuron import ABCNeuron

class HiddenNeuron(ABCNeuron):
    # TODO: può andare bene questo tipo?? considerando che sono parametri per funzioni di attivazione
    '''
    Implementation of an hidden neuron composing the NN
    
    Attributes
    ----------
    predecessors : list of neurons
        list of neurons sending their outputs in input to this neuron
    successors : list of neurons
        list of neurons receiving this neuron's outputs
    w : array of float
        weights vector
    f : callable
        activation function
    f_parameters : list of float or integers
        the list for the additional (optional) parameters of the activation function
    output_list : list of float
        list of the previous output of the neuron
    delta_error : float
        the delta error calculated in the backpropagation

    '''

    def __init__(self, n_input:int, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = True, activation_fun:callable = ActivationFunctions.sigmoid,  *args):
        '''
        Neuron initialisation
        
        :param n_input: the number of inputs receivable by the Neuron
        :param activation_fun: the Neuron's actviation function
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :param args: additional (optional) parameters of the activation function
        :return: -
        '''
        self.predecessors = [] # list of neurons sending their outputs in input to this neuron
        self.successors = [] # list of neurons receiving this neuron's outputs
        self.w = numpy.zeros(n_input) # weights vector
        self.initialise_weights(rand_range_min, rand_range_max, fan_in) # initialises the weights' vector
        self.f = activation_fun # activation function
        self.f_parameters = list(args) # creates the list for the additional (optional) parameters of the activation function

        self.output_list = [] # creates the output list
        self.delta_error = None # creates the delta error variable
        # the creation of the variable is not necessary because can be created in any moment, just having the istance of the object but
        # the None value can help in preventing error, also resetting the variable can help in this sense
    
    def update_weights(self, new_w:numpy.array):
        '''
        Updates the weight vector (w) of the Neuron
        
        :param new_w: the new weight vector of the Neuron
        :return: -
        '''
        self.w = new_w
        
        
    def initialise_weights(self, rand_range_min:float, rand_range_max:float, fan_in:bool):
        '''
        Initialises the Neuron's weights vector (w)
        
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :return: -
        '''
        self.w = numpy.random.uniform(rand_range_min, rand_range_max, self.w.size)
        if fan_in:
            self.w = self.w * 2/fan_in
        

    def forward(self, input:numpy.array):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units, adding the output in the output_list
        
        :param input: Neuron's input vector
        :return: the Neuron's output
        '''
        output_value = self.f(numpy.inner(self.w, input), *self.f_parameters)
        self.output_list.append(output_value)
    

    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors
        
        :param neuron: the Neuron to add to the list of successors
        :return: -
        '''
        self.successors.append(neuron)
    
    def extend_successors(self, neurons:list):
        '''
        Extends the list of the Neuron's successors
        
        :param neurons: the list of Neurons to add to the list of successors
        :return: -
        '''
        self.successors.extend(neurons)

    def extend_predecessors(self, neurons:list):
        '''
        Extends the list of the Neuron's predecessors
        
        :param neurons: the list of Neurons to add to the list of predecessors
        :return: -
        '''
        self.predecessors.extend(neurons)

    def add_predecessor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's predecessors
        
        :param neuron: the Neuron to add to the list of predecessors
        :return: -
        '''
        self.predecessors.append(neuron)

    def reset_neuron_history(self):
        '''
        Resets the history of the neuron by clearing the list of previous outputs and the delta error
        
        :return: -
        '''
        self.output_list = []
        self.delta_error = None

    
            

if __name__ == "__main__":
    import random

    for i in range(1000):
        fun = random.choice((ActivationFunctions.gaussian, ActivationFunctions.identity, ActivationFunctions.sigmoid,
                         ActivationFunctions.tanh, ActivationFunctions.softplus))
        neuron = HiddenNeuron(11, -0.7, 0.7, False, fun, 2)
        w = neuron.w
        x = numpy.random.rand(11)
        y = numpy.inner(w, x)
        forward = neuron.forward(x)
        fun_out = fun(y, 2)
        print("INPUT: ", y, "NEURON: ", forward, "NOT NEURON: ", fun_out, '\t', forward==fun_out)
    for i in range(25):
        w = neuron.w
        x = numpy.random.rand(11)
        y = numpy.inner(w, x)
        forward = neuron.forward(x)
        fun_out = fun(y, 2)
        print("INPUT: ", y, "NEURON: ", forward, "NOT NEURON: ", fun_out, '\t', forward==fun_out)
    print(neuron)