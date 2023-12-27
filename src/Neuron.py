import numpy
from ActivationFunctions import ActivationFunctions
import math

class Neuron:
    '''Implementation of a neuron composing the NN'''

    def __init__(self, n_input:int, activation_fun:callable, rand_range_min:float, rand_range_max:float, fan_in:bool, *args):
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
        Calculates the Neuron's output on the inputs incoming from the other units
        
        :param input: Neuron's input vector
        :return: the Neuron's output
        '''
        return self.f(numpy.inner(self.w, input), *self.f_parameters)
            

if __name__ == "__main__":
    fun = ActivationFunctions.softplus
    neuron = Neuron(11, fun, -0.7, 0.7, False, 2)

    w = neuron.w
    x = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = numpy.inner(w, x)
    print("INPUT: ", y)
    print("NEURON: ", neuron.forward(x))
    print("NOT NEURON: ", fun(y, 2))