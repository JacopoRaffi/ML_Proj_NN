import numpy
from ActivationFunctions import ActivationFunctions

class Neuron:
    '''Implementation of a neuron composing the NN
    
    Attributes
    ----------
    predecessors : list
        list of neurons sending their outputs in input to this neuron
    successors : list
        list of neurons receiving this neuron's outputs
    w : array of float
        weights vector
    f : callable
        activation function
    f_parameters : ??
        creates the list for the additional (optional) parameters of the activation function

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
    

    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors
        
        :param neuron: the Neuron to add to the list of successors
        :return: -
        '''
        self.successors.append(neuron)

    def add_predecessor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's predecessors
        
        :param neuron: the Neuron to add to the list of predecessors
        :return: -
        '''
        self.predecessors.append(neuron)
            

if __name__ == "__main__":
    fun = ActivationFunctions.gaussian
    neuron = Neuron(11, -0.7, 0.7, False, fun, 2)

    w = neuron.w
    x = numpy.random.rand(11)
    y = numpy.inner(w, x)
    print("INPUT: ", y)
    print("NEURON: ", neuron.forward(x))
    print("NOT NEURON: ", fun(y, 2))